#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import torch
from torch import nn
from parlai.agents.transformer import transformer as Transformer
from parlai.agents.transformer.modules import TransformerEncoder

word_embedding_file = '/private/home/kshuster/data/crawl-300d-2M.vec'
transformer_file = '/private/home/kshuster/data/redditbest.mdl'


class TransResNetModel(nn.Module):
    """
        Model for image dialog.
        There are two options for incorporating dialog history:
            1. Use the same transformer to encode dialog history and candidate
               responses; sum the image encoding, personality encoding, and
               dialog history encoding, and use that as query for candidate response
            2. (Multi-modal) Feed (something) into a separate transformer
               after computing the encoding; use that as query
    """

    @staticmethod
    def add_cmdline_args(argparser):
        Transformer.add_common_cmdline_args(argparser)

        agent = argparser.add_argument_group('CommentBattleModelURU task arguments')
        # The following override similar parameters in Transformer.add_common_cmdline
        agent.add_argument('-esz', '--embedding-size', type=int, default=300,
                           help='Size of all embedding layers')
        agent.add_argument('--ffn-size', type=int, default=300*4,
                           help='Hidden size of the FFN layers')
        agent.add_argument('--n-heads', type=int, default=6)
        agent.add_argument('--learn-positional-embeddings', type='bool', default=False)
        agent.add_argument('--embeddings-scale', type='bool', default=True)

        agent.add_argument('--image-features-dim', type=int, default=2048)
        agent.add_argument('--share-encoder', type='bool', default=False,
                           help='Whether to share the text encoder for the '
                           'labels and the dialog history')
        agent.add_argument('--hidden-dim', type=int, default=500)
        agent.add_argument('--num-layers-all', type=int, default=-1)
        agent.add_argument('--num-layers-text-encoder', type=int, default=1)
        agent.add_argument('--num-layers-image-encoder', type=int, default=1)
        agent.add_argument('--num-layers-multimodal-encoder', type=int, default=1)
        agent.add_argument('--dropout', type=float, default=0.4)
        agent.add_argument('--multimodal', type='bool', default=False,
                           help='If true, feed a query term into a separate '
                           'transformer prior to computing final rank '
                           'scores')
        agent.add_argument('--multimodal-combo', type=str,
                           choices=['concat', 'sum'], default='sum',
                           help='How to combine the encoding for the '
                           'multi-modal transformer')
        agent.add_argument('--encode-image', type='bool', default=True,
                           help='Whether to include the image encoding when '
                           'retrieving a candidate response')
        agent.add_argument('--encode-dialog-history', type='bool', default=True,
                           help='Whether to include the dialog history '
                           'encoding when retrieving a candidate response')
        agent.add_argument('--encode-personality', type='bool', default=True,
                           help='Whether to include the personality encoding '
                           'when retrieving a candidate response')

    def __init__(self, opt, personalities_list, dictionary):
        super().__init__()
        self.opt = opt
        self.use_cuda = not opt['no_cuda'] and torch.cuda.is_available()
        self.text_encoder_frozen = False

        # blank encoding (for concat)
        self.blank_encoding = torch.Tensor(opt['hidden_dim']).fill_(0).detach_()
        if self.use_cuda:
            self.blank_encoding = self.blank_encoding.cuda()

        # Encoders
        self.encode_image = opt.get('encode_image', True)
        self.encode_personality = opt.get('encode_personality', True)
        self.encode_dialog_history = opt.get('encode_dialog_history', True)

        # Multimodal Combiner (build if `--multimodal true` )
        self.build_multimodal(dictionary)

        # Label and Context (dialog history) Encoders
        self.build_encoders(dictionary)

        # Image Encoder
        self.build_image_encoder()

        # Persona Encoder
        self.build_persona_encoder(personalities_list)

    def build_persona_encoder(self, personalities_list):
        # Initialize personas dictionary
        self.persona_to_id = {}
        for i, p in enumerate(personalities_list):
            self.persona_to_id[p] = i
        self.persona_dim = len(personalities_list) + 1
        persona_layers = [nn.BatchNorm1d(self.persona_dim),
                          nn.Dropout(p=self.opt['dropout']),
                          nn.Linear(self.persona_dim, self.opt['hidden_dim'])]
        self.persona_encoder = nn.Sequential(*persona_layers)

    def build_image_encoder(self):
        nlayers_img = (self.opt['num_layers_all'] if self.opt['num_layers_all'] != -1
                       else self.opt['num_layers_image_encoder'])
        image_layers = [nn.BatchNorm1d(self.opt['image_features_dim']),
                        nn.Dropout(p=self.opt['dropout']),
                        nn.Linear(self.opt['image_features_dim'],
                        self.opt['hidden_dim'])]
        for _ in range(nlayers_img - 1):
            image_layers += [nn.ReLU(),
                             nn.Dropout(p=self.opt['dropout']),
                             nn.Linear(self.opt['hidden_dim'],
                                       self.opt['hidden_dim'])]
        self.image_encoder = nn.Sequential(*image_layers)

    def build_multimodal(self, dictionary):
        self.multimodal = self.opt.get('multimodal')
        if self.multimodal:
            self.multimodal_combo = self.opt.get('multimodal_combo', 'sum')
            nlayers_mm = (self.opt['num_layers_all'] if self.opt['num_layers_all'] != -1
                          else self.opt['num_layers_multimodal_encoder'])
            self.multimodal_encoder = TransformerEncoder(
                n_heads=self.opt['n_heads'],
                n_layers=nlayers_mm,
                embedding_size=self.opt['embedding_size'],
                ffn_size=self.opt['embedding_size']*4,
                vocabulary_size=len(dictionary),
                embedding=None,
                attention_dropout=self.opt['dropout'],
                relu_dropout=self.opt['dropout'],
                padding_idx=dictionary[dictionary.null_token],
                learn_positional_embeddings=self.opt.get('learn_positional_embeddings',
                                                         False),
                embeddings_scale=self.opt['embeddings_scale'],
                reduction=True)

    def build_encoders(self, dictionary):
        embeddings = nn.Embedding(len(dictionary),
                                  self.opt['embedding_size'])
        kwargs = {
            'n_heads': self.opt['n_heads'],
            'n_layers': self.opt['num_layers_text_encoder'],
            'embedding_size': self.opt['embedding_size'],
            'ffn_size': self.opt['embedding_size']*4,
            'vocabulary_size': len(dictionary),
            'embedding': embeddings,
            'attention_dropout': self.opt['dropout'],
            'relu_dropout': self.opt['dropout'],
            'padding_idx': dictionary[dictionary.null_token],
            'learn_positional_embeddings': self.opt.get('learn_positional_embeddings',
                                                        False),
            'embeddings_scale': self.opt['embeddings_scale'],
            'reduction': True
        }

        self.label_encoder = TransformerEncoder(**kwargs)
        if self.opt.get('share_encoder'):
            self.context_encoder = self.label_encoder
        else:
            embeddings_ctx = nn.Embedding(len(dictionary),
                                          self.opt['embedding_size'])
            kwargs['embedding'] = embeddings_ctx
            self.context_encoder = TransformerEncoder(**kwargs)
        self.additional_layer = LinearWrapper(
            self.opt['embedding_size'],
            self.opt['hidden_dim'],
            dropout=self.opt['dropout'])

    def forward(self, batch, cands, cands_type='batch', train=False):
        """
            Input: Batch
            Outputs: total_encoded: query encoding
        """
        # dialog history
        d_hist_encoded = self.forward_text_encoder(batch.text_vec,
                                                   dialog_history=True,
                                                   batchsize=len(batch.valid_indices))
        # images
        img_encoded = self.forward_image(batch.image)
        # personas
        pers_encoded = self.forward_persona(batch.personalities,
                                            len(batch.valid_indices))
        total_encoded = self.get_rep([img_encoded,
                                      d_hist_encoded,
                                      pers_encoded],
                                     batchsize=len(batch.valid_indices))
        return self.get_scores(total_encoded, cands, cands_type, train=train)

    def forward_persona(self, personas, bsz):
        if not self.encode_personality:
            if self.multimodal and self.multimodal_combo == 'concat':
                return self.blank_encoding
            return None
        if personas is None:
            personas = [''] * bsz
        pers_vec = torch.FloatTensor(len(personas), self.persona_dim).fill_(0)
        pers_list = [self.persona_to_id.get(p, 0) + 1 for p in personas]
        for i, index in enumerate(pers_list):
            pers_vec[i, index] = 1  # no personality corresponds to 0
        if self.use_cuda:
            pers_vec = pers_vec.cuda()
        return self.persona_encoder(pers_vec)

    def forward_text_encoder(self, texts, dialog_history=False, batchsize=None):
        if texts is None or (dialog_history and not self.encode_dialog_history):
            if (self.multimodal and self.multimodal_combo == 'concat' and
                dialog_history
            ):
                encoding = torch.stack([self.blank_encoding for _ in range(batchsize)])
                return encoding
            return None
        encoder = self.context_encoder if dialog_history else self.label_encoder
        texts_encoded = encoder(texts)
        if self.text_encoder_frozen:
            texts_encoded = texts_encoded.detach()
        texts_encoded = self.additional_layer(texts_encoded)
        return texts_encoded

    def forward_image(self, image_features):
        if image_features is None or not self.encode_image:
            if self.multimodal and self.multimodal_combo == 'concat':
                return self.blank_encoding
            return None
        if len(image_features) == 1:
            imgs = image_features[0]
        else:
            imgs = torch.stack(image_features)
        if self.use_cuda:
            imgs = imgs.cuda()
        return self.image_encoder(imgs)

    def get_rep(self, encodings, batchsize=None):
        if not self.multimodal:
            rep = self.sum(encodings)
        else:
            if self.multimodal_combo == 'sum':
                encodings = self.sum(encodings).unsqueeze(1)
            elif self.multimodal_combo == 'concat':
                encodings = self.cat(encodings)
            all_one_mask = torch.ones(encodings.size()[:2])
            if self.use_cuda:
                all_one_mask = all_one_mask.cuda()
            rep = self.multimodal_encoder(encodings, all_one_mask)
        if rep is None:
            rep = torch.stack([self.blank_encoding for _ in range(batchsize)])
        return rep

    def get_scores(self, query_vecs, cand_vecs, cands_type='batch', train=False):
        """
            combines a little bit of elect_best_comment and choose_topk to get
            the scores for candidates
        """
        if cands_type == 'inline':
            if not train:
                cand_vecs = [self.forward_text_encoder(c).detach() for c in cand_vecs]
            else:
                cand_vecs = [self.forward_text_encoder(c) for c in cand_vecs]
            scores = torch.cat(
                [
                    torch.mm(cand_vecs[idx], query_vecs[idx:idx+1, :].transpose(0, 1))
                    for idx in range(len(query_vecs))
                ],
                dim=1
            ).transpose(0, 1)
        else:
            cand_vecs = self.forward_text_encoder(cand_vecs)
            if not train:
                cand_vecs = cand_vecs.detach()
            scores = query_vecs.mm(cand_vecs.t())
        return scores

    def freeze_text_encoder(self):
        self.text_encoder_frozen = True

    def unfreeze_text_encoder(self):
        self.text_encoder_frozen = False

    ##################################################
    #     Util Funcs
    ##################################################
    def sum(self, addends):
        addends = [a for a in addends if a is not None]
        return sum(addends) if len(addends) > 0 else None

    def cat(self, tensors):
        tensors = [t for t in tensors if t is not None]
        return torch.cat([t.unsqueeze(1) for t in tensors], dim=1)


#########################################
#    Linear Wrapper
#########################################


class LinearWrapper(nn.Module):
    """
        Adds one linear layer on top of a module.
        This was designed for the transformer, since pretrained
        instance don't use dropout, and they are constrained to
        keep the dimension of the word embedding.
    """

    def __init__(self, in_dim, out_dim, dropout):
        super(LinearWrapper, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, input):
        return self.lin(self.dp(input))
