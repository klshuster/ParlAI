#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from parlai.agents.transformer.modules import TransformerEncoder, \
    create_position_codes, TransformerEncoderLayer


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
        self.build_multimodal()

        # Label and Context (dialog history) Encoders
        self.build_encoders(dictionary)

        # Image Encoder
        self.build_image_encoder()

        # personality Encoder
        self.build_personality_encoder(personalities_list)

    def build_personality_encoder(self, personalities_list):
        # Initialize personas dictionary
        self.personality_to_id = {}
        for i, p in enumerate(personalities_list):
            self.personality_to_id[p] = i
        self.personality_dim = len(personalities_list) + 1
        personality_layers = [nn.BatchNorm1d(self.personality_dim),
                              nn.Dropout(p=self.opt['dropout']),
                              nn.Linear(self.personality_dim, self.opt['hidden_dim'])]
        self.personality_encoder = nn.Sequential(*personality_layers)

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

    def build_multimodal(self):
        self.multimodal = self.opt.get('multimodal')
        if self.multimodal:
            self.multimodal_combo = self.opt.get('multimodal_combo', 'sum')
            nlayers_mm = (self.opt['num_layers_all'] if self.opt['num_layers_all'] != -1
                          else self.opt['num_layers_multimodal_encoder'])
            self.multimodal_encoder = MultimodalCombiner(
                n_heads=self.opt['n_heads'],
                n_layers=nlayers_mm,
                hidden_dim=self.opt['hidden_dim'],
                ffn_size=self.opt['embedding_size']*4,
                attention_dropout=self.opt['attention_dropout'],
                relu_dropout=self.opt['relu_dropout'],
                learn_positional_embeddings=self.opt.get('learn_positional_embeddings',
                                                         False),
                reduction=True)

    def build_encoders(self, dictionary):
        if self.opt['load_reddit']:
            self.load_from_reddit()
            return
        embeddings = nn.Embedding(len(dictionary),
                                  self.opt['embedding_size'])
        kwargs = {
            'n_heads': self.opt['n_heads'],
            'n_layers': self.opt['num_layers_text_encoder'],
            'embedding_size': self.opt['embedding_size'],
            'ffn_size': self.opt['embedding_size']*4,
            'vocabulary_size': len(dictionary),
            'embedding': embeddings,
            'attention_dropout': self.opt['attention_dropout'],
            'relu_dropout': self.opt['relu_dropout'],
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

    def load_from_reddit(self):
        reddit = torch.load('/private/home/kshuster/data/redditbest.mdl')

        kwargs = {
            'n_heads': reddit['transformer_n_heads'],
            'n_layers': reddit['transformer_n_layers'],
            'embedding_size': reddit['transformer_dim'],
            'ffn_size': reddit['transformer_dim']*4,
            'vocabulary_size': reddit['vocabulary_size'],
            'attention_dropout': self.opt['attention_dropout'],
            'relu_dropout': self.opt['relu_dropout'],
            'padding_idx': reddit['padding_idx'],
            'learn_positional_embeddings': self.opt.get('learn_positional_embeddings',
                                                        False),
            'embeddings_scale': self.opt['embeddings_scale'],
            'reduction': True
        }

        states = reddit['transformer_state']
        self.label_encoder = TransformerEncoder(**kwargs)
        import pdb; pdb.set_trace()

        transformed = {}
        for k, v in list(states.items()):
            k = k.replace('attentions.0', 'layers.0.attention')
            k = k.replace('attentions.1', 'layers.1.attention')
            k = k.replace('attentions.2', 'layers.2.attention')
            k = k.replace('attentions.3', 'layers.3.attention')
            k = k.replace('layer_norm1.0', 'layers.0.norm1')
            k = k.replace('layer_norm1.1', 'layers.1.norm1')
            k = k.replace('layer_norm1.2', 'layers.2.norm1')
            k = k.replace('layer_norm1.3', 'layers.3.norm1')
            k = k.replace('layer_norm2.0', 'layers.0.norm2')
            k = k.replace('layer_norm2.1', 'layers.1.norm2')
            k = k.replace('layer_norm2.2', 'layers.2.norm2')
            k = k.replace('layer_norm2.3', 'layers.3.norm2')
            k = k.replace('ffns.0', 'layers.0.ffn')
            k = k.replace('ffns.1', 'layers.1.ffn')
            k = k.replace('ffns.2', 'layers.2.ffn')
            k = k.replace('ffns.3', 'layers.3.ffn')
            transformed[k] = v
        self.label_encoder.load_state_dict(states)


    def forward(self, batch, cands, cands_type='batch', train=False):
        """
            :param batch: a Batch object (defined in torch_agent.py)
            :param cands: candidates for this batch
            :param cands_type: source of candidates for this batch, one of
                ['batch', 'inline', 'fixed']
            :param train: True if model is training

            :return: model scores for each item in the batch
        """
        # dialog history
        d_hist_encoded = self.forward_context(batch.text_vec,
                                              batchsize=len(batch.valid_indices))
        # images
        img_encoded = self.forward_image(batch.image)
        # personalities
        pers_encoded = self.forward_personality(batch.personalities,
                                                len(batch.valid_indices))
        # combined
        total_encoded = self.get_rep([img_encoded,
                                      d_hist_encoded,
                                      pers_encoded],
                                     batchsize=len(batch.valid_indices))
        return self.get_scores(total_encoded, cands, cands_type, train=train)

    def forward_personality(self, personalities, bsz):
        """
            :param personalities: [bsz] list of personalities (or None)
            :param bsz: batchsize

            :return: a [bsz, hidden_dim] FloatTensor of encoded personalities
        """
        if not self.encode_personality:
            if self.multimodal and self.multimodal_combo == 'concat':
                return self.blank_encoding
            else:
                return None

        if personalities is None:
            personalities = [''] * bsz
        pers_vec = torch.FloatTensor(len(personalities), self.personality_dim).fill_(0)
        pers_list = [self.personality_to_id.get(p, 0) + 1 for p in personalities]
        for i, index in enumerate(pers_list):
            pers_vec[i, index] = 1  # no personality corresponds to 0
        if self.use_cuda:
            pers_vec = pers_vec.cuda()
        return self.personality_encoder(pers_vec)

    def forward_context(self, context, batchsize=None):
        """
            :param context: a [bsz, seq_len] LongTensor of token indices
            :param batchsize: batch size

            :return: a [bsz, hidden_dim] FloatTensor of encoded context
        """
        if context is None or not self.encode_dialog_history:
            if self.multimodal and self.multimodal_combo == 'concat':
                return torch.stack([self.blank_encoding for _ in range(batchsize)])
            else:
                return None

        encoded = self.context_encoder(context)
        if self.text_encoder_frozen:
            encoded = encoded.detach()
        encoded = self.additional_layer(encoded)
        return encoded

    def forward_candidates(self, cands, batchsize=None):
        """
            :param cands:
        """
        if cands is None:
            return None

        encoded = self.label_encoder(cands)
        if self.text_encoder_frozen:
            encoded = encoded.detach()
        encoded = self.additional_layer(encoded)
        return encoded

    def forward_image(self, image_features):
        """
            :param image_features: a [bsz] list of [image_features_dim] FloatTensors

            :return: a [bsz, hidden_dim] FloatTensor of encoded images
        """
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
        """
            :param encodings: a 3-element list, where each element is either
                a tensor of dimension [bsz, hidden_dim]
                OR None
            :param batchsize: size of batch

            :return: a [bsz, hidden_dim] FloatTensor of encodings
        """
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
            :param query_vecs: a [bsz, hidden_dim] FloatTensor of example encodings
            :param cand_vecs: *dependent on cands_type*
                if 'batch', a [bsz, seq_len] LongTensor of token indices
                if 'inline', a [bsz, num_cands_per_example, seq_len]
                    LongTensor of token indices
            :param cands_type: source of candidates for this batch, one of
                ['batch', 'inline', 'fixed']
            :param train: whether this is a train batch

            :return: a [bsz, num_cands_per_example] FloatTensor of scores
        """
        if cands_type == 'inline':
            if not train:
                cand_vecs = [self.forward_candidates(c).detach() for c in cand_vecs]
            else:
                cand_vecs = [self.forward_candidates(c) for c in cand_vecs]
            scores = torch.cat(
                [
                    torch.mm(cand_vecs[idx], query_vecs[idx:idx+1, :].transpose(0, 1))
                    for idx in range(len(query_vecs))
                ],
                dim=1
            ).transpose(0, 1)
        else:
            cand_vecs = self.forward_candidates(cand_vecs)
            if not train:
                cand_vecs = cand_vecs.detach()
            scores = query_vecs.mm(cand_vecs.t())
        return scores

    def freeze_text_encoder(self):
        self.text_encoder_frozen = True

    def unfreeze_text_encoder(self):
        self.text_encoder_frozen = False

    ##################################################
    #     tensor combination functions
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
    """

    def __init__(self, in_dim, out_dim, dropout):
        super(LinearWrapper, self).__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.dp = nn.Dropout(dropout)

    def forward(self, input):
        return self.lin(self.dp(input))


########################################
# Multimodal Combiner                  #
########################################


class MultimodalCombiner(nn.Module):
    """
        Essentially a transformer, with no embeddings. See TransformerEncoder
        in parlai.agents.transformer.modules.
    """
    def __init__(
        self,
        n_heads,
        n_layers,
        hidden_dim,
        ffn_size,
        reduction=True,
        attention_dropout=0.0,
        relu_dropout=0.0,
        learn_positional_embeddings=False
    ):
        super().__init__()
        self.ffn_size = ffn_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.out_dim = hidden_dim
        self.dim = hidden_dim
        self.reduction = reduction
        assert hidden_dim % n_heads == 0, \
            'MM-Combiner dim must be multiple of n_heads'
        n_positions = 1024
        self.position_embeddings = nn.Embedding(n_positions, hidden_dim)
        if not learn_positional_embeddings:
            create_position_codes(
                n_positions, hidden_dim, out=self.position_embeddings.weight
            )
        else:
            nn.init.normal_(self.position_embeddings.weight, 0, hidden_dim ** -0.5)

        self.layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.layers.append(TransformerEncoderLayer(
                n_heads, hidden_dim, ffn_size, attention_dropout, relu_dropout
            ))

    def forward(self, tensor, mask):
        """
            :param tensor: a [bsz, seq_len, hidden_dim] FloatTensor
            :param mask: a [bsz, seq_len] ByteTensor filled with 1 when
                inside the sequence and 0 outside.

            :return: output: a [bsz, hidden_dim] FloatTensor of encodings
                     mask: the same as before
        """
        seq_len = tensor.size(1)
        positions = tensor.new(seq_len).long()
        positions = torch.arange(seq_len, out=positions).unsqueeze(0)
        tensor = tensor + self.position_embeddings(positions).expand_as(tensor)

        tensor *= mask.unsqueeze(-1).float()
        for i in range(self.n_layers):
            tensor = self.layers[i](tensor, mask)

        if self.reduction:
            divisor = mask.float().sum(dim=1).unsqueeze(-1).clamp(min=1e-20)
            output = tensor.sum(dim=1) / divisor
            return output
        else:
            output = tensor
            return output, mask
