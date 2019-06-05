#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import ParlAIDialogTeacher
import parlai.core.agents as core_agents
from .build import build

import copy
import os


def _path(exsz, task, opt, dt=''):
    # Build the data if it doesn't exist.
    build(opt)
    if dt == '':
        dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'bAbI', 'tasks_1-20_v1-2',
                        'en-valid{exsz}-nosf'.format(exsz=exsz),
                        'qa{task}_{type}.txt'.format(task=task, type=dt))


def mod_labels(ys, task):
    if ys is not None:
        # replace comma-labeled babi tasks with spaces
        # this is more friendly to our tokenizer which makes commas full tokens
        # this way models won't be penalized for not generating a comma
        if task == '8':
            # holding: labels like 'milk,cookies,football'
            # replace with spaces 'milk football cookies'
            ys = [y.replace(',', ' ') for y in ys]
        elif task == '19':
            # pathfinding: labels like 'n,e' or 's,w'
            # replace with spaces, 'n e'
            ys = [y.replace(',', ' ') for y in ys]

    return ys


# Single bAbI task (1k training).
class Task1kTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task1k:1')
        self.task_num = task.split(':')[2]
        opt['datafile'] = _path('', self.task_num, opt)
        opt['cands_datafile'] = _path('', task.split(':')[2], opt, 'train')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        opt['parlaidialogteacher_cands_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['cands_datafile']
        )
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        """Override to modify labels."""
        ex = super().get(episode_idx, entry_idx)
        ex['labels'] = mod_labels(ex['labels'], self.task_num)
        return ex

    def _load_cands(self, path):
        super()._load_cands(path)
        self.cands = mod_labels(self.cands, self.task_num)

# Single bAbI task (10k training).
class Task10kTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        task = opt.get('task', 'babi:Task10k:1')
        self.task_num = task.split(':')[2]
        opt['datafile'] = _path('-10k', self.task_num, opt)
        opt['cands_datafile'] = _path('-10k', task.split(':')[2], opt, 'train')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        opt['parlaidialogteacher_cands_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['cands_datafile']
        )
        super().__init__(opt, shared)

    def get(self, episode_idx, entry_idx=None):
        """Override to modify labels."""
        ex = super().get(episode_idx, entry_idx)
        ex['labels'] = mod_labels(ex['labels'], self.task_num)
        return ex

    def _load_cands(self, path):
        super()._load_cands(path)
        self.cands = mod_labels(self.cands, self.task_num)


# By default train on all tasks at once.
class All1kTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('babi:Task1k:%d' % (i + 1) for i in range(20))
        super().__init__(opt, shared)


# By default train on all tasks at once.
class All10kTeacher(core_agents.MultiTaskTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['task'] = ','.join('babi:Task10k:%d' % (i + 1) for i in range(20))
        super().__init__(opt, shared)


# By default train on all tasks at once.
class DefaultTeacher(All1kTeacher):
    pass
