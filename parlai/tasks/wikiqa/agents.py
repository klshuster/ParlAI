#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.core.teachers import ParlAIDialogTeacher
from .build import build

import copy
import os


def _path(opt, filtered):
    # Build the data if it doesn't exist.
    build(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(opt['datapath'], 'WikiQA', dt + filtered + '.txt')


class FilteredTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '-filtered')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class UnfilteredTeacher(ParlAIDialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        opt['datafile'] = _path(opt, '')
        opt['parlaidialogteacher_datafile'] = ParlAIDialogTeacher._convert_from_fbdialog(
            opt['datafile']
        )
        super().__init__(opt, shared)


class DefaultTeacher(FilteredTeacher):
    pass
