
from reversible2.graph import Node
from reversible2.branching import CatChans, ChunkChans, Select
from reversible2.constantmemory import graph_to_constant_memory

from copy import deepcopy
from reversible2.graph import Node
from reversible2.blocks import dense_add_no_switch, conv_add_3x3_no_switch
from reversible2.rfft import RFFT, Interleave
import torch as th
from reversible2.splitter import SubsampleSplitter
from torch import nn
from reversible2.view_as import ViewAs


def add_bnorm_before_relu(feature_model):
    for m in feature_model.modules():
        for attr in ['Fm', 'Gm', 'FA', 'GA']:
            if hasattr(m, attr):
                old_children = list(m._modules[attr].children())
                new_children = []
                bnorm_inserted = False
                for c in old_children:
                    if c.__class__.__name__ == 'ReLU':
                        prev = new_children[-1]
                        if hasattr(prev, 'out_channels'):
                            num_features = prev.out_channels
                            new_children.append(
                                nn.BatchNorm2d(num_features=num_features))
                        else:
                            num_features = prev.out_features
                            new_children.append(
                                nn.BatchNorm1d(num_features=num_features))

                        bnorm_inserted = True
                    new_children.append(c)

                assert bnorm_inserted
                m._modules[attr] = nn.Sequential(*new_children)


def smaller_model(n_chans, n_time, final_fft, constant_memory):
    base_model = nn.Sequential(
            SubsampleSplitter(stride=[2, 1], chunk_chans_first=False),  # 4 x 32
            conv_add_3x3_no_switch(2 * n_chans, 32),
            conv_add_3x3_no_switch(2 * n_chans, 32),
            SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 8 x 16
            conv_add_3x3_no_switch(4 * n_chans, 32),
            conv_add_3x3_no_switch(4 * n_chans, 32),
            SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 16 x 8
            conv_add_3x3_no_switch(8 * n_chans, 32),
            conv_add_3x3_no_switch(8 * n_chans, 32),
        )
    base_model.cuda()

    branch_1_a = nn.Sequential(
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=False),  # 16 x 4
        conv_add_3x3_no_switch(8 * n_chans, 32),
        conv_add_3x3_no_switch(8 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 32 x 2
        conv_add_3x3_no_switch(16 * n_chans, 32),
        conv_add_3x3_no_switch(16 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 64 x 1
        ViewAs((-1, n_chans * 32, 1, 1), (-1, n_chans * 32)),
    )
    branch_1_b = deepcopy(branch_1_a)
    branch_1_a.cuda()
    branch_1_b.cuda()

    final_model = nn.Sequential(
        dense_add_no_switch(n_time * n_chans, 256),
        dense_add_no_switch(n_time * n_chans, 256),
        dense_add_no_switch(n_time * n_chans, 256),
        dense_add_no_switch(n_time * n_chans, 256),
    )
    if final_fft:
        final_model.add_module('final_fft', RFFT())
    final_model.cuda()
    o = Node(None, base_model)
    o = Node(o, ChunkChans(2))
    o1a = Node(o, Select(0))
    o1b = Node(o, Select(1))
    o1a = Node(o1a, branch_1_a)
    o1b = Node(o1b, branch_1_b)
    o = Node([o1a, o1b], CatChans())
    o = Node(o, final_model)
    if constant_memory:
        o = graph_to_constant_memory(o)
    feature_model = o
    return feature_model

def larger_model(n_chans, n_time, final_fft, constant_memory):
    base_model = nn.Sequential(
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=False),
        conv_add_3x3_no_switch(2 * n_chans, 32),
        conv_add_3x3_no_switch(2 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 4 x 128
        conv_add_3x3_no_switch(4 * n_chans, 32),
        conv_add_3x3_no_switch(4 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 8 x 64
        conv_add_3x3_no_switch(8 * n_chans, 32),
        conv_add_3x3_no_switch(8 * n_chans, 32))
    base_model.cuda();

    branch_1_a = nn.Sequential(
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=False),  # 8 x 32
        conv_add_3x3_no_switch(8 * n_chans, 32),
        conv_add_3x3_no_switch(8 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 16 x 16
        conv_add_3x3_no_switch(16 * n_chans, 32),
        conv_add_3x3_no_switch(16 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 32 x 8
        conv_add_3x3_no_switch(32 * n_chans, 32),
        conv_add_3x3_no_switch(32 * n_chans, 32),
    )
    branch_1_b = nn.Sequential(
        *(list(deepcopy(branch_1_a).children()) + [
            ViewAs((-1, 32 * n_chans, n_time // 64, 1),
                   (-1, (n_time // 2) * n_chans)),
            dense_add_no_switch((n_time // 2) * n_chans, 32),
            dense_add_no_switch((n_time // 2) * n_chans, 32),
            dense_add_no_switch((n_time // 2) * n_chans, 32),
            dense_add_no_switch((n_time // 2) * n_chans, 32),
        ]))
    branch_1_a.cuda();
    branch_1_b.cuda();

    branch_2_a = nn.Sequential(
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=False),  # 32 x 4
        conv_add_3x3_no_switch(32 * n_chans, 32),
        conv_add_3x3_no_switch(32 * n_chans, 32),
        SubsampleSplitter(stride=[2, 1], chunk_chans_first=True),  # 64 x 2
        conv_add_3x3_no_switch(64 * n_chans, 32),
        conv_add_3x3_no_switch(64 * n_chans, 32),
        ViewAs((-1, (n_time // 4) * n_chans, 1, 1),
               (-1, (n_time // 4) * n_chans)),
        dense_add_no_switch((n_time // 4) * n_chans, 64),
        dense_add_no_switch((n_time // 4) * n_chans, 64),
        dense_add_no_switch((n_time // 4) * n_chans, 64),
        dense_add_no_switch((n_time // 4) * n_chans, 64),
    )

    branch_2_b = deepcopy(branch_2_a).cuda()
    branch_2_a.cuda();
    branch_2_b.cuda();

    final_model = nn.Sequential(
        dense_add_no_switch(n_time * n_chans, 256),
        dense_add_no_switch(n_time * n_chans, 256),
        dense_add_no_switch(n_time * n_chans, 256),
        dense_add_no_switch(n_time * n_chans, 256),
    )
    if final_fft:
        final_model.add_module('final_fft', RFFT())
    final_model.cuda();
    o = Node(None, base_model)
    o = Node(o, ChunkChans(2))
    o1a = Node(o, Select(0))
    o1b = Node(o, Select(1))
    o1a = Node(o1a, branch_1_a)
    o1b = Node(o1b, branch_1_b)
    o2 = Node(o1a, ChunkChans(2))
    o2a = Node(o2, Select(0))
    o2b = Node(o2, Select(1))
    o2a = Node(o2a, branch_2_a)
    o2b = Node(o2b, branch_2_b)
    o = Node([o1b, o2a, o2b], CatChans())
    o = Node(o, final_model)
    if constant_memory:
        o = graph_to_constant_memory(o)
    feature_model = o
    return feature_model