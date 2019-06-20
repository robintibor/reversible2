from torch import nn
from reversible2.affine import AdditiveBlock
from reversible2.constantmemory import AdditiveBlockConstantMemory


def conv_add_3x3_no_switch(n_c, n_i_c):
    return conv_add_block_3x3(n_c, n_i_c, switched_order=False)


def  dense_add_no_switch(n_c, n_i_c):
    return dense_add_block(n_c, n_i_c, switched_order=False)


def conv_add_block_3x3(n_c, n_i_c, switched_order=True,):
    return AdditiveBlock(
        nn.Sequential(
            nn.Conv2d(n_c // 2, n_i_c, (3, 1), stride=1, padding=(1, 0),
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(n_i_c, n_c // 2, (3, 1), stride=1, padding=(1, 0),
                      bias=True)),

        nn.Sequential(
            nn.Conv2d(n_c // 2, n_i_c, (3, 1), stride=1, padding=(1, 0),
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(n_i_c, n_c // 2, (3, 1), stride=1, padding=(1, 0),
                      bias=True)),
    switched_order=switched_order)


def dense_add_block(n_c, n_i_c, switched_order=True):
    return AdditiveBlock(
        nn.Sequential(
            nn.Linear(n_c // 2, n_i_c, ),
            nn.ReLU(),
            nn.Linear(n_i_c, n_c // 2, )),

        nn.Sequential(
            nn.Linear(n_c // 2, n_i_c, ),
            nn.ReLU(),
            nn.Linear(n_i_c, n_c // 2, )),
        switched_order=switched_order)


def conv_add_3x3_const(n_c, n_i_c, keep_input=False, keep_output=False):
    return AdditiveBlockConstantMemory(
        nn.Sequential(
            nn.Conv2d(n_c // 2, n_i_c, (3, 1), stride=1, padding=(1, 0),
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(n_i_c, n_c // 2, (3, 1), stride=1, padding=(1, 0),
                      bias=True)),

        nn.Sequential(
            nn.Conv2d(n_c // 2, n_i_c, (3, 1), stride=1, padding=(1, 0),
                      bias=True),
            nn.ReLU(),
            nn.Conv2d(n_i_c, n_c // 2, (3, 1), stride=1, padding=(1, 0),
                      bias=True)),
        keep_input=keep_input, keep_output=keep_output)


def dense_add_const(n_c, n_i_c, keep_input=False, keep_output=False):
    return AdditiveBlockConstantMemory(
        nn.Sequential(
            nn.Linear(n_c // 2, n_i_c, ),
            nn.ReLU(),
            nn.Linear(n_i_c, n_c // 2, )),

        nn.Sequential(
            nn.Linear(n_c // 2, n_i_c, ),
            nn.ReLU(),
            nn.Linear(n_i_c, n_c // 2, )),
    keep_input=keep_input, keep_output=keep_output)
