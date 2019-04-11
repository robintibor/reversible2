from torch import nn
from reversible2.affine import AdditiveBlock


def conv_add_block_3x3(n_c, n_i_c):
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
                      bias=True)))


def dense_add_block(n_c, n_i_c):
    return AdditiveBlock(
        nn.Sequential(
            nn.Linear(n_c // 2, n_i_c, ),
            nn.ReLU(),
            nn.Linear(n_i_c, n_c // 2, )),

        nn.Sequential(
            nn.Linear(n_c // 2, n_i_c, ),
            nn.ReLU(),
            nn.Linear(n_i_c, n_c // 2, )))