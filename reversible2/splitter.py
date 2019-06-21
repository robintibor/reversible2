import torch as th

class SubsampleSplitter(th.nn.Module):
    def __init__(self, stride, chunk_chans_first=True, checkerboard=False):
        super(SubsampleSplitter, self).__init__()
        if not hasattr(stride, '__len__'):
            stride = (stride, stride)
        self.stride = stride
        self.chunk_chans_first = chunk_chans_first
        self.checkerboard = checkerboard
        if checkerboard:
            assert stride[0] == 2
            assert stride[1] == 2

    def forward(self, x):
        # Chunk chans first to ensure that each of the two streams in the
        # reversible network will see a subsampled version of the whole input
        # (in case the preceding blocks would not alter the input)
        # and not one half of the input
        new_x = []
        if self.chunk_chans_first:
            xs = th.chunk(x, 2, dim=1)
        else:
            xs = [x]
        for one_x in xs:
            if not self.checkerboard:
                for i_stride in range(self.stride[0]):
                    for j_stride in range(self.stride[1]):
                        new_x.append(
                            one_x[:, :, i_stride::self.stride[0],
                            j_stride::self.stride[1]])
            else:
                new_x.append(one_x[:,:,0::2,0::2])
                new_x.append(one_x[:,:,1::2,1::2])
                new_x.append(one_x[:,:,0::2,1::2])
                new_x.append(one_x[:,:,1::2,0::2])

        new_x = th.cat(new_x, dim=1)
        return new_x


    def invert(self, features):
        # after splitting the input into two along channel dimension if possible
        # for i_stride in range(self.stride):
        #    for j_stride in range(self.stride):
        #        new_x.append(one_x[:,:,i_stride::self.stride, j_stride::self.stride])
        n_all_chans_before = features.size()[1] // (
                self.stride[0] * self.stride[1])
        # if ther was only one chan before, chunk had no effect
        if self.chunk_chans_first and (n_all_chans_before > 1):
            chan_features = th.chunk(features, 2, dim=1)
        else:
            chan_features = [features]
        all_previous_features = []
        for one_chan_features in chan_features:
            previous_features = th.zeros(
                one_chan_features.size()[0],
                one_chan_features.size()[1] // (
                        self.stride[0] * self.stride[1]),
                one_chan_features.size()[2] * self.stride[0],
                one_chan_features.size()[3] * self.stride[1],
            device=features.device)

            n_chans_before = previous_features.size()[1]
            cur_chan = 0
            if not self.checkerboard:
                for i_stride in range(self.stride[0]):
                    for j_stride in range(self.stride[1]):
                        previous_features[:, :, i_stride::self.stride[0],
                        j_stride::self.stride[1]] = (
                            one_chan_features[:,
                            cur_chan * n_chans_before:
                            cur_chan * n_chans_before + n_chans_before])
                        cur_chan += 1
            else:
                # Manually go through 4 checkerboard positions
                assert self.stride[0] == 2
                assert self.stride[1] == 2
                previous_features[:, :, 0::2, 0::2] = (
                    one_chan_features[:,
                    0 * n_chans_before:0 * n_chans_before + n_chans_before])
                previous_features[:, :, 1::2, 1::2] = (
                    one_chan_features[:,
                    1 * n_chans_before:1 * n_chans_before + n_chans_before])
                previous_features[:, :, 0::2, 1::2] = (
                    one_chan_features[:,
                    2 * n_chans_before:2 * n_chans_before + n_chans_before])
                previous_features[:, :, 1::2, 0::2] = (
                    one_chan_features[:,
                    3 * n_chans_before:3 * n_chans_before + n_chans_before])
            all_previous_features.append(previous_features)
        features = th.cat(all_previous_features, dim=1)
        return features

    def __repr__(self):
        return ("SubsampleSplitter(stride={:s}, chunk_chans_first={:s}, "
               "checkerboard={:s})").format(str(self.stride),
                                           str(self.chunk_chans_first),
                                           str(self.checkerboard))

