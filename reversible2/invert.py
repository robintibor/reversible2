import torch as th


def invert(feature_model, features, return_all=False, residual_iterations=7):
    if (feature_model.__class__.__name__ in [
        'ReversibleBlock','SubsampleSplitter', 'ReversibleBlockOld',
        'ResidualBlock', 'AmplitudePhase', 'ViewAs', 'ConstantPad2d',
        'ReversibleBlockOld'] or
            hasattr(feature_model, 'invert')
    ):
        feature_model = th.nn.Sequential(feature_model, )
    all_features = []
    all_features.append(features)
    switched_dims = False
    for module in reversed(list(feature_model.children())):
        if module.__class__.__name__ in ['AffineBlock', 'AdditiveBlock',
            'MultiplicativeBlock']:
            n_chans = features.size()[1]
            y1 = features[:, :n_chans // 2]
            y2 = features[:, n_chans // 2:]
            if module.switched_order:
                # y1 = self.FA(x1) + x2 * th.exp(self.FM(x1))
                # y2 = self.GA(y1) + x1 * th.exp(self.GM(y1))
                switched_dims = not switched_dims
                x1 = y2
                if module.GA is not None:
                    x1 = x1 - module.GA(y1)
                if module.GM is not None:
                    x1 = x1 / th.exp(module.GM(y1) + module.eps)

                if switched_dims:
                    all_features.append(th.cat((y1, x1), dim=1))
                else:
                    all_features.append(th.cat((x1, y1), dim=1))
                x2 = y1
                if module.FA is not None:
                    x2 = x2 - module.FA(x1)
                if module.FM is not None:
                    x2 = x2 / th.exp(module.FM(x1) + module.eps)
                if switched_dims:
                    all_features.append(th.cat((x2, x1), dim=1))
                else:
                    all_features.append(th.cat((x1, x2), dim=1))
            else:
                x2 = y2
                if module.GA is not None:
                    x2 = x2 - module.GA(y1)
                if module.GM is not None:
                    x2 = x2 / th.exp(module.GM(y1) + module.eps)
                all_features.append(th.cat((y1, x2), dim=1))
                x1 = y1
                if module.FA is not None:
                    x1 = x1 - module.FA(x2)
                if module.FM is not None:
                    x1 = x1 / th.exp(module.FM(x2) + module.eps)
                all_features.append(th.cat((x1, x2), dim=1))


            features = th.cat((x1, x2), dim=1)
        elif module.__class__.__name__ == 'ResidualBlock':
            x = features.detach()  # initial guess
            with th.no_grad():
                for _ in range(residual_iterations):
                    x = features - module.F(x)
            # do a final step to get gradients
            features = features - module.F(x)
        elif module.__class__.__name__ == 'ConstantPad2d':
            assert len(module.padding) == 4
            left, right, top, bottom = module.padding  # see pytorch docs
            features = features[:, :, top:-bottom, left:-right]  # see pytorch docs
        elif hasattr(module, 'invert'):
            features = module.invert(features)
        else:
            raise ValueError("Cannot invert {:s}".format(str(module)))
        if return_all:
            if not module.__class__.__name__ in [
                'AffineBlock', 'AdditiveBlock', 'MultiplicativeBlock']:
                all_features.append(features)
                assert not switched_dims
    if return_all:
        return all_features
    else:
        return features
