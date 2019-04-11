import torch as th
from torch import nn

class ProjectionDiscriminator(nn.Module):
    def __init__(self, adv_feature_model, n_features, n_classes):
        super(ProjectionDiscriminator, self).__init__()
        self.adv_feature_model = adv_feature_model
        self.n_features = n_features
        self.n_classes = n_classes
        self.embed = nn.Embedding(n_classes,n_features,)

    def forward(self, x, y):
        features = self.adv_feature_model(x)
        embedding =  self.embed(y)
        out = th.sum(embedding * features, dim=1)
        return out