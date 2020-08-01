"""Flow consisting of ActNorm, DoubleVectorCouplingBlock and Shuffle. Additionally, powerful conditioning encodings are
learned."""
import torch
import torch.nn as nn
import numpy as np
from edflow.util import retrieve

from invariances.model.blocks import ActNorm, ConditionalFlow, FeatureLayer, DenseEncoderLayer
from invariances.util.ckpt_util import get_ckpt_path, URL_MAP, CONFIG_MAP


class DenseEmbedder(nn.Module):
    """Basically an MLP. Maps vector-like features to some other vector of given dimenionality"""
    def __init__(self, in_dim, up_dim, depth=4, given_dims=None):
        super().__init__()
        self.net = nn.ModuleList()
        if given_dims is not None:
            assert given_dims[0] == in_dim
            assert given_dims[-1] == up_dim
            dims = given_dims
        else:
            dims = np.linspace(in_dim, up_dim, depth).astype(int)
        for l in range(len(dims)-2):
            self.net.append(nn.Conv2d(dims[l], dims[l + 1], 1))
            self.net.append(ActNorm(dims[l + 1]))
            self.net.append(nn.LeakyReLU(0.2))

        self.net.append(nn.Conv2d(dims[-2], dims[-1], 1))

    def forward(self, x):
        for layer in self.net:
            x = layer(x)
        return x.squeeze(-1).squeeze(-1)


class Embedder(nn.Module):
    """Embeds a 4-dim tensor onto dense latent code."""
    def __init__(self, in_spatial_size, in_channels, emb_dim, n_down=4):
        super().__init__()
        self.feature_layers = nn.ModuleList()
        norm = 'an'  # hard coded
        bottleneck_size = in_spatial_size // 2**n_down
        self.feature_layers.append(FeatureLayer(0, in_channels=in_channels, norm=norm))
        for scale in range(1, n_down):
            self.feature_layers.append(FeatureLayer(scale, norm=norm))
        self.dense_encode = DenseEncoderLayer(n_down, bottleneck_size, emb_dim)
        if n_down == 1:
            # add some extra parameters to make model a little more powerful ? # TODO
            print(" Warning: Embedder for ConditionalTransformer has only one down-sampling step. You might want to "
                  "increase its capacity.")

    def forward(self, input):
        h = input
        for layer in self.feature_layers:
            h = layer(h)
        h = self.dense_encode(h)
        return h.squeeze(-1).squeeze(-1)


class ConditionalTransformer(nn.Module):
    """Conditional Transformer"""
    def __init__(self, config):
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        super().__init__()
        self.config = config
        # get all the hyperparameters
        in_channels = retrieve(config, "Transformer/in_channels")
        mid_channels = retrieve(config, "Transformer/mid_channels")
        hidden_depth = retrieve(config, "Transformer/hidden_depth")
        n_flows = retrieve(config, "Transformer/n_flows")
        conditioning_option = retrieve(config, "Transformer/conditioning_option")
        flowactivation = retrieve(config, "Transformer/activation", default="lrelu")
        embedding_channels = retrieve(config, "Transformer/embedding_channels", default=in_channels)
        n_down = retrieve(config, "Transformer/embedder_down", default=4)

        self.emb_channels = embedding_channels
        self.in_channels = in_channels

        self.flow = ConditionalFlow(in_channels=in_channels, embedding_dim=self.emb_channels, hidden_dim=mid_channels,
                                    hidden_depth=hidden_depth, n_flows=n_flows, conditioning_option=conditioning_option,
                                    activation=flowactivation)
        conditioning_spatial_size = retrieve(config, "Transformer/conditioning_spatial_size")
        conditioning_in_channels = retrieve(config, "Transformer/conditioning_in_channels")
        if conditioning_spatial_size == 1:
            depth = retrieve(config, "Transformer/conditioning_depth",
                             default=4)
            dims = retrieve(config, "Transformer/conditioning_dims",
                            default="none")
            dims = None if dims == "none" else dims
            self.embedder = DenseEmbedder(conditioning_in_channels,
                                          in_channels,
                                          depth=depth,
                                          given_dims=dims)
        else:
            self.embedder = Embedder(conditioning_spatial_size, conditioning_in_channels, in_channels, n_down=n_down)

    def embed(self, conditioning):
        # embed it via embedding layer
        embedding = self.embedder(conditioning)
        return embedding

    def sample(self, shape, conditioning):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, self.embed(conditioning))
        return sample

    def forward(self, input, conditioning, train=False):
        embedding = self.embed(conditioning)
        out, logdet = self.flow(input, embedding)
        if train:
            return self.flow.last_outs, self.flow.last_logdets
        return out, logdet

    def reverse(self, out, conditioning):
        embedding = self.embed(conditioning)
        return self.flow(out, embedding, reverse=True)

    def get_last_layer(self):
        return getattr(self.flow.sub_layers[-1].coupling.t[-1].main[-1], 'weight')

    @classmethod
    def from_pretrained(cls, name, config=None):
        if name not in URL_MAP:
            raise NotImplementedError(name)
        if config is None:
            config = CONFIG_MAP[name]

        model = cls(config)
        ckpt = get_ckpt_path(name)
        model.load_state_dict(torch.load(ckpt, map_location=torch.device("cpu")))
        model.eval()
        return model


class PretrainedModel(ConditionalTransformer):
    # TODO: does this need to be a nn.Module?
    def __init__(self, config):
        pretrained_key = retrieve(config, "pretrained_key")
        assert pretrained_key in CONFIG_MAP, 'The model under the provided key {} is not (yet) available.'.format(
            pretrained_key)
        self.model = ConditionalTransformer.from_pretrained(pretrained_key)

    def __getattr__(self, name):
        try:
            return getattr(self.model, name)
        except AttributeError:
            return getattr(self, name)

class Dummy:
    def __init__(self, dummy):
        pass

if __name__ == "__main__":
    # TODO: kick this whole part
    from invariances.util.ckpt_util import URL_MAP
    for key in URL_MAP:
        print("loading key {}...".format(key))
        model = ConditionalTransformer.from_pretrained(key)
        print("model sucessfully loaded.")
        z = torch.randn(11, 128, 1, 1)
        #zz, _ = model(z, cond)
        #zrec  = model.reverse(zz, cond)
        #print("norm:", torch.norm(z-zz))
    print("loaded all")
    print("done.")
