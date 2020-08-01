import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
from edflow.util import retrieve
from edflow import get_logger

from autoencoders.distributions import DiracDistribution

from invariances.model.blocks import ActNorm
from invariances.model.cinn import DenseEmbedder, Embedder
from invariances.util.model_util import Flatten

rescale = lambda x: 0.5*(x+1)


class AbstractGreybox(nn.Module):
    """
    Subclassed by all greybox-models (i.e. models which are to be interpreted).
    The .forward() pass should produce the standard model operation, i.e. give logits if the
    greybox is a classifier. Thus, each greybox needs and encode() and decode() method, which
    splits the model into 'phi' and 'psi', such that the concatenation decode(encode(x)) mimics the
    complete forward pass.

    """
    def __init__(self, config):
        super().__init__()
        self.config = config["subconfig"]
        # per default, we "split" at the input, i.e. no splitting is happening
        self.split_at = retrieve(config, "split_idx", default=0)

    def forward(self, x):
        z = self.encode(x)
        try:
            z = z.mode()
        except Exception as e:
            print("Did you wrap the output of '.encode()' in a 'Distribution' object?")
            raise e
        output = self.decode(z)
        return output

    def encode(self, x):
        """The phi-part from the paper. Needs to wrap its output into a 'Distribution' Object."""
        raise NotImplementedError

    def decode(self, x):
        """The psi-part from the paper"""
        raise NotImplementedError


class AlexNetClassifier(AbstractGreybox):
    """split model at specified index, e.g. to extract layer 'conv5'"""
    def __init__(self, config):
        super().__init__(config=config)
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        self.logger = get_logger(self.__class__.__name__)
        pretrained = retrieve(self.config, "Model/pretrained", default=True)
        model = torchvision.models.alexnet(pretrained=pretrained)
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(227, 227), mode="bilinear")),
            torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        ])

        # prepare the model for analysis with variable layer indices. This needs to be handcrafted if for any
        # other model you may want to choose -- see the ResNet and SqueezeNet models for examples
        self.layers = nn.ModuleList()
        for layer in model.features:
            self.layers.append(layer)
        self.layers.append(model.avgpool)
        self.layers.append(Flatten(1))
        for layer in model.classifier:
            self.layers.append(layer)
        self.layers.append(nn.Softmax())
        assert len(self.layers) == 23
        self.logger.info("Layer Information: \n {}".format(self.layers))
        del model   # don't need this hanging around

    def encode(self, x):
        """
        return before x is passed through the layer indexed by 'self.split_at'
        """
        x = self._pre_process(x)
        for i in range(len(self.layers)+1):
            if i != self.split_at:
                x = self.layers[i](x)
            else:
                if len(x.shape) == 2:
                    x = x[:,:,None,None]
                return DiracDistribution(x)

    def decode(self, x):
        for i in range(self.split_at, len(self.layers)):
            x = self.layers[i](x)
        return x

    def dense_predict(self, x):
        x = self.model.fc(x)
        return x

    def _pre_process(self, x):
        x = self.image_transform(x)
        return x

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]


class ResnetClassifier(AbstractGreybox):
    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
    shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to
    a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    A classifier pretrained on stylized imagenet is also available (see https://arxiv.org/abs/1811.12231)
    """
    def __init__(self, config):
        super().__init__(config=config)
        __possible_resnets = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet50stylized': models.resnet50
        }

        style_model_urls = {
            'resnet50_trained_on_SIN':
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN':
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN':
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'
        }

        __classification_heads = {"linear": nn.Linear,
                                  "nonlinear": DenseEmbedder}
        from torch.utils import model_zoo
        self.logger = get_logger(self.__class__.__name__)
        self.n_out = retrieve(self.config, "Model/n_classes")
        self.type = retrieve(self.config, "Model/type", default='resnet50')
        finetune = retrieve(self.config, "Model/finetune", default=True)
        clf_head_type = retrieve(self.config, "Model/clf_head", default="linear")

        model = __possible_resnets[type](pretrained=True)
        if finetune:
            self.logger.info("Warning: Using a new classification head of type {}".format(clf_head_type))
            self.model.fc = __classification_heads[clf_head_type](self.model.fc.in_features, self.n_out)
        if type in ["resnet50stylized"]:
            self.logger.info("Loading pretrained Resnet-50 trained on stylized ImageNet")
            which_stylized = retrieve(self.config, "Model/whichstyle",
                                      default="resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN")
            self.logger.info("Loading {} from url {}".format(which_stylized, style_model_urls[which_stylized]))
            assert not finetune, 'Not possible for now (eccv2020)'
            #state = torch.load('/export/home/rrombach/.cache/torch/checkpoints/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar')
            url = style_model_urls[which_stylized]
            state = model_zoo.load_url(url)
            # remove the .module in keys of state dict (from DataParallel)
            # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686
            state_unboxed = dict()
            for k in tqdm(state["state_dict"].keys(), desc="StateDict"):
                state_unboxed[k[7:]] = state["state_dict"][k]
            model.load_state_dict(state_unboxed)
            self.logger.info("Loaded resnet50 trained on stylized ImageNet, version {}".format(which_stylized))

        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
            torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        ])
        self.layers = nn.ModuleList()
        self.layers.append(model.conv1)
        self.layers.append(model.bn1)
        self.layers.append(model.relu)
        self.layers.append(model.maxpool)
        self.layers.append(model.layer1)
        self.layers.append(model.layer2)
        self.layers.append(model.layer3)
        self.layers.append(model.layer4)
        self.layers.append(model.avgpool)
        self.layers.append(model.fc)

    def forward(self, x):
        x = self._pre_process(x)
        for layer in self.layers[:-1]:
            x = layer(x)
        x = torch.flatten(x, 1)
        prediction = self.layers[-1](x)
        return prediction

    def encode(self, x):
        x = self._pre_process(x)
        for i in range(len(self.layers)):
            if i != self.split_at:
                x = self.layers[i](x)
            else:
                return DeltaDistribution(x)

    def decode(self, x):
        for i in range(self.split_at, len(self.layers)):
            x = self.layers[i](x)
        return x

    def return_features(self, x):
        """returns intermediate features and logits. Could also add softmaxed class decisions.

        For Resnet-101, the following sizes are returned (11 is batch-size and can be ignored,
        the ones marked with an 'x' were used in the paper.):

            torch.Size([11, 3, 224, 224])   ---  150528                                     x
            torch.Size([11, 64, 112, 112])  ---  802816
            torch.Size([11, 64, 112, 112])  ---  802816
            torch.Size([11, 64, 112, 112])  ---  802816
            torch.Size([11, 64, 56, 56])    ---  200704                                     x
            torch.Size([11, 256, 56, 56])   ---  802816
            torch.Size([11, 512, 28, 28])   ---  401408
            torch.Size([11, 1024, 14, 14])  ---  200704                                     x
            torch.Size([11, 2048, 7, 7])    ---  100352
            torch.Size([11, 2048, 1, 1])    ---  2048                                       x
            torch.Size([11, n_classes, 1, 1])     ---  n_classes  +++  logits
            torch.Size([11, n_classes, 1, 1])     ---  n_classes  +++  softmaxed logits     x

        """
        features = list()
        x = self._pre_process(x)
        features.append(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        features.append(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        features.append(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        features.append(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        features.append(F.softmax(x, dim=1)[:,:,None,None])   # class probs
        return features

    def dense_predict(self, x):
        x = self.model.fc(x)
        return x

    def test(self, x):
        with torch.no_grad():
            if x.shape[1] == 1:
                x = x[:, [0, 0, 0], :, :]
            h = self.return_features(x)
        return h[-2].shape

    def _pre_process(self, x):
        x = self.image_transform(x)
        return x

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @property
    def input_size(self):
        return [3, 224, 224]


class WideResnetClassifier(ResnetClassifier):
    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
     shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to
     a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
    """
    def __init__(self, config):
        super().__init__(config)
        __possible_resnets = {
            'resnet50': models.wide_resnet50_2,
            'resnet101': models.wide_resnet101_2
        }

        __available_norms = {'bn': nn.BatchNorm2d,
                             'an': ActNorm}

        __classification_heads = {"linear": nn.Linear,
                                  "nonlinear": DenseEmbedder}

        self.config = config
        n_out = retrieve(config, "Model/n_classes")
        type = retrieve(config, "Model/type", default='resnet50')
        norm_layer = __available_norms[retrieve(config, "Model/norm")]
        finetune = retrieve(config, "Model/finetune", default=True)
        clf_head_type = retrieve(config, "Model/clf_head", default="linear")
        self.type = type
        self.n_out = n_out
        self.model = __possible_resnets[type](pretrained=retrieve(config, "Model/norm")=='bn', norm_layer=norm_layer)
        if finetune:
            self.model.fc = __classification_heads[clf_head_type](self.model.fc.in_features, n_out)

        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
            torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        ])
