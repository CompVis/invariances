import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from edflow.util import retrieve, get_obj_from_str
from edflow import get_logger

from autoencoders.distributions import DiracDistribution
from invariances.util.model_util import Flatten

rescale = lambda x: 0.5 * (x + 1)


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
        self.model_config = config["subconfig"]
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

    def prepare(self, model_as_str, checkpoint="none", pretrained_key="none"):
        """Prepare the model , e.g. instantiate a pre-trained classifier from torchvision"""
        # e.g. model=config["model"]
        if pretrained_key is not "none":
            model = get_obj_from_str(model_as_str).from_pretrained(pretrained_key)
        else:
            model = get_obj_from_str(model_as_str)(self.model_config)
            if checkpoint is not "none":
                assert type(checkpoint) == str, 'please provide a path to a checkpoint'
                state = torch.load(checkpoint)["model"]
                model.load_state_dict(state)
                self.logger.info("Restored model from {}".format(checkpoint))
        return model


class AlexNetClassifier(AbstractGreybox):
    """split model at specified index, e.g. to extract layer 'conv5'"""
    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    def __init__(self, config):
        super().__init__(config=config)
        self.logger = get_logger(self.__class__.__name__)
        model = self.prepare(model_as_str=self.model_config["model"],
                             checkpoint=retrieve(self.model_config, "checkpoint", default="none"),
                             )
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(227, 227), mode="bilinear")),
            torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        ])

        # prepare the model for analysis with variable layer indices. This needs to be handcrafted for any
        # other model you may want to choose -- see the ResNet and SqueezeNet models for more examples
        self.layers = nn.ModuleList()
        for layer in model.features:
            self.layers.append(layer)
        self.layers.append(model.avgpool)
        self.layers.append(Flatten(1))
        for layer in model.classifier:
            self.layers.append(layer)
        if retrieve(config, "append_softmax", default=True):
            self.layers.append(nn.Softmax())
            assert len(self.layers) == 23
        self.logger.info("Layer Information: \n {}".format(self.layers))
        del model  # don't need this hanging around

    def encode(self, x):
        """
        return before x is passed through the layer indexed by 'self.split_at'
        """
        x = self._pre_process(x)
        for i in range(len(self.layers) + 1):
            if i != self.split_at:
                x = self.layers[i](x)
            else:
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                return DiracDistribution(x)

    def decode(self, x):
        for i in range(self.split_at, len(self.layers)):
            x = self.layers[i](x)
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
        self.logger = get_logger(self.__class__.__name__)
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
            torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        ])
        model = self.prepare(model_as_str=self.model_config["model"],
                             checkpoint=retrieve(self.model_config, "checkpoint", default="none"),
                             pretrained_key=retrieve(self.model_config, "pretrained_key", default="none")
                             ).model

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
        if retrieve(config, "append_softmax", default=True):
            self.logger.info("Note: Appending Softmax as last layer in classifier.")
            self.layers.append(nn.Softmax())
        self.logger.info("Layer Information: \n {}".format(self.layers))

    def encode(self, x):
        x = self._pre_process(x)
        for i in range(len(self.layers) + 1):
            if i != self.split_at:
                x = self.layers[i](x)
            else:
                return DiracDistribution(x)

    def decode(self, x):
        for i in range(self.split_at, len(self.layers)):
            x = self.layers[i](x)
        return x

    def return_features(self, x):
        """
        TODO: not really necessary, remove?
        returns intermediate features and logits. Could also add softmaxed class decisions.

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
        raise NotImplementedError

    def _pre_process(self, x):
        x = self.image_transform(x)
        return x

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

