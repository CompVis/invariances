"""
Model-Classes in this file were used for training
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import models
from edflow import get_logger
from edflow.util import retrieve

from invariances.util.ckpt_util import STYLE_MODEL_URLS, URL_MAP, CONFIG_MAP, get_ckpt_path

rescale = lambda x: 0.5 * (x + 1)


class AlexNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = models.alexnet(pretrained=retrieve(config, "Model/imagenet_pretrained", default=True))
        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(227, 227), mode="bilinear")),
            torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
        ])

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @classmethod
    def from_pretrained(cls, name):
        raise NotImplementedError("Please refer to the torchvision-model pretrained on Imagenet.")

    def forward(self, x):
        x = self.image_transform(x)
        output = self.model(x)
        return output


class ResNet(nn.Module):
    """
    All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of
    shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to
    a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    A classifier pretrained on stylized imagenet is also available (see https://arxiv.org/abs/1811.12231)
    """
    def __init__(self, config):
        super().__init__()
        possible_resnets = {
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'resnet50stylized': models.resnet50,
        }
        from torch.utils import model_zoo
        self.logger = get_logger(self.__class__.__name__)
        self.n_out = retrieve(config, "Model/n_classes")
        self.type = retrieve(config, "Model/type", default='resnet50')
        custom_head = retrieve(config, "Model/custom_head", default=True)
        self.model = possible_resnets[self.type](pretrained=retrieve(config, "Model/imagenet_pretrained", default=True))

        if custom_head:
            self.model.fc = nn.Linear(self.model.fc.in_features, self.n_out)

        if self.type in ["resnet50stylized"]:
            self.logger.info("Loading pretrained Resnet-50 trained on stylized ImageNet")
            which_stylized = retrieve(config, "Model/whichstyle",
                                      default="resnet50_trained_on_SIN")

            self.logger.info("Loading {} from url {}".format(which_stylized, STYLE_MODEL_URLS[which_stylized]))
            assert not custom_head
            url = STYLE_MODEL_URLS[which_stylized]
            state = model_zoo.load_url(url)
            # remove the .module in keys of state dict (from DataParallel)
            state_unboxed = dict()
            for k in tqdm(state["state_dict"].keys(), desc="StateDict"):
                state_unboxed[k[7:]] = state["state_dict"][k]
            self.model.load_state_dict(state_unboxed)
            self.logger.info("Loaded resnet50 trained on stylized ImageNet, version {}".format(which_stylized))

        normalize = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        self.image_transform = torchvision.transforms.Compose([
                torchvision.transforms.Lambda(lambda image: F.interpolate(image, size=(224, 224), mode="bilinear")),
                torchvision.transforms.Lambda(lambda image: torch.stack([normalize(rescale(x)) for x in image]))
            ])

    def forward(self, x):
        x = self._pre_process(x)
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        prediction = self.model.fc(x)
        return prediction

    @classmethod
    def from_pretrained(cls, name, config=None, num_remove=0):
        if name not in URL_MAP:
            raise NotImplementedError(name)
        if config is None:
            config = CONFIG_MAP[name]

        model = cls(config)
        ckpt = get_ckpt_path(name)
        state_dict = torch.load(ckpt, map_location=torch.device("cpu"))
        if num_remove > 0:
            # removes the leading characters in the state-dicts' keys, e.g. to remove the '.module' for
            # data parallel training
            sd_new = dict()
            for key in state_dict.keys():
                sd_new[key[num_remove:]] = state_dict[key]
            state_dict = sd_new
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _pre_process(self, x):
        x = self.image_transform(x)
        return x

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]
