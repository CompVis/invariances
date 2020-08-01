import torch
import numpy as np
from edflow import TemplateIterator, get_obj_from_str
from edflow.util import retrieve


def totorch(x, guess_image=True, device=None, dontmap=False):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    x = torch.tensor(x)
    if guess_image and len(x.size()) == 4:
        x = x.transpose(2, 3).transpose(1, 2)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    if not dontmap:
        x = x.to(device)
    else:
        pass
    return x


def tonp(x, guess_image=True):
    try:
        if guess_image and len(x.shape) == 4:
            x = x.transpose(1, 2).transpose(2, 3)
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def repeat_channels(x, num=3):
    if x.size(1) == 1:
        x = x.repeat(1, num, 1, 1)
    return x


def get_learning_rate(config):
    if "learning_rate" in config:
        learning_rate = config["learning_rate"]
    elif "base_learning_rate" in config:
        learning_rate = config["base_learning_rate"]*config["batch_size"]
    else:
        learning_rate = 0.001
    return learning_rate


def interpolate_corners(z, side=5, permute=False):
    """
        Interpolate the first four encodings obtained from z.
        :param z: tensor of shape bs x nc x iw x ih
    """
    device = z.get_device()
    if permute:
        ridx = torch.randperm(z.size(0))
        z = z[ridx]
    n = side * side
    xv, yv = np.meshgrid(np.linspace(0, 1, side),
                         np.linspace(0, 1, side))
    xv = xv.reshape(n, 1, 1, 1)
    yv = yv.reshape(n, 1, 1, 1)

    xv, yv = torch.tensor(xv).to(device).float(), torch.tensor(yv).to(device).float()

    z_interp = \
        z[0] * (1 - xv) * (1 - yv) + \
        z[1] * xv * (1 - yv) + \
        z[2] * (1 - xv) * yv + \
        z[3] * xv * yv

    if permute:
        return z_interp, ridx
    return z_interp, None


class Iterator(TemplateIterator):
    """
    Base class to handle device and state.

    Config parameters:
        - test_mode : boolean : Put model into .eval() mode.
        - no_restore_keys : string1,string2 : Submodels which should not be
                                              restored from checkpoint.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.test_mode = self.config.get("test_mode", False)
        self.submodules = ["model"]

        if "pretrained_key" in self.config:
            self.model = get_obj_from_str(self.config["pretrained_model"]).from_pretrained(
                self.config["pretrained_key"])
            self.logger.info("Loaded pretrained model from key {}".format(self.config["pretrained_key"]))
            self.logger.info("Warning: This overrides any model specified as 'model' in the edflow config.")
            self.model.eval()

        if self.config.get("test_mode", False):
            # in eval mode
            self.model.eval()
        self.model.to(self.device)
        self.do_not_restore_keys = retrieve(self.config, 'no_restore_keys', default='').split(',')

    def get_state(self):
        state = dict()
        for k in self.submodules:
            state[k] = getattr(self, k).state_dict()
        return state

    def save(self, checkpoint_path):
        torch.save(self.get_state(), checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        keys = list(state.keys())
        for k in keys:
            if hasattr(self, k):
                if k not in self.do_not_restore_keys:
                    try:
                        missing, unexpected = getattr(self, k).load_state_dict(state[k], strict=False)
                        if missing:
                            self.logger.info("Missing keys for {}: {}".format(k, missing))
                        if unexpected:
                            self.logger.info("Unexpected keys for {}: {}".format(k, unexpected))
                    except TypeError:
                        self.logger.info(k)
                        try:
                            getattr(self, k).load_state_dict(state[k])
                        except ValueError:
                            self.logger.info("Could not load state dict for key {}".format(k))
                    else:
                        self.logger.info('Restored key `{}`'.format(k))
                else:
                    self.logger.info('Not restoring key `{}` (as specified)'.format(k))
            # avoid running out of memory when restoring large models
            # https://discuss.pytorch.org/t/load-state-dict-causes-memory-leak/36189/10
            del state[k]


class Trainer(Iterator):
    """
    Base Trainer. All other iterators should be build on top of this class.
    Adds optimizer and loss.

    Config parameters:
        - learning_rate : float : Learning rate of Adam
        - base_learning_rate : float : Learning_rate per example to adjust for
                                       batch size (ignored if learning_rate is present)
        - loss : string : Import path of loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.learning_rate = get_learning_rate(self.config)
        self.loss_lr_factor = retrieve(self.config, "loss_lr_factor", default=1.0)
        self.loss_lr = self.loss_lr_factor*self.learning_rate
        self.logger.info("learning_rate: {}".format(self.learning_rate))
        self.logger.info("loss learning_rate: {}".format(self.loss_lr))
        if "loss" in self.config:
            self.loss = get_obj_from_str(self.config["loss"])(self.config)
            self.loss.to(self.device)
            self.submodules.append("loss")
            self.optimizer = torch.optim.Adam([
                {"params": self.model.parameters()},
                {"params": self.loss.parameters(),
                 "lr": self.loss_lr}], lr=self.learning_rate, betas=(0.5, 0.9))
            self.submodules.append("optimizer")

    def totorch(self, x, guess_image=True, dontmap=False):
        return totorch(x, guess_image=guess_image, device=self.device, dontmap=dontmap)

    def tonp(self, x, guess_image=True):
        return tonp(x, guess_image=guess_image)

    def interpolate_corners(self, x, num_side, permute=False):
        return interpolate_corners(x, side=num_side, permute=permute)

    def step_op(self, model, **kwargs):
        return model.step_op(iterator=self, **kwargs)
