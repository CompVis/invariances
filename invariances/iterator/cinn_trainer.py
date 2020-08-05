import torch
import numpy as np
import autoencoders
from edflow.util import retrieve
from edflow.util import get_obj_from_str

from invariances.iterator.base_trainer import Trainer

halfscale = lambda x: 0.5 * (x + 1.)


class AutoencoderConcatTrainer(Trainer):
    """
    Concatenates a given neural network (e.g. a classifier) and an autoencoder.
    Example: see configs/alexnet
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        greybox_config = self.config["GreyboxModel"]
        self.init_greybox(greybox_config)
        ae_config = self.config["AutoencoderModel"]
        self.init_ae(ae_config)
        self.log_n_samples = retrieve(self.config, "n_samples_logging",
                                      default=3)  # visualize n samples per representation

    def init_greybox(self, config):
        """Initializes a provided 'Greybox', i.e. a model one wants to interpret/analyze."""
        self.greybox = get_obj_from_str(config["model"])(config)
        self.greybox.to(self.device)
        self.greybox.eval()

    def init_ae(self, config):
        """Initializes autoencoder"""
        if "pretrained_ae_key" in config:
            # load from the 'autoencoders' repo
            ae_key = config["pretrained_ae_key"]
            self.autoencoder = autoencoders.get_model(ae_key)
            self.logger.info("Loaded autoencoder {} from 'autoencoders'".format(ae_key))
        else:
            # in case you want to use a checkpoint different from the one provided.
            subconfig = config["subconfig"]
            self.autoencoder = get_obj_from_str(config["model"])(subconfig)
            if "checkpoint" in config:
                checkpoint = config["checkpoint"]
                state = torch.load(checkpoint)["model"]
                self.autoencoder.load_state_dict(state)
                self.logger.info("Restored autoencoder from {}".format(checkpoint))
        self.autoencoder.to(self.device)
        self.autoencoder.eval()

    def eval_op_samples(self, xin, zae, zrep):
        with torch.no_grad():
            # reconstruction
            zz, _ = self.model(zae, zrep)
            # samples
            log = dict()
            for n in range(self.log_n_samples):
                zz_sample = torch.randn_like(zz)
                zae_sample = self.model.reverse(zz_sample, zrep)
                xae_sample = self.autoencoder.decode(zae_sample)
                log["samples_{:02}".format(n)] = self.tonp(xae_sample)

            # autoencoder reconstruction without inn
            xae_rec = self.autoencoder.decode(zae)
            # samples from the autoencoder
            ae_sample = self.autoencoder.decode(zz_sample)
            log["reconstructions"] = self.tonp(xae_rec)
            log["autoencoder_sample"] = self.tonp(ae_sample)
            log["inputs"] = self.tonp(xin)
        return log

    def step_op(self, *args, **kwargs):
        with torch.no_grad():
            xin = kwargs["image"]
            xin = self.totorch(xin)
            zrep = self.greybox.encode(xin).sample()
            zae = self.autoencoder.encode(xin).sample()

        zz, logdet = self.model(zae, zrep)
        loss, log_dict, _ = self.loss(zz, logdet, self.get_global_step())

        if not "images" in log_dict:
            log_dict["images"] = dict()

        def train_op():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():
            with torch.no_grad():
                # input
                log_dict["images"]["xin"] = xin
                # reconstruction
                zz, _ = self.model(zae, zrep)
                zae_rec = self.model.reverse(zz, zrep)
                xae_rec = self.autoencoder.decode(zae_rec)
                log_dict["images"]["xdec_rec"] = xae_rec

                # samples
                for n in range(self.log_n_samples):
                    zz_sample = torch.randn_like(zz)
                    zae_sample = self.model.reverse(zz_sample, zrep)
                    xae_sample = self.autoencoder.decode(zae_sample)
                    log_dict["images"]["xdec_sample_{:02}".format(n)] = xae_sample
                # swaps
                zz_swapped = torch.flip(zz, [0])
                zz_swap_ae = self.model.reverse(zz_swapped, zrep)
                xae_swap = self.autoencoder.decode(zz_swap_ae)
                log_dict["images"]["xdec_swap_zz"] = xae_swap

            for k in log_dict:
                for kk in log_dict[k]:
                    log_dict[k][kk] = self.tonp(log_dict[k][kk])
            return log_dict

        def eval_op():
            return self.eval_op_samples(xin, zae, zrep)

        return {"train_op": train_op,
                "log_op": log_op,
                "eval_op": eval_op
                }
