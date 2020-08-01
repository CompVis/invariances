import torch

from edflow.util import retrieve
from edflow.util import get_obj_from_str
import autoencoders

from invariances.iterator.base_trainer import Trainer

halfscale = lambda x: 0.5*(x+1.)


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
        if "checkpoint" in config:
            checkpoint = config["checkpoint"]
            state = torch.load(checkpoint)["model"]
            self.greybox.load_state_dict(state)
            self.logger.info("Restored greybox from {}".format(checkpoint))
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
            # todo: really need this?
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
            zae_rec = self.model.reverse(zz, zrep)
            # sanity check:
            if torch.norm(zae - zae_rec) > 1e-2:
                self.logger.info(
                    "Warning: INN might be broken. Reconstruction norm is: {}".format(torch.norm(zae - zae_rec)))
            # samples
            log = dict()
            for n in range(self.log_n_samples):
                zz_sample = torch.randn_like(zz)
                zae_sample = self.model.reverse(zz_sample, zrep)
                xae_sample = self.autoencoder.decode(zae_sample)
                log["samples_{:02}".format(n)] = self.tonp(xae_sample)

            # autoencoder reconstruction without inn
            xae_rec = self.autoencoder.decode(zae)
            # autoencoder reconstruction with inn
            xae_rec_inn = self.autoencoder.decode(zae_rec)
            # samples from the autoencoder
            ae_sample = self.autoencoder.decode(zz_sample)
            log["reconstructions"] = self.tonp(xae_rec)
            log["reconstructions_inn"] = self.tonp(xae_rec_inn)
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

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


class BigGANConcatTrainer(Trainer):
    """
    todo: description
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_config = self.config["GreyboxModel"]
        self.init_model(model_config)
        dec_config = self.config["DecoderModel"]
        self.init_dec(dec_config)
        # todo: load minivae through the 'autoencoders' repo
        helper_vae_config = self.config["EmbedderVAE"]
        self.init_embedder_vae(helper_vae_config)

    def init_model(self, config):
        """Initializes model to be analyzed"""
        subconfig = config["subconfig"]
        self.network = get_obj_from_str(config["model"])(subconfig)
        self.network.to(self.device)
        self.network.eval()

    def init_dec(self, config):
        """Initializes Decoder"""
        subconfig = config["subconfig"]
        self.decoder = get_obj_from_str(config["model"])(subconfig)
        self.decoder.to(self.device)
        self.decoder.eval()

    def init_embedder_vae(self, config):
        # TODO: need to load this one through the autoencoders repo
        subconfig = config["subconfig"]
        self.vae = get_obj_from_str(config["model"])(subconfig)
        vae_checkpoint_path = retrieve(subconfig, "vae_checkpoint", default='none')
        self.fix_vae = retrieve(subconfig, "vae_fixed", default=True)
        if vae_checkpoint_path is not 'none':
            state = torch.load(vae_checkpoint_path)["model"]
            self.vae.load_state_dict(state)
            self.logger.info("Loaded embedder-vae from {}".format(vae_checkpoint_path))
        self.vae.to(self.device)
        self.vae.eval()

    def step_op(self, *args, **kwargs):
        # todo: namespaces
        with torch.no_grad():
            zin = self.totorch(kwargs["z"], guess_image=False)
            cin = self.totorch(kwargs["class"], guess_image=False)
            ein = self.decoder.embed_labels(cin, labels_are_one_hot=False)
            xin = self.decoder.generate_from_embedding(zin, ein)

        eposterior = self.vae.encode(ein)
        einrec = self.vae.decode(eposterior.sample())

        with torch.no_grad():
            xinrec = self.decoder.generate_from_embedding(zin, einrec)

        split_sizes = [zin.shape[1], ein.shape[1]]
        zdec = torch.cat([zin, einrec.detach()], dim=1)[:, :, None, None]   # this will be flowed

        zfeatures = self.network.encode(xin).sample()
        znet = zfeatures[self.layer_idx]
        zz, logdet = self.model(zdec, znet)
        loss, log_dict = self.loss(zz, logdet, self.get_global_step())

        if not "images" in log_dict:
            log_dict["images"] = dict()

        def train_op():
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        def log_op():
            with torch.no_grad():
                log_dict["images"]["xin"] = xin
                log_dict["images"]["xinrec"] = xinrec

                for n in range(3):
                    zz_sample = torch.randn_like(zdec)
                    zdec_sample = self.model.reverse(zz_sample,znet).squeeze(-1).squeeze(-1)
                    xdec_sample = self.decoder.generate_from_embedding(
                        *torch.split(zdec_sample, split_sizes, dim=1))

                    xcls_sample = self.decoder(torch.randn_like(zin), cin)
                    log_dict["images"]["xdec_sample_{:02}".format(n)] = xdec_sample
                    log_dict["images"]["xcls_sample_{:02}".format(n)] = xcls_sample

            for k in log_dict:
                for kk in log_dict[k]:
                    log_dict[k][kk] = self.tonp(log_dict[k][kk], guess_image=k == 'images')
            return log_dict

        return {"train_op": train_op,
                "log_op": log_op}