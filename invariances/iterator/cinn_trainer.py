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

        return {"train_op": train_op, "log_op": log_op, "eval_op": eval_op}


class VisualizeAttacks(AutoencoderConcatTrainer):
    # TODO: the "logits" part might be tricky -- check with the softmax
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = retrieve(self.config, "Attack/epsilon",
                                default=1.0)
        self.loss = torch.nn.CrossEntropyLoss()
        self.logger.info("Attack with epsilon = {}".format(self.epsilon))
        self.disable_callback = retrieve(self.config, "disable_callback",
                                         default=False)
        self.n_samples = retrieve(self.config, "Attack/n_samples",
                                  default=100)
        self.logger.info("Attack: Drawing {} samples per example.".format(self.n_samples))

    @property
    def callbacks(self):
        if self.disable_callback:
            return {"eval_op": dict()}
        return {"eval_op": {"visualization": visualization}}

    def fgsm_attack(self, image, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [-1,1] range
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad

    def no_attack(self, image, epsilon, data_grad):
        # a random pertubation
        # Collect the element-wise sign of the data gradient
        fake_grad = torch.tensor(np.random.RandomState(1).randn(*data_grad.shape)).to(data_grad)
        sign_data_grad = fake_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [-1,1] range
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad

    def step_op(self, *args, **kwargs):
        log = dict()

        # original image and representation z
        original_image = self.totorch(kwargs["image"])
        original_image.requires_grad = True
        original_z = self.greybox.encode(original_image).sample()
        original_logits = self.greybox.decode(original_z)

        # attack
        cls = self.totorch(kwargs["class"])
        loss = self.loss(original_logits.squeeze(-1).squeeze(-1),
                         cls.type(torch.long))
        # self.greybox.zero_grad()  todo: wat ???
        loss.backward()
        grad = original_image.grad.data
        attacked_image, attacked_noise = self.fgsm_attack(original_image, self.epsilon, grad)
        noisy_image, noisy_noise = self.no_attack(original_image, self.epsilon, grad)

        # encode attacked image to z
        attacked_z = self.greybox.encode(attacked_image).sample()
        attacked_logits = self.greybox.decode(attacked_z)

        # encode noisy image to z
        noisy_z = self.greybox.encode(noisy_image).sample()
        noisy_logits = self.greybox.decode(noisy_z)

        # encode original image with autoencoder
        original_z_ae = self.autoencoder.encode(original_image).sample()
        # flow it into second stage z ss_z conditioned on original
        # representation
        original_ss_z, original_ss_z_logdet = self.model(original_z_ae,
                                                         original_z)
        # invert (for debugging, should be == original_z_ae)
        recovered_z_ae = self.model.reverse(original_ss_z, original_z)
        # swap conditioning to attacked one
        attacked_z_ae = self.model.reverse(original_ss_z, attacked_z)
        # swap conditioning to noisy one
        noisy_z_ae = self.model.reverse(original_ss_z, noisy_z)

        with torch.no_grad():
            # sample
            samples_attacked = list()
            samples_original = list()
            samples_noisy = list()

            for cond_orig, cond_atta, cond_noisy in zip(original_z, attacked_z,
                                                        noisy_z):
                sss = torch.randn(self.n_samples, original_ss_z.shape[1], 1, 1).to(original_z)

                cond_orig = cond_orig.expand(
                    (self.n_samples, cond_orig.shape[0], cond_orig.shape[1], cond_orig.shape[2])).clone()
                cond_atta = cond_atta.expand(
                    (self.n_samples, cond_atta.shape[0], cond_atta.shape[1], cond_atta.shape[2])).clone()
                cond_noisy = cond_noisy.expand(
                    (self.n_samples, cond_noisy.shape[0], cond_noisy.shape[1], cond_noisy.shape[2])).clone()

                first_stage_sample_original = self.model.reverse(sss, cond_orig)
                first_stage_sample_attacked = self.model.reverse(sss, cond_atta)
                first_stage_sample_noisy = self.model.reverse(sss, cond_noisy)
                samples_attacked.append(first_stage_sample_attacked)
                samples_original.append(first_stage_sample_original)
                samples_noisy.append(first_stage_sample_noisy)

            samples_attacked = torch.cat(samples_attacked)  # shape b*n_samples x 128 x 1 x 1
            samples_original = torch.cat(samples_original)
            samples_noisy = torch.cat(samples_noisy)
            # decode to images
            original_reconstruction = self.autoencoder.decode(original_z_ae)
            recovered_reconstruction = self.autoencoder.decode(recovered_z_ae)
            attacked_reconstruction = self.autoencoder.decode(attacked_z_ae)
            noisy_reconstruction = self.autoencoder.decode(noisy_z_ae)

            decoded_samples_attacked = self.autoencoder.decode(samples_attacked)
            decoded_samples_original = self.autoencoder.decode(samples_original)
            decoded_samples_noisy = self.autoencoder.decode(samples_noisy)

            assert decoded_samples_original.shape == decoded_samples_attacked.shape
            assert decoded_samples_original.shape == decoded_samples_noisy.shape
            shape = decoded_samples_original.shape

        log["original_image"] = self.tonp(original_image)
        log["attacked_image"] = self.tonp(attacked_image)
        log["noisy_image"] = self.tonp(noisy_image)
        log["attacked_noise"] = self.tonp(attacked_noise)
        log["noisy_noise"] = self.tonp(noisy_noise)
        log["original_noise"] = np.zeros_like(log["original_image"])
        log["original_noise"][:, 0, 0, 0] = -0.001
        log["original_reconstruction"] = self.tonp(original_reconstruction)
        log["recovered_reconstruction"] = self.tonp(recovered_reconstruction)
        log["attacked_reconstruction"] = self.tonp(attacked_reconstruction)
        log["noisy_reconstruction"] = self.tonp(noisy_reconstruction)
        log["labels"] = dict()
        log["labels"]["original_logits"] = self.tonp(original_logits)
        log["labels"]["attacked_logits"] = self.tonp(attacked_logits)
        log["labels"]["noisy_logits"] = self.tonp(noisy_logits)

        log["labels"]["samples_attacked"] = self.tonp(decoded_samples_attacked).reshape(original_image.shape[0],
                                                                                        self.n_samples,
                                                                                        shape[1], shape[2], shape[3])
        log["labels"]["samples_noisy"] = self.tonp(decoded_samples_noisy).reshape(original_image.shape[0],
                                                                                  self.n_samples,
                                                                                  shape[1], shape[2], shape[3])
        log["labels"]["samples_original"] = self.tonp(decoded_samples_original).reshape(original_image.shape[0],
                                                                                        self.n_samples,
                                                                                        shape[1], shape[2], shape[3])
        return {"eval_op": log}


def visualization(root, data_in, data_out, config):
    # visualizaiton of attacks, maybe add a 'callbacks.py' file
    import edflow
    import numpy as np
    epsilon = retrieve(config, "Attack/epsilon", default=1.0)
    logger = edflow.get_logger("attack_visualization")
    logger.info("fgsm @ {}".format(epsilon))

    acc = np.mean(data_in.labels["class"] ==
                  data_out.labels["original_logits"].squeeze().argmax(1))
    logger.info("{:20}: {:.2}".format("original accuracy", acc))

    attacked_acc = np.mean(data_in.labels["class"] ==
                           data_out.labels["attacked_logits"].squeeze().argmax(1))
    logger.info("{:20}: {:.2}".format("attacked accuracy", attacked_acc))

    original_pred = data_out.labels["original_logits"].squeeze().argmax(1)
    attacked_pred = data_out.labels["attacked_logits"].squeeze().argmax(1)
    gt = data_in.labels["class"]
    correct_mask = original_pred == gt
    fooled_mask = attacked_pred != original_pred
    example_indices = np.where(correct_mask & fooled_mask)[0]
    np.random.shuffle(example_indices)  # si?
    import matplotlib.pyplot as plt
    import os
    rows = 6
    cols = 1
    subcols = 4
    h_factor = rows / 2
    fig, axes = plt.subplots(rows, cols * subcols, figsize=[12.8, 7.2 * h_factor], dpi=100, constrained_layout=True)
    for i in range(rows):
        for j in range(cols):
            idx = example_indices[cols * i + j]

            # original and prediction
            axes[i, subcols * j + 0].imshow(halfscale(data_in[idx]["image"].squeeze()))
            axes[i, subcols * j + 0].axis('off')
            pred = data_out.labels["original_logits"][idx].squeeze().argmax(0)
            axes[i, subcols * j + 0].set_title("original prediction: {}".format(pred))

            # attacked and prediction
            axes[i, subcols * j + 1].imshow(data_out[idx]["attacked_image"])
            axes[i, subcols * j + 1].axis('off')
            pred = data_out.labels["attacked_logits"][idx].squeeze().argmax(0)
            axes[i, subcols * j + 1].set_title("prediction: {}".format(pred))

            # original reconstruction
            axes[i, subcols * j + 2].imshow(data_out[idx]["recovered_reconstruction"])
            axes[i, subcols * j + 2].axis('off')
            axes[i, subcols * j + 2].set_title("reconstruction from original z")

            # attacked reconstruction
            axes[i, subcols * j + 3].imshow(data_out[idx]["attacked_reconstruction"])
            axes[i, subcols * j + 3].axis('off')
            axes[i, subcols * j + 3].set_title("reconstruction from attacked z")

    outpath = os.path.join(root, "examples.png")
    logger.info(outpath)
    fig.savefig(outpath)

    return {}


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
        zdec = torch.cat([zin, einrec.detach()], dim=1)[:, :, None, None]  # this will be flowed

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
                    zdec_sample = self.model.reverse(zz_sample, znet).squeeze(-1).squeeze(-1)
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
