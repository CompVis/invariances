import torch
import streamlit as st
import numpy as np
from streamlit import caching
import io
import imageio
from PIL import Image
import autoencoders
from autoencoders.models.bigae import BigAE
from autoencoders.data import Folder

from invariances.data import LabelFolder
from invariances.util.ckpt_util import URL_MAP
from invariances.util.label_util import ANIMALFACES10_TO_HUMAN
from invariances.model.cinn import ConditionalTransformer
from invariances.greybox.models import AlexNetClassifier, ResnetClassifier

import torch.nn as nn


@st.cache(allow_output_mutation=True)
def get_state(key):
    return {"key": key, "previous": None, "current": None}

def clear_on_change(key, value):
    state = get_state(key)
    state["current"] = value
    if state["previous"] is not None and state["current"] != state["previous"]:
        print(state)
        caching.clear_cache()
        state = get_state(key)
        state["current"] = value
    state["previous"] = state["current"]

rescale = lambda x: (x + 1.) / 2.


@st.cache(allow_output_mutation=True)
def get_ae_state(gpu, name="animals"):
    model = BigAE.from_pretrained(name)
    if gpu:
        model.cuda()
    model.eval()
    state = {"model": model}
    return state


@st.cache(allow_output_mutation=True)
def get_alex_state(gpu, split_idx):
    model = AlexNetClassifier({"split_idx": split_idx,
                               "subconfig": {
                                   "model": "invariances.greybox.classifiers.AlexNet"
                                                }
                               })
    if gpu:
        model.cuda()
    model.eval()
    state = {"model": model}
    return state


@st.cache(allow_output_mutation=True)
def get_cinn_state(gpu, name="cinn_alexnet_aae_conv5"):
    assert name in URL_MAP
    model = ConditionalTransformer.from_pretrained(name)
    if gpu:
        model.cuda()
    model.eval()
    state = {"model": model}
    return state


def visualize(example, config):
    layer_dir = {"conv5": {"index": 12,
                           "load_key": "cinn_alexnet_aae_conv5"},
                 "fc6": {"index": 18,
                         "load_key": "cinn_alexnet_aae_fc6"},
                 "fc7": {"index": 21,
                         "load_key": "cinn_alexnet_aae_fc7"},
                 "fc8": {"index": 22,
                         "load_key": "cinn_alexnet_aae_fc8"},
                 "softmax": {"index": 23,
                             "load_key": "cinn_alexnet_aae_softmax"},
                 }

    st.sidebar.text("Options")
    if torch.cuda.is_available():
        gpu = st.sidebar.checkbox("gpu", value=True)
    else:
        gpu = False

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?",
                                      ("conv5", "fc6", "fc7", "fc8", "softmax"))

    ae_model = get_ae_state(gpu=gpu, name="animals")["model"]
    alex_model = get_alex_state(gpu, layer_dir[cinn_layer]["index"])["model"]
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]

    image = example["image"]
    xin = torch.tensor(image)[None, ...].transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()

    zrep = alex_model.encode(xin).sample()
    zae = ae_model.encode(xin).sample()
    zz, _ = cinn_model(zae, zrep)

    num_samples = st.sidebar.slider("number of samples", 2, 16, 8)
    st.sidebar.button("Resample Visualizations")

    outputs = {"reconstruction": ae_model.decode(zae)}

    def sample():
        for n in range(num_samples):
            zzsample = torch.randn_like(zz)
            zsample_ae = cinn_model.reverse(zzsample, zrep)
            outputs["sample{}".format(n)] = ae_model.decode(zsample_ae)
        return outputs

    outputs = sample()
    for k in outputs:
        outputs[k] = outputs[k].detach().cpu().numpy().transpose(0, 2, 3, 1)

    xrec = rescale(outputs["reconstruction"])
    inrec = np.concatenate((rescale(image)[None, :, :, :], xrec))
    st.write("Input & Autoencoder Reconstruction")
    st.image(inrec)

    # concat the samples, then display
    samples = np.concatenate([rescale(outputs["sample{}".format(n)]) for n in range(num_samples)])
    st.write("cINN samples of input reconstructions from layer __{}__".format(cinn_layer))
    st.image(samples)


def recombine(example, config):
    layer_dir = {"input": {"index": 0,
                           "load_key": "cinn_resnet_animalfaces10_ae_input"},
                 "maxpool": {"index": 4,
                             "load_key": "cinn_resnet_animalfaces10_ae_maxpool"},
                 "layer1": {"index": 5,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer1"},
                 "layer2": {"index": 6,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer2"},
                 "layer3": {"index": 7,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer3"},
                 "layer4": {"index": 8,
                            "load_key": "cinn_resnet_animalfaces10_ae_layer4"},
                 "avgpool": {"index": 9,
                            "load_key": "cinn_resnet_animalfaces10_ae_avgpool"},
                 "fc": {"index": 11,
                             "load_key": "cinn_resnet_animalfaces10_ae_fc"},
                 "softmax": {"index": 12,
                            "load_key": "cinn_resnet_animalfaces10_ae_softmax"},
                 }

    st.sidebar.text("Options")
    if torch.cuda.is_available():
        gpu = st.sidebar.checkbox("gpu", value=True)
    else:
        gpu = False

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?",
                                      ("maxpool", "layer1", "layer2",
                                       "layer3", "layer4", "avgpool",
                                       "fc", "softmax"), 7)
    clear_on_change("layer", cinn_layer)

    # prepare models
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]
    ae_model = get_ae_state(gpu=gpu, name="animalfaces")["model"]
    resnet_greybox = get_resnet_state(gpu, layer_dir[cinn_layer]["index"])["greybox"]

    image1 = example["example1"]["image"]
    image2 = example["example2"]["image"]
    images = np.stack([image1, image2])
    xin = torch.tensor(images).transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()

    zrep = resnet_greybox.encode(xin).sample()
    zae = ae_model.encode(xin).sample()
    zz, _ = cinn_model(zae, zrep)
    zae_recombined = cinn_model.reverse(zz, torch.flip(zrep, [0]))
    x_recombined = ae_model.decode(zae_recombined)

    def toimg(x):
        return rescale(x.detach().cpu().numpy().transpose(0, 2, 3, 1))

    st.text("Inputs")
    st.image(toimg(xin))
    st.text("Recombinations")
    st.image(toimg(x_recombined))


def video_demo(example, videoreader):
    layer_dir = {"input": {"index": 0,
                           "load_key": "cinn_resnet_animalfaces10_ae_input"},
                 "maxpool": {"index": 4,
                             "load_key": "cinn_resnet_animalfaces10_ae_maxpool"},
                 "layer1": {"index": 5,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer1"},
                 "layer2": {"index": 6,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer2"},
                 "layer3": {"index": 7,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer3"},
                 "layer4": {"index": 8,
                            "load_key": "cinn_resnet_animalfaces10_ae_layer4"},
                 "avgpool": {"index": 9,
                            "load_key": "cinn_resnet_animalfaces10_ae_avgpool"},
                 "fc": {"index": 11,
                             "load_key": "cinn_resnet_animalfaces10_ae_fc"},
                 "softmax": {"index": 12,
                            "load_key": "cinn_resnet_animalfaces10_ae_softmax"},
                 }

    st.sidebar.text("Options")
    if torch.cuda.is_available():
        gpu = st.sidebar.checkbox("gpu", value=True)
    else:
        gpu = False

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?",
                                      ("maxpool", "layer1", "layer2",
                                       "layer3", "layer4", "avgpool",
                                       "fc", "softmax"), 7)
    clear_on_change("layer", cinn_layer)

    # prepare models
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]
    ae_model = get_ae_state(gpu=gpu, name="animalfaces")["model"]

    resnet_greybox = get_resnet_state(gpu, layer_dir[cinn_layer]["index"])["greybox"]

    outvid = "video_translation.mp4"
    writer = imageio.get_writer(outvid, fps=25)

    xin = torch.tensor(example["image"])[None,...].transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()
    zrep = resnet_greybox.encode(xin).mode()

    st.text("Invariances Input")
    displayinv = st.empty()
    st.text("Representation Input")
    st.image((example["image"]+1)/2)
    st.text("Recovered Input")
    displayrec = st.empty()
    frame = st.empty()
    for i, im in enumerate(videoreader):
        im = Image.fromarray(im)
        im = im.resize((128,128))
        im = np.array(im)
        displayinv.image(im)
        frame.text("Frame {}".format(i))

        xinv = torch.tensor(im/127.5-1.0)[None,...].transpose(3, 2).transpose(2, 1).float()
        if gpu:
            xinv = xinv.cuda()
        zinvrep = resnet_greybox.encode(xinv).mode()
        zinvae = ae_model.encode(xinv).mode()
        zinvz, _ = cinn_model(zinvae, zinvrep)

        zae_recombined = cinn_model.reverse(zinvz, zrep)
        x_recombined = ae_model.decode(zae_recombined)

        x_recombined = x_recombined.detach().cpu().numpy().transpose(0,2,3,1)
        x_recombined = ((x_recombined+1.0)*127.5).astype(np.uint8)

        displayrec.image(x_recombined)
        writer.append_data(x_recombined[0])

    writer.close()
    st.text("Final Video")
    st.video(outvid)


@st.cache(allow_output_mutation=True)
def get_texturebias_models(gpu):
    vanilla_cfg = {
        "model": "invariances.greybox.classifiers.ResNet",
        "Model": {
            "n_classes": 1000,
            "type": "resnet50",
            "custom_head": False
        }
    }
    stylized_cfg = {
        "model": "invariances.greybox.classifiers.ResNet",
        "Model": {
            "n_classes": 1000,
            "type": "resnet50stylized",
            "custom_head": False
        }
    }

    models = {
        "decoder": autoencoders.get_model("biggan_256"),
        "vanilla": ResnetClassifier({"subconfig": vanilla_cfg,
                                     "split_idx": 9}), # avgpool
        "stylized": ResnetClassifier({"subconfig": stylized_cfg,
                                      "split_idx": 9}), # avgpool
        "vanilla_cinn": ConditionalTransformer.from_pretrained("cinn_resnet_avgpool"),
        "stylized_cinn": ConditionalTransformer.from_pretrained("cinn_stylizedresnet_avgpool"),
    }

    if gpu:
        for model in models.values():
            model.cuda()
    for model in models.values():
        model.eval()
    return models


def texturebias(example, config):
    st.sidebar.text("Options")
    if torch.cuda.is_available():
        gpu = st.sidebar.checkbox("gpu", value=True)
    else:
        gpu = False

    num_samples = st.sidebar.slider("number of samples", 2, 16, 2)
    st.sidebar.button("Resample Visualizations")

    models = get_texturebias_models(gpu)
    decoder_model = models["decoder"]
    vanilla_model = models["vanilla"]
    stylized_model = models["stylized"]
    cinn_vanilla_model = models["vanilla_cinn"]
    cinn_stylized_model = models["stylized_cinn"]

    image = example["image"]
    xin = torch.tensor(image)[None, ...].transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()

    z_vanilla_rep = vanilla_model.encode(xin).sample()
    z_stylized_rep = stylized_model.encode(xin).sample()

    outputs = {"input": xin}

    chunk_sizes = (140, 128)
    zzshape = (1,sum(chunk_sizes),1,1)
    def decode(z):
        z = z.squeeze(-1).squeeze(-1)
        z_z, z_emb = torch.split(z, chunk_sizes, dim=1)
        return decoder_model(z_z, z_emb, from_class_embedding=True)

    def sample():
        for n in range(num_samples):
            zzsample = torch.randn(zzshape).to(xin)
            zsample_vanilla_ae = cinn_vanilla_model.reverse(zzsample, z_vanilla_rep)
            zsample_stylized_ae = cinn_stylized_model.reverse(zzsample, z_stylized_rep)
            outputs["sample_vanilla_{}".format(n)] = decode(zsample_vanilla_ae)
            outputs["sample_stylized_{}".format(n)] = decode(zsample_stylized_ae)
        return outputs

    sample()

    for k in outputs:
        outputs[k] = rescale(outputs[k].detach().cpu().numpy().transpose(0, 2, 3, 1))

    st.write("Input")
    st.image(outputs["input"])

    # concat the samples, then display
    samples = np.concatenate([outputs["sample_vanilla_{}".format(n)] for n in range(num_samples)])
    st.write("cINN samples of input reconstructions from __vanilla ResNet__")
    st.image(samples)
    samples = np.concatenate([outputs["sample_stylized_{}".format(n)] for n in range(num_samples)])
    st.write("cINN samples of input reconstructions from __stylized ResNet__")
    st.image(samples)


@st.cache(allow_output_mutation=True)
def get_resnet_state(gpu, split_idx, pretrained_key="resnet101_animalfaces_10"):
    greybox = ResnetClassifier({"split_idx": split_idx,
                                "subconfig": {
                                   "pretrained_key": pretrained_key,
                                   "model": "invariances.greybox.classifiers.ResNet"
                                                }
                               })
    model = greybox.prepare(model_as_str="invariances.greybox.classifiers.ResNet",
                                     pretrained_key=pretrained_key)
    if gpu:
        greybox.cuda()
        model.cuda()
    state = {"greybox": greybox.eval(), "classifier": model.eval()}
    return state


class AttackFGSM(object):
    def __init__(self):
        self.loss = torch.nn.CrossEntropyLoss()

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
        """
            a random pertubation
        """
        # Collect the element-wise sign of the data gradient
        fake_grad = torch.tensor(np.random.RandomState(1).randn(*data_grad.shape)).to(data_grad)
        sign_data_grad = fake_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_image = image + epsilon * sign_data_grad
        # Adding clipping to maintain [-1,1] range
        perturbed_image = torch.clamp(perturbed_image, -1, 1)
        # Return the perturbed image
        return perturbed_image, sign_data_grad


def visualize_attacks(example, config):
    layer_dir = {"input": {"index": 0,
                           "load_key": "cinn_resnet_animalfaces10_ae_input"},
                 "maxpool": {"index": 4,
                             "load_key": "cinn_resnet_animalfaces10_ae_maxpool"},
                 "layer1": {"index": 5,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer1"},
                 "layer2": {"index": 6,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer2"},
                 "layer3": {"index": 7,
                             "load_key": "cinn_resnet_animalfaces10_ae_layer3"},
                 "layer4": {"index": 8,
                            "load_key": "cinn_resnet_animalfaces10_ae_layer4"},
                 "avgpool": {"index": 9,
                            "load_key": "cinn_resnet_animalfaces10_ae_avgpool"},
                 "fc": {"index": 11,
                             "load_key": "cinn_resnet_animalfaces10_ae_fc"},
                 "softmax": {"index": 12,
                            "load_key": "cinn_resnet_animalfaces10_ae_softmax"},
                 }

    st.sidebar.text("Options")
    eps = float(st.sidebar.text_input("fgsm_epsilon", 0.02))
    if torch.cuda.is_available():
        gpu = st.sidebar.checkbox("gpu", value=True)
    else:
        gpu = False

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?",
                                      ("maxpool", "layer1", "layer2",
                                       "layer3", "layer4", "avgpool",
                                       "fc", "softmax"), 7)
    clear_on_change("layer", cinn_layer)

    # prepare models
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]
    ae_model = get_ae_state(gpu=gpu, name="animalfaces")["model"]

    resnet_greybox = get_resnet_state(gpu, layer_dir[cinn_layer]["index"])["greybox"]
    resnet_classifier = get_resnet_state(gpu, layer_dir[cinn_layer]["index"])["classifier"]   # used for the attack

    original_image = example["image"]
    xin = torch.tensor(original_image)[None, ...].transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()

    # get original representation
    original_z = resnet_greybox.encode(xin).sample()
    # prepare attack
    xin.requires_grad = True
    original_logits = resnet_classifier(xin)
    predicted_class_original = torch.argmax(original_logits)
    cls = torch.tensor([predicted_class_original]).long()
    predicted_class_original = ANIMALFACES10_TO_HUMAN[predicted_class_original.item()]

    if gpu:
        cls = cls.cuda()

    # attack
    attackor = AttackFGSM()
    loss = attackor.loss(original_logits, cls)
    loss.backward()

    grad = xin.grad.data
    attacked_image, attacked_noise = attackor.fgsm_attack(xin, eps, grad)
    noisy_image, noisy_noise = attackor.no_attack(xin, eps, grad)

    # encode attacked image to z
    attacked_z = resnet_greybox.encode(attacked_image).sample()

    # encode noisy image to z
    noisy_z = resnet_greybox.encode(noisy_image).sample()

    # encode original image with autoencoder
    original_z_ae = ae_model.encode(xin).sample()

    # flow it into second stage z ss_z conditioned on original
    # representation
    original_ss_z, _ = cinn_model(original_z_ae, original_z)

    # invert (for debugging, should be == original_z_ae)
    recovered_z_ae = cinn_model.reverse(original_ss_z, original_z)
    # swap conditioning to attacked one
    attacked_z_ae = cinn_model.reverse(original_ss_z, attacked_z)
    # swap conditioning to noisy one
    noisy_z_ae = cinn_model.reverse(original_ss_z, noisy_z)

    num_samples = st.sidebar.slider("number of samples", 2, 6, 3)
    outputs = {"reconstruction": ae_model.decode(original_z_ae),
               "reconstruction_original_z": ae_model.decode(recovered_z_ae),
               "reconstruction_attacked_z": ae_model.decode(attacked_z_ae),
               "reconstruction_noisy_z": ae_model.decode(noisy_z_ae)}

    def sample():
        for n in range(num_samples):
            zzsample = torch.randn_like(original_ss_z)
            zsample_original_ae = cinn_model.reverse(zzsample, original_z)
            zsample_attacked_ae = cinn_model.reverse(zzsample, attacked_z)
            outputs["sample_original{}".format(n)] = ae_model.decode(zsample_original_ae)
            outputs["sample_attacked{}".format(n)] = ae_model.decode(zsample_attacked_ae)
        return outputs

    outputs = sample()
    if st.sidebar.button("Resample Visualizations"):
        outputs = sample()
    for k in outputs:
        outputs[k] = outputs[k].detach().cpu().numpy().transpose(0, 2, 3, 1)

    # analysis of attack
    with torch.no_grad():
        attacked_logits = resnet_classifier(attacked_image)
    predicted_class_attacked = torch.argmax(attacked_logits).item()
    predicted_class_attacked = ANIMALFACES10_TO_HUMAN[predicted_class_attacked]

    # plot of attack
    atta_noise = rescale(attacked_noise.cpu().detach().numpy().transpose(0, 2, 3, 1))
    atta_img = rescale(attacked_image.cpu().detach().numpy().transpose(0, 2, 3, 1))
    input_plus_attack = np.concatenate((rescale(original_image)[None, :, :, :], atta_noise, atta_img))
    st.write("Input $+$ Attack $=$ Attacked Image")
    st.image(input_plus_attack)
    st.write("Predicted from Original: __{}__".format(predicted_class_original))
    st.write("Predicted from Attacked Image: __{}__".format(predicted_class_attacked))

    # display the reconstructions
    st.text("What a human/the network sees:               (+ sanity check: AE reconstruction)")
    recons = np.concatenate((atta_img,
                             rescale(outputs["reconstruction_attacked_z"]),
                             np.ones_like(atta_img),
                             rescale(outputs["reconstruction_original_z"])
                             ))
    st.image(recons)

    # concat the samples, then display
    samples_original = np.concatenate([rescale(outputs["sample_original{}".format(n)]) for n in range(num_samples)])
    st.write("Samples conditioned on original network representation")
    st.image(samples_original)

    samples_attacked = np.concatenate([rescale(outputs["sample_attacked{}".format(n)]) for n in range(num_samples)])
    st.write("Samples conditioned on attacked network representation")
    st.image(samples_attacked)


if __name__ == "__main__":
    st.title("Making Sense of CNNs")

    demo = st.sidebar.selectbox(
        "Demo",
        ["Visualization of Adversarial Attacks",
         "Visualization of Network Representations",
         "Revealing Texture Bias",
         "Visualizing Invariances from a Video",
         "Image Mixing",
         ],
    )
    clear_on_change("demo", demo)

    if demo == "Visualization of Network Representations":
        st.header("Visualizing Network Representations from AlexNet")
        dset = Folder({
            "Folder":{
                "folder": "data/custom",
                "size": 128}
        })

        dataidx = st.slider("Example", 0, len(dset)-1, len(dset)-1)
        example = dset[dataidx]
        visualize(example, None)
    elif demo == "Image Mixing":
        st.header("Image Mixing via Their Invariances")
        dset = LabelFolder({"Folder": {"folder": "data/animalfaces10",
                                       "size": 128,
                                       "label_level": 2}})

        dataidx1 = st.slider("Example 1", 0, len(dset)-1, 97)
        dataidx2 = st.slider("Example 2", 0, len(dset)-1, 587)
        example1 = dset[dataidx1]
        example2 = dset[dataidx2]
        example = {"example1": example1, "example2": example2}
        recombine(example, None)
    elif demo == "Visualizing Invariances from a Video":
        st.header("Visualizing Invariances from a Video")
        dset = LabelFolder({"Folder": {"folder": "data/animalfaces10",
                                       "size": 128,
                                       "label_level": 2}})

        st.set_option('deprecation.showfileUploaderEncoding', False)

        dataidx = st.slider("Example", 0, len(dset)-1, 618)
        example = dset[dataidx]
        video = st.file_uploader("Driving Video", type=["mp4", "mkv", "avi"])
        if video is None:
            video = open("data/doggo2short02.mp4", "rb")
        reader = imageio.get_reader(video, "ffmpeg")
        video_demo(example, reader)
    elif demo == "Revealing Texture Bias":
        st.header("Revealing the Texture Bias on ResNets")
        dset = Folder({
            "Folder":{
                "folder": "data/texturebias",
                "size": 256}
        })

        dataidx = st.slider("Example", 0, len(dset)-1, 0)
        example = dset[dataidx]
        texturebias(example, None)
    elif demo == "Visualization of Adversarial Attacks":
        st.header("Visualization of Adversarial Attacks")
        dset = LabelFolder({"Folder": {"folder": "data/animalfaces10",
                                       "size": 128,
                                       "label_level": 2}})
        dataidx = st.slider("data index", 0, len(dset)-1, 18)
        example = dset[dataidx]
        visualize_attacks(example, None)
