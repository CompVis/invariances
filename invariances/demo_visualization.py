import torch
import streamlit as st
import numpy as np
# from edflow.util.edexplore import isimage, st_get_list_or_dict_item
import autoencoders
from autoencoders.models.bigae import BigAE

from invariances.util.ckpt_util import URL_MAP
from invariances.model.cinn import ConditionalTransformer
from invariances.greybox.models import AlexNetClassifier, ResnetClassifier

import torch.nn as nn

rescale = lambda x: (x + 1.) / 2.


@st.cache(allow_output_mutation=True)
def get_ae_state(gpu, name="animals"):
    model = BigAE.from_pretrained(name)
    if gpu:
        model.cuda()
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
    state = {"model": model}
    return state


def visualize(example, config):
    layer_dir = {"conv5": {"index": 12,  # 10 is interesting, 12 works
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

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?", ("conv5", "fc6", "fc7", "fc8", "softmax"))
    ae_model = get_ae_state(gpu=gpu, name="animals")["model"]
    alex_model = get_alex_state(gpu, layer_dir[cinn_layer]["index"])["model"]
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]

    # image, image_key = st_get_list_or_dict_item(ex, "image", description="input image", filter_fn=isimage)
    image = example["image"]
    xin = torch.tensor(image)[None, ...].transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()

    zrep = alex_model.encode(xin).sample()
    zae = ae_model.encode(xin).sample()
    zz, _ = cinn_model(zae, zrep)

    num_samples = st.sidebar.slider("number of samples", 2, 16, 4)
    outputs = {"reconstruction": ae_model.decode(zae)}

    def sample():
        for n in range(num_samples):
            zzsample = torch.randn_like(zz)
            zsample_ae = cinn_model.reverse(zzsample, zrep)
            outputs["sample{}".format(n)] = ae_model.decode(zsample_ae)
        return outputs

    outputs = sample()
    if st.sidebar.button("Resample Visualizations"):
        outputs = sample()
    for k in outputs:
        outputs[k] = outputs[k].detach().cpu().numpy().transpose(0, 2, 3, 1)

    xrec = rescale(outputs["reconstruction"])
    inrec = np.concatenate((rescale(image)[None, :, :, :], xrec))
    st.text("Input & Autoencoder Reconstruction")
    st.image(inrec)

    # concat the samples, then display
    samples = np.concatenate([rescale(outputs["sample{}".format(n)]) for n in range(num_samples)])
    st.text("cINN Samples of Input Reconstructions from Layer {}".format(cinn_layer))
    st.image(samples)


def recombine(example, config):
    layer_dir = {"conv5": {"index": 12,  # 10 is interesting, 12 works
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

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?", ("conv5", "fc6", "fc7", "fc8", "softmax"))
    ae_model = get_ae_state(gpu=gpu, name="animals")["model"]
    alex_model = get_alex_state(gpu, layer_dir[cinn_layer]["index"])["model"]
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]

    image1 = example["example1"]["image"]
    image2 = example["example2"]["image"]
    images = np.stack([image1, image2])
    xin = torch.tensor(images).transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()

    zrep = alex_model.encode(xin).sample()
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
    layer_dir = {"conv5": {"index": 12,  # 10 is interesting, 12 works
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

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?", ("conv5", "fc6", "fc7", "fc8", "softmax"))
    ae_model = get_ae_state(gpu=gpu, name="animals")["model"]
    alex_model = get_alex_state(gpu, layer_dir[cinn_layer]["index"])["model"]
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]

    #######################
    import io
    import imageio
    from PIL import Image
    outvid = "tmpvid.mp4"
    writer = imageio.get_writer(outvid, fps=25)

    xin = torch.tensor(example["image"])[None,...].transpose(3, 2).transpose(2, 1).float()
    if gpu:
        xin = xin.cuda()
    zrep = alex_model.encode(xin).mode()

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
        zinvrep = alex_model.encode(xinv).mode()
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

    st.text("Input")
    st.image(outputs["input"])

    # concat the samples, then display
    samples = np.concatenate([outputs["sample_vanilla_{}".format(n)] for n in range(num_samples)])
    st.text("cINN Samples of Input Reconstructions from Vanilla ResNet")
    st.image(samples)
    samples = np.concatenate([outputs["sample_stylized_{}".format(n)] for n in range(num_samples)])
    st.text("cINN Samples of Input Reconstructions from Stylized ResNet")
    st.image(samples)


if __name__ == "__main__":
    demo = st.sidebar.selectbox(
        "Demo",
        ["samples", "recombinations", "video", "texturebias"],
        3,
    )
    from autoencoders.data import Folder

    st.title("Making Sense of CNNs")

    if demo == "samples":
        dset = Folder({
            "Folder":{
                "folder": "data/custom",
                "size": 128}
        })

        dataidx = st.sidebar.slider("Example", 0, len(dset)-1, 0)
        example = dset[dataidx]
        visualize(example, None)
    elif demo == "recombinations":
        dset = Folder({
            "Folder":{
                "folder": "data/custom",
                "size": 128}
        })

        dataidx1 = st.sidebar.slider("Example 1", 0, len(dset)-1, 0)
        dataidx2 = st.sidebar.slider("Example 2", 0, len(dset)-1, 0)
        example1 = dset[dataidx1]
        example2 = dset[dataidx2]
        example = {"example1": example1, "example2": example2}
        recombine(example, None)
    elif demo == "video":
        dset = Folder({
            "Folder":{
                "folder": "data/custom",
                "size": 128}
        })

        st.set_option('deprecation.showfileUploaderEncoding', False)
        import imageio

        dataidx = st.sidebar.slider("Example", 0, len(dset)-1, 0)
        example = dset[dataidx]
        video = st.sidebar.file_uploader("Driving Video", type=["mp4", "mkv", "avi"])
        if video is None:
            video = open("data/custom/vid.mkv", "rb")
        reader = imageio.get_reader(video, "ffmpeg")
        video_demo(example, reader)
    elif demo == "texturebias":
        dset = Folder({
            "Folder":{
                "folder": "data/texturebias",
                "size": 256}
        })

        dataidx = st.sidebar.slider("Example", 0, len(dset)-1, 0)
        example = dset[dataidx]
        texturebias(example, None)
