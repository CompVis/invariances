import torch
import streamlit as st
import numpy as np
# from edflow.util.edexplore import isimage, st_get_list_or_dict_item
from autoencoders.models.bigae import BigAE

from invariances.util.ckpt_util import URL_MAP
from invariances.model.cinn import ConditionalTransformer
from invariances.greybox.models import AlexNetClassifier

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


def visualize(ex, config):
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

    st.write("Options")
    if torch.cuda.is_available():
        gpu = st.checkbox("gpu", value=True)
    else:
        gpu = False

    cinn_layer = st.selectbox("Which layer do you want to visualize?", ("conv5", "fc6", "fc7", "fc8", "softmax"))
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

    num_samples = st.slider("number of samples", 2, 16, 4)
    outputs = {"reconstruction": ae_model.decode(zae)}

    def sample():
        for n in range(num_samples):
            zzsample = torch.randn_like(zz)
            zsample_ae = cinn_model.reverse(zzsample, zrep)
            outputs["sample{}".format(n)] = ae_model.decode(zsample_ae)
        return outputs

    outputs = sample()
    if st.checkbox("Resample Visualizations"):
        outputs = sample()
    for k in outputs:
        outputs[k] = outputs[k].detach().cpu().numpy().transpose(0, 2, 3, 1)

    xrec = rescale(outputs["reconstruction"])
    inrec = np.concatenate((rescale(image)[None, :, :, :], xrec))
    st.write("Input & Reconstruction")
    st.image(inrec)

    # concat the samples, then display
    samples = np.concatenate([rescale(outputs["sample{}".format(n)]) for n in range(num_samples)])
    st.write("Samples")
    st.image(samples)


if __name__ == "__main__":
    from autoencoders.data import Folder

    dset = Folder({"Folder":{
                        "folder": "data/custom",
                        "size": 128}
                 })

    dataidx = st.slider("data index", 0, len(dset), 0)
    example = dset[dataidx]
    visualize(example, None)
