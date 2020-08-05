import torch
import streamlit as st
import numpy as np
from streamlit import caching
# from edflow.util.edexplore import isimage, st_get_list_or_dict_item
from autoencoders.models.bigae import BigAE

from invariances.util.ckpt_util import URL_MAP
from invariances.util.label_util import ANIMALFACES10_TO_HUMAN
from invariances.model.cinn import ConditionalTransformer
from invariances.greybox.models import ResnetClassifier


rescale = lambda x: (x + 1.) / 2.


@st.cache(allow_output_mutation=True)
def get_ae_state(gpu, name="animals"):
    model = BigAE.from_pretrained(name)
    if gpu:
        model.cuda()
    state = {"model": model.eval()}
    return state


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


@st.cache(allow_output_mutation=True)
def get_cinn_state(gpu, name="cinn_resnet_animalfaces10_ae_layer2"):
    caching.clear_cache()
    assert name in URL_MAP
    model = ConditionalTransformer.from_pretrained(name)
    if gpu:
        model.cuda()
    state = {"model": model.eval()}
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


def visualize(example, config):
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

    cinn_layer = st.sidebar.selectbox("Which layer do you want to visualize?", ("maxpool", "layer1", "layer2",
                                                                                "layer3", "layer4", "avgpool",
                                                                                "fc", "softmax"))

    # prepare models
    ae_model = get_ae_state(gpu=gpu, name="animalfaces")["model"]
    cinn_model = get_cinn_state(gpu, layer_dir[cinn_layer]["load_key"])["model"]

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
    cls = torch.tensor([example["class_label"]]).long()
    if gpu:
        cls = cls.cuda()
    human_label = example["human_label"]

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
    true_class = ANIMALFACES10_TO_HUMAN[cls.item()]
    predicted_class_original = torch.argmax(original_logits).item()
    predicted_class_original = ANIMALFACES10_TO_HUMAN[predicted_class_original]
    with torch.no_grad():
        attacked_logits = resnet_classifier(attacked_image)
    predicted_class_attacked = torch.argmax(attacked_logits).item()
    predicted_class_attacked = ANIMALFACES10_TO_HUMAN[predicted_class_attacked]

    # plot of attack
    atta_noise = rescale(attacked_noise.cpu().detach().numpy().transpose(0, 2, 3, 1))
    atta_img = rescale(attacked_image.cpu().detach().numpy().transpose(0, 2, 3, 1))
    input_plus_attack = np.concatenate((rescale(original_image)[None, :, :, :], atta_noise, atta_img))
    st.write("__{}__ $+$ Attack $=$ Attacked Image".format(true_class))
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
    from autoencoders.data import AnimalFacesRestrictedTest
    dset = AnimalFacesRestrictedTest({"size":128})
    dataidx = st.slider("data index", 0, len(dset)-1, 0)
    example = dset[dataidx]
    visualize(example, None)
