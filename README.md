# Making Sense of CNNs: Interpreting Deep Representations & Their Invariances with Invertible Neural Networks

PyTorch code accompanying the [ECCV 2020](https://eccv2020.eu/) paper

[**Making Sense of CNNs: Interpreting Deep Representations & Their Invariances with Invertible Neural Networks**](https://compvis.github.io/invariances/)<br/>
[Robin Rombach](https://github.com/rromb)\*,
[Patrick Esser](https://github.com/pesser)\*,
[Björn Ommer](https://hci.iwr.uni-heidelberg.de/Staff/bommer)<br/>
\* equal contribution

![teaser](data/overview.png)<br/>
[arXiv](https://arxiv.org/) | [BibTeX](#bibtex) | [Project Page](https://compvis.github.io/invariances/)

Table of Contents
=================

* [Requirements](#requirements)
* [Demos](#demos)
* [Data](#data)
* [Training](#training)
   * [Conditional INNs](#conditional-inns)
   * [Invertible Interpretation Networks](#invertible-interpretation-networks)
* [Evaluation](#evaluation)
* [Pretrained Models](#pretrained-models)
* [Results](#results)
* [BibTeX](#bibtex)


## Requirements
A suitable [conda](https://conda.io/) environment named `invariances` can be created
and activated with:

```
conda env create -f environment.yaml
conda activate invariances
```

Optionally, you can then also `conda install tensorflow-gpu=1.14` to speed up
FID evaluations.

## Demos
To get started you can directly dive into some demos. Please note that
checkpoints will be downloaded on demand, which can take a while. You can see
the download progress displayed in the terminal running the streamlit command.
Note that the demonstration scripts are highly customizable, and you can add 
your own images into `data/custom`.
### Feature Visualization
To run a interactive demo that allows to visualize representations and their invariances of AlexNet,
run 
```
streamlit run invariances/demo.py
```

### Adversarial Attacks
coming soon

## Data
If not present on your disk, all required datasets (*ImageNet*, *AnimalFaces* and *ImageNetAnimals*)
 will be downloaded and prepared automatically. The data processing and loading rely on the
  [autoencoders](https://github.com/edflow/autoencoders) package and are described in more detail 
  [here](https://github.com/edflow/autoencoders#data).
  
  **Note:** If you already have one or more of the datasets present, follow the instructions linked 
  above to avoid downloading them again.
 
## Training

### Invariances of Classifiers

#### ResNet

To recover invariances of an ResNet classifier trained on the [AnimalFaces](https://github.com/edflow/autoencoders#animalfaces)
 dataset, run

```
edflow -b configs/resnet/animalfaces/base.yaml configs/resnet/animalfaces/train/<layer>.yaml -t
```

where `<layer>` is one of `input`, `maxpool`, `layer1`, `layer2`, `layer3`, `layer4`, 
`avgpool`, `fc`, `softmax`.
To enable logging to [wandb](https://wandb.ai), adjust
`configs/project.yaml` and add it to above command:

```
edflow -b configs/resnet/animalfaces/base.yaml configs/resnet/animalfaces/train/<layer>.yaml configs/project.yaml -t
```

#### AlexNet

To reproduce the training procedure from the paper, run

```
edflow -b configs/alexnet/base_train.yaml configs/alexnet/train/<layer>.yaml -t
```

where `<layer>` is one of `conv5`, `fc6`, `fc7`, `fc8`, `softmax`.
To enable logging to [wandb](https://wandb.ai), adjust
`configs/project.yaml` and add it to above command:

```
edflow -b configs/alexnet/base_train.yaml configs/alexnet/train/<layer>.yaml configs/project.yaml -t
```


## Evaluation

Evaluations run automatically after each epoch of training. To start an
evaluation manually, run

```
edflow -p logs/<log_folder>/configs/<config>.yaml
```

and, optionally, add `-c <path to checkpoint>` to evaluate a specific
checkpoint instead of the last one.


## Pretrained Models
Pretrained models (e.g. autoencoders and classifiers) will be downloaded automatically on their first 
use in a demo, training or evaluation script. 

## BibTeX

```
@inproceedings{esser2020invertible,
  title={A Disentangling Invertible Interpretation Network for Explaining Latent Representations},
  author={Esser, Patrick and Rombach, Robin and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2020}
}
```
