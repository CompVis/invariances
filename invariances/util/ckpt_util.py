import os, hashlib
import requests
from tqdm import tqdm


URL_MAP = {
    "cinn_alexnet_aae_conv5": "https://heibox.uni-heidelberg.de/f/62b0e29d8c544f51b79c/?dl=1",
    "cinn_alexnet_aae_fc6": "https://heibox.uni-heidelberg.de/f/5d07dc071dd1450eb0a5/?dl=1",
    "cinn_alexnet_aae_fc7": "https://heibox.uni-heidelberg.de/f/050d4d76f3cf4eeeb9b6/?dl=1",
    "cinn_alexnet_aae_fc8": "https://heibox.uni-heidelberg.de/f/cb9c93497aae4e97890c/?dl=1",
    "cinn_alexnet_aae_softmax": "https://heibox.uni-heidelberg.de/f/5a30088c51b44cc58bbe/?dl=1",
    "cinn_stylizedresnet_avgpool": "https://heibox.uni-heidelberg.de/f/7a54929dee4544248020/?dl=1",
    "cinn_resnet_avgpool": "https://heibox.uni-heidelberg.de/f/d31717bb225b4692bbb3/?dl=1",
    "resnet101_animalfaces_shared": "https://heibox.uni-heidelberg.de/f/a2c36d628f574ed8aa68/?dl=1",
    "resnet101_animalfaces_10": "https://heibox.uni-heidelberg.de/f/314926cb0d754cd9bb02/?dl=1",
    "cinn_resnet_animalfaces10_ae_maxpool": "https://heibox.uni-heidelberg.de/f/30dc2640dfd54b339f93/?dl=1",
    "cinn_resnet_animalfaces10_ae_input": "https://heibox.uni-heidelberg.de/f/abe885bdaf1c46139020/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer1": "https://heibox.uni-heidelberg.de/f/00687f7bc1d6409aa680/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer2": "https://heibox.uni-heidelberg.de/f/51795cdc045246c49cdc/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer3": "https://heibox.uni-heidelberg.de/f/860cfa3327f3426db1f6/?dl=1",
    "cinn_resnet_animalfaces10_ae_layer4": "https://heibox.uni-heidelberg.de/f/1a2058efb9564cb7bcf7/?dl=1",
    "cinn_resnet_animalfaces10_ae_avgpool": "https://heibox.uni-heidelberg.de/f/859c201776ee4b5f8b15/?dl=1",
    "cinn_resnet_animalfaces10_ae_fc": "https://heibox.uni-heidelberg.de/f/b0a485885ab44e4daa44/?dl=1",
    "cinn_resnet_animalfaces10_ae_softmax": "https://heibox.uni-heidelberg.de/f/19c56eff387c40f3bd44/?dl=1",
}

CKPT_MAP = {
    "cinn_alexnet_aae_conv5": "invariances/pretrained_models/cinns/alexnet/conv5.ckpt",
    "cinn_alexnet_aae_fc6": "invariances/pretrained_models/cinns/alexnet/fc6.ckpt",
    "cinn_alexnet_aae_fc7": "invariances/pretrained_models/cinns/alexnet/fc7.ckpt",
    "cinn_alexnet_aae_fc8": "invariances/pretrained_models/cinns/alexnet/fc8.ckpt",
    "cinn_alexnet_aae_softmax": "invariances/pretrained_models/cinns/alexnet/softmax.ckpt",
    "cinn_stylizedresnet_avgpool": "invariances/pretrained_models/cinns/stylized_resnet50/avgpool.ckpt",
    "cinn_resnet_avgpool": "invariances/pretrained_models/cinns/resnet50/avgpool.ckpt",
    "resnet101_animalfaces_shared": "invariances/pretrained_models/classifiers/resnet101/animalfaces149_modelub_16908.ckpt",
    "resnet101_animalfaces_10": "invariances/pretrained_models/classifiers/resnet101/animalfaces10_modelub_6118.ckpt",
    "cinn_resnet_animalfaces10_ae_maxpool": "invariances/pretrained_models/cinns/maxpool_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_input": "invariances/pretrained_models/cinns/input_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer1": "invariances/pretrained_models/cinns/layer1_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer2": "invariances/pretrained_models/cinns/layer2_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer3": "invariances/pretrained_models/cinns/layer3_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_layer4": "invariances/pretrained_models/cinns/layer4_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_avgpool": "invariances/pretrained_models/cinns/avgpool_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_fc": "invariances/pretrained_models/cinns/fc_model-7000.ckpt",
    "cinn_resnet_animalfaces10_ae_softmax": "invariances/pretrained_models/cinns/softmax_model-7000.ckpt",
}

MD5_MAP = {
    "cinn_alexnet_aae_conv5": "ae9367e3cf7486218375c2b328d1273a",
    "cinn_alexnet_aae_fc6": "e7128567ee0686d362ed83cfda9fabc1",
    "cinn_alexnet_aae_fc7": "342ab8f9280ed83f30150e586c7df13b",
    "cinn_alexnet_aae_fc8": "ce6b5bfc316a855693b7e1808d6f3a46",
    "cinn_alexnet_aae_softmax": "15fbdd0d8d51ba0031fbc07ebf0ba2f6",
    "cinn_stylizedresnet_avgpool": "0dd3f262a8179920d77db7404a599e35",
    "cinn_resnet_avgpool": "c956206cadc928febfa7c940425f2edd",
    "resnet101_animalfaces_shared": "784b6b52eb30341484b5fb69e3b9ae60",
    "resnet101_animalfaces_10": "473cd4c25e7dedf87967489992498f06",
    "cinn_resnet_animalfaces10_ae_maxpool": "bc6ef45423b44920a21f5b99b637a77e",
    "cinn_resnet_animalfaces10_ae_input": "fdabc84090536cb86cc7c26ac1882602",
    "cinn_resnet_animalfaces10_ae_layer1": "22bc960b6b17e7ba20a835610c900546",
    "cinn_resnet_animalfaces10_ae_layer2": "36708e3cde454bf76c37300abf90a7e8",
    "cinn_resnet_animalfaces10_ae_layer3": "57001f54ba533af446702bf960644e9f",
    "cinn_resnet_animalfaces10_ae_layer4": "94e9a044fbca68fe8e994e7521be3f2d",
    "cinn_resnet_animalfaces10_ae_avgpool": "0993b79c3d0e7be4be8f99bda82db934",
    "cinn_resnet_animalfaces10_ae_fc": "1fc381a48c9a2a9001609ad1b8a2f941",
    "cinn_resnet_animalfaces10_ae_softmax": "37303f1d9e2ef71ae75bc765415bbb0e",
}

STYLE_MODEL_URLS = {
            'resnet50_trained_on_SIN':
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_trained_on_SIN_and_IN':
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN':
                'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'
            }

ALEXNET_BASE_CONFIG = {
    "Transformer": {
      "activation": "none",
      "conditioning_option": "none",
      "hidden_depth": 2,
      "in_channels": 128,
      "mid_channels": 1024,
      "n_flows": 20
    }
}

CONFIG_MAP = {
    "cinn_alexnet_aae_conv5":
        {"Transformer": {
              "activation": "none",
              "conditioning_option": "none",
              "hidden_depth": 2,
              "in_channels": 128,
              "mid_channels": 1024,
              "n_flows": 20,
              "conditioning_in_channels": 256,
              "conditioning_spatial_size": 13,
              "embedder_down": 2,
            }
        },
    "cinn_alexnet_aae_fc6":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 4096,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_fc7":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 4096,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_fc8":
        {"Transformer": {
                      "activation": "none",
                      "conditioning_option": "none",
                      "hidden_depth": 2,
                      "in_channels": 128,
                      "mid_channels": 1024,
                      "n_flows": 20,
                      "conditioning_in_channels": 1000,
                      "conditioning_spatial_size": 1,
                      "embedder_down": 3,
                    }
                },
    "cinn_alexnet_aae_softmax":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 1000,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "cinn_stylizedresnet_avgpool":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 268,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "cinn_resnet_avgpool":
        {"Transformer": {
            "activation": "none",
            "conditioning_option": "none",
            "hidden_depth": 2,
            "in_channels": 268,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "embedder_down": 3,
            }
        },
    "resnet101_animalfaces_shared":
        {"Model": {
            "n_classes": 149,
            "type": "resnet101"
            }
        },

    "resnet101_animalfaces_10":
        {"Model": {
                "n_classes": 10,
                "type": "resnet101"
                }
        },
    "cinn_resnet_animalfaces10_ae_maxpool":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 64,
            "conditioning_spatial_size": 56,
            "embedder_down": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_input":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 3,
            "conditioning_spatial_size": 224,
            "embedder_down": 5,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer1":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 256,
            "conditioning_spatial_size": 56,
            "embedder_down": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer2":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 512,
            "conditioning_spatial_size": 28,
            "embedder_down": 3,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer3":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 1024,
            "conditioning_spatial_size": 14,
            "embedder_down": 2,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_layer4":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 7,
            "embedder_down": 1,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_avgpool":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 2048,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 6,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_fc":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 10,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
    "cinn_resnet_animalfaces10_ae_softmax":
        {"Transformer": {
            "hidden_depth": 2,
            "in_channels": 128,
            "mid_channels": 1024,
            "n_flows": 20,
            "conditioning_in_channels": 10,
            "conditioning_spatial_size": 1,
            "conditioning_depth": 4,
            "activation": "none",
            "conditioning_option": "none"
            }
        },
}


def download(url, local_path, chunk_size = 1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream = True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total = total_size, unit = "B", unit_scale = True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size = chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)


def md5_hash(path):
    with open(path, "rb") as f:
        content = f.read()
    return hashlib.md5(content).hexdigest()


def get_ckpt_path(name, root=None, check=False):
    assert name in URL_MAP
    cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    root = root if root is not None else os.path.join(cachedir, "invariances")
    path = os.path.join(root, CKPT_MAP[name])
    if not os.path.exists(path) or (check and not md5_hash(path)==MD5_MAP[name]):
        print("Downloading {} model from {} to {}".format(name, URL_MAP[name], path))
        download(URL_MAP[name], path)
        md5 = md5_hash(path)
        assert md5==MD5_MAP[name], md5
    return path
