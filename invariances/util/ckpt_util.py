import os, hashlib
import requests
from tqdm import tqdm


URL_MAP = {
    "cinn_alexnet_aae_conv5": "https://heibox.uni-heidelberg.de/f/62b0e29d8c544f51b79c/?dl=1",
    "cinn_alexnet_aae_fc6": "https://heibox.uni-heidelberg.de/f/5d07dc071dd1450eb0a5/?dl=1",
    "cinn_alexnet_aae_fc7": "https://heibox.uni-heidelberg.de/f/050d4d76f3cf4eeeb9b6/?dl=1",
    "cinn_alexnet_aae_fc8": "https://heibox.uni-heidelberg.de/f/cb9c93497aae4e97890c/?dl=1",
    "cinn_alexnet_aae_softmax": "https://heibox.uni-heidelberg.de/f/5a30088c51b44cc58bbe/?dl=1"
}

CKPT_MAP = {
    "cinn_alexnet_aae_conv5": "invariances/pretrained_models/cinns/alexnet/conv5.ckpt",
    "cinn_alexnet_aae_fc6": "invariances/pretrained_models/cinns/alexnet/fc6.ckpt",
    "cinn_alexnet_aae_fc7": "invariances/pretrained_models/cinns/alexnet/fc7.ckpt",
    "cinn_alexnet_aae_fc8": "invariances/pretrained_models/cinns/alexnet/fc8.ckpt",
    "cinn_alexnet_aae_softmax": "invariances/pretrained_models/cinns/alexnet/softmax.ckpt"
}

MD5_MAP = {
    "cinn_alexnet_aae_conv5": "ae9367e3cf7486218375c2b328d1273a",
    "cinn_alexnet_aae_fc6": "e7128567ee0686d362ed83cfda9fabc1",
    "cinn_alexnet_aae_fc7": "342ab8f9280ed83f30150e586c7df13b",
    "cinn_alexnet_aae_fc8": "ce6b5bfc316a855693b7e1808d6f3a46",
    "cinn_alexnet_aae_softmax": "15fbdd0d8d51ba0031fbc07ebf0ba2f6"
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
