import numpy as np
from tqdm import tqdm
from edflow.iterators.batches import DatasetMixin
from edflow.data.dataset import PRNGMixin
from edflow.util import retrieve
from edflow import get_logger


class TrainSamples(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.config = config["BigGANData"]
        self.logger = get_logger(self.__class__.__name__)
        self.n_samples = self.config["n_train_samples"]
        self.z_shape = self.config["z_shape"]
        self.n_classes = self.config["n_classes"]
        self.truncation_threshold = retrieve(self.config, "truncation", default=-1)
        if self.truncation_threshold > -1:
            self.logger.info("Applying truncation at level {}".format(self.truncation_threshold))

    def __len__(self):
        return self.n_samples

    def get_example(self, i):
        z = self.prng.randn(*self.z_shape)
        if self.truncation_threshold > -1:
            for k, zi in enumerate(z):
                while abs(zi) > self.truncation_threshold:
                    zi = self.prng.randn(1)
                z[k] = zi
        cls = self.prng.randint(self.n_classes)
        return {"z": z, "class": cls}


class TestSamples(DatasetMixin):
    def __init__(self, config):
        self.prng = np.random.RandomState(1)
        self.config = config["BigGANData"]
        self.logger = get_logger(self.__class__.__name__)
        self.n_samples = self.config["n_test_samples"]
        self.z_shape = self.config["z_shape"]
        self.n_classes = self.config["n_classes"]
        self.truncation_threshold = retrieve(self.config, "truncation", default=-1)
        self.zs = self.prng.randn(self.n_samples, *self.z_shape)
        if self.truncation_threshold > -1:
            self.logger.info("Applying truncation at level {}".format(self.truncation_threshold))
            ix = 0
            for z in tqdm(self.zs, desc="Truncation:"):
                for k, zi in enumerate(z):
                    while abs(zi) > self.truncation_threshold:
                        zi = self.prng.randn(1)
                    z[k] = zi
                self.zs[ix] = z
                ix += 1
            self.logger.info("Created truncated test data.")
        self.clss = self.prng.randint(self.n_classes, size=(self.n_samples,))

    def __len__(self):
        return self.n_samples

    def get_example(self, i):
        return {"z": self.zs[i], "class": self.clss[i]}
