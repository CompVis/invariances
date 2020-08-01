import os
from edflow.util import retrieve
from autoencoders.data import ImagePaths


class Folder(ImagePaths):
    # TODO: delete, because we use the one from 'autoencoders'
    def __init__(self, config):
        folder = retrieve(config, "Folder/folder")
        size = retrieve(config, "Folder/size", default=0)
        random_crop = retrieve(config, "Folder/random_crop", default=False)

        relpaths = sorted(os.listdir(folder))
        abspaths = [os.path.join(folder, p) for p in relpaths]
        labels = {"relpaths": relpaths}

        self.data = ImagePaths(paths=abspaths,
                               labels=labels,
                               size=size,
                               random_crop=random_crop)