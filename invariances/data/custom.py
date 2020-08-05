import os
import numpy as np
from edflow.util import retrieve
from autoencoders.data import ImagePaths, Folder


class LabelFolder(Folder):
    def __init__(self, config):
        super().__init__(config=config)
        folder = retrieve(config, "Folder/folder")
        label_level = retrieve(config, "Folder/label_level")
        size = retrieve(config, "Folder/size", default=0)
        random_crop = retrieve(config, "Folder/random_crop", default=False)

        all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(folder) for name in files]
        level_names = [f.split(os.sep)[label_level] for f in sorted(all_files)]
        unique_levels, class_labels = np.unique(np.array(level_names), return_inverse=True)
        labels = {"class_label": class_labels,
                  "class_name": level_names}

        self.data = ImagePaths(paths=all_files,
                               labels=labels,
                               size=size,
                               random_crop=random_crop)
