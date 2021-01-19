import io
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

if '2.' in tf.__version__:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    from tensorflow.keras.applications.xception import Xception
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.models import Model
    from tensorflow.keras.utils import Sequence, to_categorical
    from tensorflow.keras.backend import concatenate

else:
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    from keras.applications.xception import Xception
    from keras.layers import Dense, GlobalAveragePooling2D
    from keras.models import Model
    from keras.utils import Sequence
    from keras.utils.np_utils import to_categorical


def custom_xception(include_top: bool, weights: str, input_shape: tuple, n_classes: int):
    base_model = Xception(include_top=include_top, weights=weights, input_shape=input_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    return model


def crop_center(img: np.ndarray, cropx: int, cropy: int, oy=0, ox=0) -> np.ndarray:
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2) + oy
    starty = y // 2 - (cropy // 2) + ox
    return img[starty:starty + cropy, startx:startx + cropx]


class Generator(Sequence):

    def __init__(self, df: pd.DataFrame, compression: bool, batch_size: int, patch_size: tuple,
                 subsample: float = None, recompression_qf: int = None):
        self.compression = compression
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.subsample = subsample if subsample is not None else None
        self.recompression_qf = recompression_qf

        if subsample is not None:
            self.db = df.sample(frac=self.subsample)  # side effect: the df is shuffled
        else:
            self.db = df

    def __len__(self):
        return len(self.db) // self.batch_size

    def __getitem__(self, idx):
        batch_df = self.db[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []

        for _, row in batch_df.iterrows():
            img = np.asarray(Image.open(row.path))

            # pad or crop to self.patch_size dimension
            if img.shape[0] < self.patch_size[0]:
                img = np.pad(img, pad_width=[(0, self.patch_size[0] - img.shape[0]), (0, 0)], mode='reflect')
            if img.shape[1] < self.patch_size[1]:
                img = np.pad(img, pad_width=[(0, 0), (0, self.patch_size[1] - img.shape[1])], mode='reflect')
            if img.shape[0] > self.patch_size[0] or img.shape[1] > self.patch_size[1]:
                img = crop_center(img, cropx=self.patch_size[0], cropy=self.patch_size[1])

            # random JPEG compression
            if self.compression:
                buffer = io.BytesIO()
                np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
                qf = np.random.randint(low=85, high=101) if self.recompression_qf is None else self.recompression_qf
                Image.fromarray(img).convert('RGB').save(buffer, 'JPEG', quality=qf)
                img = np.asarray(Image.open(buffer))

            batch_x += [img]
            batch_y += [row.label]

        batch_x = np.asarray(batch_x)
        batch_y = to_categorical(np.asarray(batch_y), num_classes=2)

        return batch_x, batch_y
