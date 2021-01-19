import glob
import os
from itertools import chain, combinations, product

import numpy as np
import pandas as pd

from params import coeff_list, base_list, jpeg_list, dataset_ext, dataset_label


def get_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1)))


def get_params_range():
    coeff_range = np.asarray([np.asarray(range(x)) for x in np.asarray(range(len(coeff_list))) + 1])
    base_range = np.asarray(list(get_subsets(base_list)))
    comp_range = [tuple(jpeg_list[:i + 1]) for i in range(len(jpeg_list))]

    params_range = list(product(comp_range, base_range, coeff_range))
    return params_range


def load_features(comp, base, coeff, features_div_dir):
    ff_list = []
    y_list = []
    y_logo_list = []
    # Loading Features
    for dataset_name, _ in dataset_ext.items():
        ff_same_param = []
        y_same_param = []
        y_logo_same_param = []
        y_orig_flag = True
        y_gan_flag = True
        for j in comp:
            for b in base:
                for c in coeff:
                    feature_div_path = glob.glob(os.path.join(features_div_dir,
                                                              'jpeg_{}/b{}/c{}/{}.pkl'.format(j, b, c,
                                                                                              dataset_name)))[0]
                    dataset_logo_label = dataset_label[dataset_name]

                    ff = pd.read_pickle(feature_div_path)

                    ff_same_param += [np.concatenate([ff['kl'][:, None],
                                                      ff['reny'][:, None],
                                                      ff['tsallis'][:, None]],
                                                     axis=-1)]

                    if '_orig' in dataset_name and y_orig_flag:
                        y_same_param += [0] * len(ff)
                        y_logo_same_param += [dataset_logo_label] * len(ff)
                        y_orig_flag = False
                    elif '_gan' in dataset_name and y_gan_flag:
                        y_same_param += [1] * len(ff)
                        y_logo_same_param += [dataset_logo_label] * len(ff)
                        y_gan_flag = False

        ff_list += [np.concatenate(ff_same_param, axis=1)]
        y_list += y_same_param
        y_logo_list += y_logo_same_param

    ff_list = np.concatenate(ff_list, axis=0)
    y_list = np.array(y_list)
    y_logo_list = np.array(y_logo_list)

    return ff_list, y_list, y_logo_list
