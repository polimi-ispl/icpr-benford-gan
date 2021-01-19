import argparse
import glob
import os
import warnings
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from functions import all_subsets
from params import dataset_ext, results_root, features_div_root, dataset_label, coeff_list, base_list, jpeg_list

warnings.simplefilter('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_compression', help='Apply random compression to training images', action='store_true',
                        default=False)
    parser.add_argument('--test_compression', help='Apply random compression to testing images', action='store_true',
                        default=False)
    parser.add_argument('--param_idx', required=False, type=int, help='List of specific index for param list',
                        nargs='*')
    parser.add_argument('--recompression_qf', type=int)

    args = parser.parse_args()
    train_compression = args.train_compression
    test_compression = args.test_compression
    param_idx = args.param_idx
    recompression_qf = args.recompression_qf

    np.random.seed(21)

    task_name_no_suffix = 'rf_combinations_logo'
    print('TASK: {}'.format(task_name_no_suffix))

    recompression_qf_suf = '_{}'.format(recompression_qf)
    suffix = '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                    test_compression)
    suffix_neg = '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                        not test_compression)
    task_name = task_name_no_suffix + suffix
    task_name_neg = task_name_no_suffix + suffix_neg

    feature_compact_dir = features_div_root + '_recompression{}'.format(
        recompression_qf_suf) if test_compression else features_div_root

    os.makedirs(os.path.join(results_root, task_name), exist_ok=True)

    feature_type = ''
    if feature_type != '':
        feature_type = '_' + feature_type

    coeff_range = np.asarray([np.asarray(range(x)) for x in np.asarray(range(len(coeff_list))) + 1])
    base_range = np.asarray(list(all_subsets(base_list)))
    comp_range = [tuple(jpeg_list[:i + 1]) for i in range(len(jpeg_list))]

    params_range = list(product(comp_range, base_range, coeff_range))

    if param_idx is not None:
        params_range = [params_range[x] for x in param_idx]

    for comp, base, coeff in tqdm(params_range):

        name = 'ff{}_comp_{}_base_{}_coeff_{}.npy'.format(feature_type, comp, base, coeff)

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
                        feature_compact_path = glob.glob(os.path.join(feature_compact_dir,
                                                                      'jpeg_{}/b{}/c{}/{}.pkl'.format(j, b, c,
                                                                                                      dataset_name)))[0]
                        dataset_logo_label = dataset_label[dataset_name]

                        ff = pd.read_pickle(feature_compact_path)

                        ff_same_param += [np.concatenate([ff['kl{}'.format(feature_type)][:, None],
                                                          ff['reny{}'.format(feature_type)][:, None],
                                                          ff['tsallis{}'.format(feature_type)][:, None]],
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

        # Subsampling
        sub_idx = np.random.choice(np.arange(len(ff_list)), int(len(ff_list) // 3))

        X = ff_list[sub_idx]
        y = y_list[sub_idx]
        y_logo = y_logo_list[sub_idx]

        # Remove inf values
        non_inf_idx = ~np.isinf(X).any(axis=1)
        X = X[non_inf_idx]
        y = y[non_inf_idx]
        y_logo = y_logo[non_inf_idx]

        # Replace nan values
        nan_idx = np.isnan(X)
        X[nan_idx] = -999

        # Load the trained models
        estimator_arr = np.load(os.path.join(results_root,
                                             task_name_neg,
                                             name)).item()['estimator']

        acc_list = []
        for leave_out_label in range(len(estimator_arr)):
            rf = estimator_arr[leave_out_label]
            y_hat = rf.predict(X[y_logo == leave_out_label])
            acc_list += [balanced_accuracy_score(y[y_logo == leave_out_label], y_hat)]

        acc = np.array(acc_list)

        result_data = {
            'acc': acc
        }

        np.save(os.path.join(results_root, task_name, name), result_data)

        del X, y, y_logo, rf

    return 0


if __name__ == '__main__':
    main()
