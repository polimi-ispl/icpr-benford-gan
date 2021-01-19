import argparse
import glob
import os
import warnings
from itertools import product
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
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
    parser.add_argument('--save_estimator', help='Save the estimator', action='store_true',
                        default=False)
    parser.add_argument('--param_idx', required=False, type=int, help='List of specific index for param list',
                        nargs='*')
    parser.add_argument('--recompression_qf', type=int)
    parser.add_argument('--subsampling', type=float, default=0.3)

    args = parser.parse_args()
    train_compression = args.train_compression
    # test_compression = args.test_compression
    save_estimator = args.save_estimator
    param_idx = args.param_idx
    recompression_qf = args.recompression_qf
    subsampling = args.subsampling

    sub_coeff = 0.1
    sub_suff = sub_coeff * subsampling

    np.random.seed(21)

    task_name = __file__.split('/')[-1].split('.')[0]
    print('TASK: {}'.format(task_name))

    recompression_qf_suf = '_{}'.format(recompression_qf)

    # if train_compression or test_compression:
    test_compression = False  # For compressed test you should run rf_combinations_logo_test_only.py
    task_name += '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                      test_compression)

    feature_compact_dir = features_div_root + '_recompression{}'.format(
        recompression_qf_suf) if train_compression else features_div_root

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

        name = 'ff{}_comp_{}_base_{}_coeff_{}_subsample_{}.npy'.format(feature_type, comp, base, coeff, sub_suff)
        if os.path.exists(os.path.join(results_root, task_name, name)):
            print('{} already exists, skipping...'.format(name))
            # continue

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
        sub_idx = np.random.choice(np.arange(len(ff_list)), int(np.round(len(ff_list) * sub_coeff)))

        X = ff_list[sub_idx]
        y = y_list[sub_idx]
        y_logo = y_logo_list[sub_idx]

        if subsampling:
            sub_idx = np.random.choice(np.arange(len(ff_list)), int(np.round(len(ff_list) * sub_coeff)))

        # Remove inf values
        non_inf_idx = ~np.isinf(X).any(axis=1)
        X = X[non_inf_idx]
        y = y[non_inf_idx]
        y_logo = y_logo[non_inf_idx]

        # Replace nan values
        nan_idx = np.isnan(X)
        X[nan_idx] = -999

        # Shuffling training set
        shuffle_idx = np.arange(len(y))
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        y_logo = y_logo[shuffle_idx]

        print('\n\n\nTotal {} samples, Leave-One-Group-Out cv. Feature size: {}\n\n\n'.format(X.shape[0],
                                                                                  X.shape[1]))

        # Create a based model
        rf = RandomForestClassifier(n_jobs=cpu_count() - 2, bootstrap=True, n_estimators=100, criterion='gini')

        # LOGO cv policy
        logo = LeaveOneGroupOut()

        cv = cross_validate(estimator=rf, X=X, y=y, groups=y_logo,
                            scoring='balanced_accuracy', cv=logo, verbose=2, return_estimator=save_estimator)

        if save_estimator:
            result_data = {
                'acc': cv['test_score'],
                'estimator': cv['estimator'],
                'X': X,
                'y': y,
                'y_logo': y_logo
            }
        else:
            result_data = {
                'acc': cv['test_score']
            }

        print('Saving results')
        np.save(os.path.join(results_root, task_name, name), result_data)

        del X, y, y_logo, rf, cv

    return 0


if __name__ == '__main__':
    main()
