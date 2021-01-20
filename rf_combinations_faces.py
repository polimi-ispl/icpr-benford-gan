import argparse
import glob
import os
import warnings
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from functions import get_params_range
from params import results_root, features_div_root, gan_orig_map_faces, default_param_idx

warnings.simplefilter('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='GAN Dataset name to consider')
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
    gan_dataset_name = args.dataset
    train_compression = args.train_compression
    save_estimator = args.save_estimator
    param_idx = args.param_idx
    recompression_qf = args.recompression_qf
    subsampling = args.subsampling

    if gan_dataset_name:
        if gan_dataset_name not in gan_orig_map_faces:
            print('Dataset must be one of the following {}'.format(gan_orig_map_faces.keys()))
            return 1
        else:
            gan_dataset_list = [gan_dataset_name]
    if gan_dataset_name is None:
        gan_dataset_list = list(gan_orig_map_faces.keys())

    np.random.seed(21)

    task_name = __file__.split('/')[-1].split('.')[0]
    print('TASK: {}'.format(task_name))

    recompression_qf_suf = '_{}'.format(recompression_qf) if recompression_qf else ''

    test_compression = False  # For compressed test you should run rf_combinations_logo_test_only.py
    task_name += '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                      test_compression)

    # Append suffix
    feature_div_dir = features_div_root + '_recompression{}'.format(
        recompression_qf_suf) if train_compression else features_div_root

    os.makedirs(os.path.join(results_root, task_name), exist_ok=True)

    params_range = get_params_range()

    if param_idx is None:
        params_range = [params_range[x] for x in default_param_idx]
    else:
        params_range = [params_range[x] for x in param_idx]

    for gan_dataset in gan_dataset_list:
        os.makedirs(os.path.join(results_root, task_name, '_'.join(gan_dataset.split('_')[:-1])), exist_ok=True)
        for comp, base, coeff in tqdm(params_range):

            name = 'ff_comp_{}_base_{}_coeff_{}_subsample_{}.npy'.format(comp, base, coeff, subsampling)

            ff_list = []
            y_list = []

            # Loading Features
            dataset_tmp_list = [gan_dataset, gan_orig_map_faces[gan_dataset]]
            for dataset in dataset_tmp_list:
                ff_same_param = []
                y_same_param = []
                y_orig_flag = True
                y_gan_flag = True
                for j in comp:
                    for b in base:
                        for c in coeff:
                            feature_div_path = glob.glob(os.path.join(feature_div_dir,
                                                                      'jpeg_{}/b{}/c{}/{}.pkl'.format(j, b, c,
                                                                                                      dataset)))[0]

                            ff = pd.read_pickle(feature_div_path)

                            ff_same_param += [np.concatenate([ff['kl'][:, None],
                                                              ff['reny'][:, None],
                                                              ff['tsallis'][:, None]],
                                                             axis=-1)]

                            if '_orig' in dataset and y_orig_flag:
                                y_same_param += [0] * len(ff)
                                y_orig_flag = False
                            elif '_gan' in dataset and y_gan_flag:
                                y_same_param += [1] * len(ff)
                                y_gan_flag = False

                ff_list += [np.concatenate(ff_same_param, axis=1)]
                y_list += y_same_param

            ff_list = np.concatenate(ff_list, axis=0)
            y_list = np.array(y_list)

            # Subsampling
            sub_idx = np.random.choice(np.arange(len(ff_list)), int(np.round(len(ff_list) * sub_coeff)))

            X = ff_list[sub_idx]
            y = y_list[sub_idx]

            # Remove inf values
            non_inf_idx = ~np.isinf(X).any(axis=1)
            X = X[non_inf_idx]
            y = y[non_inf_idx]

            # Replace nan values
            nan_idx = np.isnan(X)
            X[nan_idx] = -999

            # Create split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

            # Keep always the same test dimension
            if subsampling:
                sub_idx = np.random.choice(np.arange(len(X_train)), int(np.round(len(X_train) * subsampling)))
                X_train = X_train[sub_idx]
                y_train = y_train[sub_idx]

            print('\n\n\nTrain {} on {} samples, test on {}. Feature size: {}\n\n\n'.format(
                '_'.join(gan_dataset.split('_')[:-1]),
                X_train.shape[0],
                X_test.shape[0],
                X.shape[1]))

            # Create model
            rf = RandomForestClassifier(n_jobs=cpu_count(), bootstrap=True, n_estimators=100, criterion='gini')

            # Fit model
            rf.fit(X=X_train, y=y_train)

            # Predict
            y_pred = rf.predict(X_test)

            # Compute accuracy
            acc = balanced_accuracy_score(y_pred=y_pred, y_true=y_test)

            if save_estimator:
                result_data = {
                    'acc': acc,
                    'estimator': rf,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_test': X_test,
                    'y_test': y_test
                }
            else:
                result_data = {
                    'acc': acc
                }

            print('Saving results')
            np.save(os.path.join(results_root, task_name, '_'.join(gan_dataset.split('_')[:-1]), name), result_data)

            del X, y, rf

    return 0


if __name__ == '__main__':
    main()
