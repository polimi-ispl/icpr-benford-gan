import argparse
import os
import warnings
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from tqdm import tqdm

from functions import get_params_range, load_features
from params import results_root, features_div_root

warnings.simplefilter('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_compression', help='Apply random compression to training images', action='store_true',
                        default=False)
    parser.add_argument('--save_estimator', help='Save the estimator', action='store_true',
                        default=False)
    parser.add_argument('--param_idx', required=False, type=int, help='List of specific index for param list',
                        nargs='*')
    parser.add_argument('--recompression_qf', type=int)
    parser.add_argument('--subsampling', type=float, default=0.03)
    parser.add_argument('--workers', help='Number of parallel workers', type=int, default=cpu_count() // 2)

    args = parser.parse_args()
    train_compression = args.train_compression
    save_estimator = args.save_estimator
    param_idx = args.param_idx
    recompression_qf = args.recompression_qf
    subsampling = args.subsampling
    workers = args.workers

    np.random.seed(21)

    task_name = __file__.split('/')[-1].split('.')[0]
    print('TASK: {}'.format(task_name))

    recompression_qf_suf = '_{}'.format(recompression_qf) if recompression_qf is not None else ''

    test_compression = False  # For testing compressed images you should run rf_combinations_logo_test_only.py
    task_name += '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                      test_compression)

    features_div_dir = features_div_root + '_recompression{}'.format(
        recompression_qf_suf) if train_compression else features_div_root

    os.makedirs(os.path.join(results_root, task_name), mode=0o775, exist_ok=True)

    all_params_range = get_params_range()

    if param_idx is not None:
        params_range = [all_params_range[x] for x in param_idx]
    else:
        params_range = all_params_range

    for comp, base, coeff in tqdm(params_range):

        name = 'ff_comp_{}_base_{}_coeff_{}_subsample_{}.npy'.format(comp, base, coeff, subsampling)
        if os.path.exists(os.path.join(results_root, task_name, name)):
            print('{} already exists, skipping...'.format(name))
            continue

        ff_list, y_list, y_logo_list = load_features(comp, base, coeff, features_div_dir)

        # Subsampling
        sub_idx = np.random.choice(np.arange(len(ff_list)), int(np.round(len(ff_list) * subsampling)))
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

        # Shuffling training set
        shuffle_idx = np.arange(len(y))
        np.random.shuffle(shuffle_idx)
        X = X[shuffle_idx]
        y = y[shuffle_idx]
        y_logo = y_logo[shuffle_idx]

        print('\n\n\nTotal {} samples, Leave-One-Group-Out cv. Feature size: {}\n\n\n'.format(X.shape[0],
                                                                                              X.shape[1]))

        # Create a based model
        rf = RandomForestClassifier(n_jobs=workers, bootstrap=True, n_estimators=100, criterion='gini')

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
