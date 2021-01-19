import argparse
import os
import warnings

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from functions import get_params_range, load_features
from params import results_root, features_div_root

warnings.simplefilter('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_compression', help='Apply random compression to training images', action='store_true',
                        default=False)
    parser.add_argument('--test_compression', help='Apply random compression to testing images', action='store_true',
                        default=False)
    parser.add_argument('--param_idx', required=False, type=int, help='List of specific index for param list',
                        nargs='*')
    parser.add_argument('--subsampling', type=float, default=0.3)
    parser.add_argument('--recompression_qf', type=int)

    args = parser.parse_args()
    train_compression = args.train_compression
    test_compression = args.test_compression
    param_idx = args.param_idx
    recompression_qf = args.recompression_qf
    subsampling = args.subsampling

    np.random.seed(21)

    task_name_no_suffix = 'rf_combinations_logo'
    print('TASK: {}'.format(task_name_no_suffix))

    recompression_qf_suf = '_{}'.format(recompression_qf) if recompression_qf is not None else ''
    suffix = '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                  test_compression)
    suffix_neg = '_train-compression_{}{}_test-compression_{}'.format(train_compression, recompression_qf_suf,
                                                                      not test_compression)
    task_name = task_name_no_suffix + suffix
    task_name_neg = task_name_no_suffix + suffix_neg

    features_div_dir = features_div_root + '_recompression{}'.format(
        recompression_qf_suf) if test_compression else features_div_root

    os.makedirs(os.path.join(results_root, task_name), exist_ok=True)

    all_params_range = get_params_range()

    if param_idx is not None:
        params_range = [all_params_range[x] for x in param_idx]
    else:
        params_range = all_params_range

    for comp, base, coeff in tqdm(params_range):

        name = 'ff_{}_base_{}_coeff_{}.npy'.format(comp, base, coeff)

        ff_list, y_list, y_logo_list = load_features(comp, base, coeff, features_div_dir)

        # Subsampling
        sub_idx = np.random.choice(np.arange(len(ff_list)), int(len(ff_list) // subsampling))
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
