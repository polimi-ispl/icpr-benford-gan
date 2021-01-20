import argparse
import glob
import os
import warnings
from multiprocessing import cpu_count

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut, cross_validate

from params import dataset_ext, results_root, dataset_label, cooccurrences_root

warnings.simplefilter('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_estimator', help='Save the estimator', action='store_true',
                        default=False)

    args = parser.parse_args()
    save_estimator = args.save_estimator

    np.random.seed(21)

    task_name = __file__.split('/')[-1].split('.')[0]
    print('TASK: {}'.format(task_name))

    os.makedirs(os.path.join(results_root, task_name), exist_ok=True)

    name = 'rf_cooccurrences.npy'
    if os.path.exists(os.path.join(results_root, task_name, name)):
        print('{} already exists, skipping...'.format(task_name))
        return 1

    ff_list = []
    y_list = []
    y_logo_list = []
    # Loading Features
    for dataset_name, _ in dataset_ext.items():
        y_same_param = []
        y_logo_same_param = []
        feature_path = glob.glob(os.path.join(cooccurrences_root, '{}.npy'.format(dataset_name)))[0]
        dataset_logo_label = dataset_label[dataset_name]

        ff = np.load(feature_path)

        if '_orig' in dataset_name:
            y_same_param += [0] * len(ff)
            y_logo_same_param += [dataset_logo_label] * len(ff)
        elif '_gan' in dataset_name:
            y_same_param += [1] * len(ff)
            y_logo_same_param += [dataset_logo_label] * len(ff)

        ff_list += [ff]
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

    # Shuffling training set
    shuffle_idx = np.arange(len(y))
    np.random.shuffle(shuffle_idx)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    y_logo = y_logo[shuffle_idx]

    print('Total {} samples, Leave-One-Group-Out cv. Feature size: {}'.format(X.shape[0],
                                                                              X.shape[1]))

    # Create model
    model = RandomForestClassifier(n_jobs=cpu_count() - 2, bootstrap=True, n_estimators=100, criterion='gini')

    # LOGO cv policy
    logo = LeaveOneGroupOut()

    cv = cross_validate(estimator=model, X=X, y=y, groups=y_logo,
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

    np.save(os.path.join(results_root, task_name, name), result_data)

    del X, y, y_logo, model, cv

    return 0


if __name__ == '__main__':
    main()
