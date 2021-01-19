import glob
import os

import pandas as pd
from tqdm import tqdm

from params import dataset_root, dataset_ext, dataset_label, data_root


def main():
    for leave_out_label in tqdm(set(dataset_label.values())):
        path_list_train = []
        y_list_train = []
        y_logo_list_train = []
        path_list_test = []
        y_list_test = []
        y_logo_list_test = []
        for dataset_name, logo_label in dataset_label.items():
            if logo_label != leave_out_label:
                paths = glob.glob(os.path.join(dataset_root[dataset_name],
                                               '*.{}'.format(dataset_ext[dataset_name])))
                path_list_train += paths
                y_logo_list_train += [logo_label] * len(paths)
                if '_orig' in dataset_name:
                    y_list_train += [0] * len(paths)
                elif '_gan' in dataset_name:
                    y_list_train += [1] * len(paths)
            else:
                paths = glob.glob(os.path.join(dataset_root[dataset_name],
                                               '*.{}'.format(dataset_ext[dataset_name])))
                path_list_test += paths
                y_logo_list_test += [logo_label] * len(paths)
                if '_orig' in dataset_name:
                    y_list_test += [0] * len(paths)
                elif '_gan' in dataset_name:
                    y_list_test += [1] * len(paths)

        df_train = pd.DataFrame(columns=['path', 'label', 'logo_label'],
                                data=zip(path_list_train, y_list_train, y_logo_list_train))
        df_test = pd.DataFrame(columns=['path', 'label', 'logo_label'],
                               data=zip(path_list_test, y_list_test, y_logo_list_test))
        df_train.to_csv(os.path.join(data_root, 'logo_{}_split_train.csv'.format(leave_out_label)), index=None)
        df_test.to_csv(os.path.join(data_root, 'logo_{}_split_test.csv'.format(leave_out_label)), index=None)

    return 0


if __name__ == '__main__':
    main()
