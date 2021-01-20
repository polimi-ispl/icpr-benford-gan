import argparse
import glob
import os
import warnings
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image
from tqdm import tqdm

from params import dataset_root, dataset_ext, cooccurrences_root

np.random.seed(21)

warnings.simplefilter('ignore')


def cooccurrences(args: dict):
    path = args['path']
    I = np.array(Image.open(path).convert('L')).astype(np.single)

    # Parameters
    Q = 1.
    T = 2

    # HPF
    R = I[:, 0:-3] - 3 * I[:, 1:-2] + 3 * I[:, 2:-1] - I[:, 3:]

    # Truncation and quantization
    R_q = np.round(R / Q).astype(np.int8)
    R_t = R_q
    R_t[R_t <= -T] = -T
    R_t[R_t >= T] = T

    # Compute histogram
    C = np.zeros((2 * T + 1, 2 * T + 1, 2 * T + 1, 2 * T + 1))
    H, W = R_t.shape
    for i in range(H):
        for j in range(W - 3):
            v1 = R_t[i, j] + T
            v2 = R_t[i, j + 1] + T
            v3 = R_t[i, j + 2] + T
            v4 = R_t[i, j + 3] + T
            C[v1, v2, v3, v4] += 1

    # Reshape and normalize feature vector
    feat = C.ravel() / sum(C.ravel())

    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name', type=str)
    parser.add_argument('--workers', help='Number of parallel workers', type=int, default=cpu_count() // 2)
    args = parser.parse_args()

    dataset_name = args.dataset
    workers = args.workers

    if dataset_name is None:
        dataset_list = [x for x in dataset_ext.keys()]
    else:
        dataset_list = [dataset_name]

    for dataset_name in tqdm(dataset_list):

        if dataset_name not in dataset_root:
            print('Dataset must be registered in dataset_root variable (params.py). {} not found'.format(dataset_name))
            return 1

        # Retrieve all the dataset filenames
        path_list = glob.glob(os.path.join(dataset_root[dataset_name], '*.{}'.format(dataset_ext[dataset_name])))

        feature_dir = cooccurrences_root
        dir_name = os.path.join(feature_dir)
        out_name = os.path.join(dir_name, '{}.npy'.format(dataset_name))

        if os.path.exists(out_name):
            print('Already computed, {}. Skipping...'.format(dataset_name, ))
            continue
        else:

            # prepare arguments
            args_list = list()
            for path in path_list:
                arg = dict()
                arg['path'] = os.path.join(dataset_root[dataset_name], path)
                args_list += [arg]

            print('\nComputing cooccurrences for {}'.format(dataset_name, ))
            with Pool(workers, maxtasksperchild=2) as p:
                ff = p.map(cooccurrences, args_list)

            ff = np.asarray(ff)

            # saving features
            print('Saving features')
            os.makedirs(dir_name, exist_ok=True)
            np.save(out_name, ff)

    return 0


if __name__ == '__main__':
    main()
