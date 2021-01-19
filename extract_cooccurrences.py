import argparse
import glob
import io
import os
import uuid
import warnings
from multiprocessing import Pool, cpu_count
import cv2

import jpeg
import numpy as np
from PIL import Image
from tqdm import tqdm

from params import base_list, dataset_root, dataset_ext, feature_hist_root, tmp_path, cooccurrences_root

np.random.seed(21)

warnings.simplefilter('ignore')  # we actually want nan as first digit when we meet a 0


def cooccurrences(args: dict):

    path = args['path']
    I = np.array(Image.open(path).convert('L'))

    # Compute image residuals
    # Parameters
    Q = 1.
    T = 2
    I = I.astype(np.single)

    ### Filtering (hardcoded but faster)
    R = I[:, 0:-3] - 3 * I[:, 1:-2] + 3 * I[:, 2:-1] - I[:, 3:]

    ### Truncation and quantization
    R_q = np.round(R / Q).astype(np.int8)
    R_t = R_q
    R_t[R_t <= -T] = -T
    R_t[R_t >= T] = T

    ## Compute feature vector
    ### Loop to compute histogram
    C = np.zeros((2 * T + 1, 2 * T + 1, 2 * T + 1, 2 * T + 1))
    H, W = R_t.shape
    for i in range(H):
        for j in range(W - 3):
            v1 = R_t[i, j] + T
            v2 = R_t[i, j + 1] + T
            v3 = R_t[i, j + 2] + T
            v4 = R_t[i, j + 3] + T
            C[v1, v2, v3, v4] += 1

    ### Reshape and normalize feature vector
    feat = C.ravel() / sum(C.ravel())

    return feat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name', type=str)
    # parser.add_argument('--jpeg', help='jpeg compression QF', required=True, type=int)
    # parser.add_argument('--jpeg_recompression', action='store_true', default=False)
    # parser.add_argument('--recompression_qf', type=int)
    args = parser.parse_args()

    dataset_name = args.dataset
    # jpeg_qf = args.jpeg
    # jpeg_recompression = args.jpeg_recompression
    # recompression_qf = args.recompression_qf

    # recompression_qf_suf = '_{}'.format(recompression_qf)

    # create temporary folder
    # os.makedirs(tmp_path, exist_ok=True)

    dataset_list = []
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

        # for base in base_list:

        # check if already computed
        # compression = 'jpeg_{}'.format(jpeg_qf)
        feature_dir = cooccurrences_root  # + '_recompression{}'.format(recompression_qf_suf) if jpeg_recompression else feature_root
        dir_name = os.path.join(feature_dir)  # , compression)
        out_name = os.path.join(dir_name, '{}.npy'.format(dataset_name))

        if os.path.exists(out_name):
            print('Already computed, {}. Skipping...'.format(dataset_name, ))
            # base,
            # jpeg_qf,
            # jpeg_recompression))
            continue
        else:

            # prepare arguments
            args_list = list()
            for path in path_list:
                arg = dict()
                arg['path'] = os.path.join(dataset_root[dataset_name], path)
                # arg['base'] = base
                # arg['jpeg_qf'] = jpeg_qf
                # arg['jpeg_recompression'] = jpeg_recompression
                # arg['recompression_qf'] = recompression_qf
                args_list += [arg]

            # compute first digits
            p = Pool(cpu_count(), maxtasksperchild=2)

            print('\nComputing cooccurrences for {}'.format(dataset_name,))
                                                           # base,jpeg_qf, len(args_list)))

            ff = p.map(cooccurrences, args_list)
            ff = np.asarray(ff)

            # saving features
            print('Saving features')
            os.makedirs(dir_name, exist_ok=True)
            np.save(out_name, ff)

            # print('Cleaning unused variables')
            # del ff
            # tmp_file_list = glob.glob(os.path.join(tmp_path, '*.jpg'))
            # [os.remove(x) for x in tmp_file_list]

    return 0


if __name__ == '__main__':
    main()
