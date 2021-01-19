import argparse
import os
import warnings
from itertools import product
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import entropy
from tqdm import tqdm

from params import fd_hist_root, coeff_list, base_list, compression_list, dataset_ext, results_root, \
    features_div_root

warnings.simplefilter('ignore')


def gen_benford(m, k, a, b):
    base = len(m)
    return k * (np.log10(1 + (1 / (a + m ** b))) / np.log10(base))


def renyi_div(pk, qk, alpha):
    r = np.log2(np.nansum((pk ** alpha) * (qk ** (1 - alpha)))) / (alpha - 1)
    return r


def tsallis_div(pk, qk, alpha):
    r = (np.nansum((pk ** alpha) * (qk ** (1 - alpha))) - 1) / (alpha - 1)
    return r


def feature_extraction(ff: np.ndarray):
    base = len(ff) + 1

    mse_img = []
    popt_img = []
    kl_img = []
    renyi_img = []
    tsallis_img = []

    ff_zeroes_idx = ff == 0
    try:
        # Compute regular features
        popt_k, _ = curve_fit(gen_benford, np.arange(1, base, 1), ff)
        h_fit = gen_benford(np.arange(1, base, 1), *popt_k)

        h_fit_zeroes_idx = h_fit == 0

        zeroes_idx = np.logical_or(ff_zeroes_idx, h_fit_zeroes_idx)

        ff_no_zeroes = ff[~zeroes_idx]
        h_fit_no_zeroes = h_fit[~zeroes_idx]

        popt_img += [popt_k]
        mse_img += [np.mean((ff - h_fit) ** 2)]

        kl_img += [entropy(pk=ff_no_zeroes, qk=h_fit_no_zeroes, base=2) +
                   entropy(pk=h_fit_no_zeroes, qk=ff_no_zeroes, base=2)]
        renyi_img += [renyi_div(pk=ff_no_zeroes, qk=h_fit_no_zeroes, alpha=0.3) +
                      renyi_div(pk=h_fit_no_zeroes, qk=ff_no_zeroes, alpha=0.3)]
        tsallis_img += [tsallis_div(pk=ff_no_zeroes, qk=h_fit_no_zeroes, alpha=0.3) +
                        tsallis_div(pk=h_fit_no_zeroes, qk=ff_no_zeroes, alpha=0.3)]

    except (RuntimeError, ValueError):
        mse_img += [np.nan]
        popt_img += [(np.nan, np.nan, np.nan)]
        kl_img += [np.nan]
        renyi_img += [np.nan]
        tsallis_img += [np.nan]

    mse = np.asarray(mse_img)
    popt = np.asarray(popt_img)
    kl = np.asarray(kl_img)
    renyi = np.asarray(renyi_img)
    tsallis = np.asarray(tsallis_img)

    return mse, popt[:, 0], popt[:, 1], popt[:, 2], kl, renyi, tsallis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jpeg_recompression', action='store_true', default=False)
    parser.add_argument('--recompression_qf', type=int)
    parser.add_argument('--workers', help='Number of parallel workers', type=int, default=cpu_count() // 2)

    args = parser.parse_args()
    jpeg_recompression = args.jpeg_recompression
    recompression_qf = args.recompression_qf
    workers = args.workers

    recompression_qf_suf = '_{}'.format(recompression_qf)
    np.random.seed(21)

    task_name = __file__.split('/')[-1].split('.')[0]
    print('TASK: {}'.format(task_name))

    params_range = list(product(coeff_list, base_list, compression_list))
    p = Pool(workers)

    feature_div_dir = features_div_root + '_recompression{}'.format(
        recompression_qf_suf) if jpeg_recompression else features_div_root

    feature_dir = fd_hist_root + '_recompression{}'.format(
        recompression_qf_suf) if jpeg_recompression else fd_hist_root

    for coeff, base, compression in params_range:
        for dataset_name, _ in tqdm(dataset_ext.items(), desc='feature_{}_{}_{}'.format(compression, base, coeff)):
            feature_div_path = os.path.join(feature_div_dir, compression, 'b{}'.format(base),
                                            'c{}'.format(coeff), '{}.pkl'.format(dataset_name))
            os.makedirs(os.path.dirname(feature_div_path), mode=0o755, exist_ok=True)

            if os.path.isfile(feature_div_path):
                print('{} Already exist, skipping..'.format(feature_div_path))
                continue

            # Loading histograms
            try:
                hist = np.load(os.path.join(feature_dir, '{}/b{}/{}.npy'.format(compression, base, dataset_name)))
            except(FileNotFoundError):
                print(f'You must first compute first digits '
                      f'for {dataset_name}, compression: {compression}, base: {base}. Skipping...')
                continue
            # Computing features
            ff = p.map(feature_extraction, hist[:, coeff])

            ff = np.squeeze(np.array(ff))

            ff_df = pd.DataFrame(data=ff, columns=['mse', 'popt_0', 'popt_1', 'popt_2', 'kl', 'reny', 'tsallis'])

            ff_df.to_pickle(feature_div_path)

    return 0


if __name__ == '__main__':
    main()
