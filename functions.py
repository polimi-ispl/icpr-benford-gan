from itertools import chain, combinations, product
import numpy as np
from params import coeff_list, base_list, jpeg_list


def get_subsets(ss):
    return chain(*map(lambda x: combinations(ss, x), range(1, len(ss) + 1)))


def get_params_range():
    coeff_range = np.asarray([np.asarray(range(x)) for x in np.asarray(range(len(coeff_list))) + 1])
    base_range = np.asarray(list(get_subsets(base_list)))
    comp_range = [tuple(jpeg_list[:i + 1]) for i in range(len(jpeg_list))]

    params_range = list(product(comp_range, base_range, coeff_range))
    return params_range
