import numpy as np
from scipy import stats


def lower_confidence_limit(sample, confidence=0.95):
    n_sample = len(sample)
    print('n:', n_sample, sep='\t')
    t_value = stats.t.ppf(q=confidence, df=n_sample - 1)
    print('t:', t_value, sep='\t')
    sample_std = np.std(sample, ddof=1)
    print('std:', sample_std, sep='\t')
    sample_mean = np.mean(sample)
    print('mean:', sample_mean, sep='\t')
    print('confidence:', confidence, sep='\t')
    return sample_mean - t_value * (sample_std / np.sqrt(n_sample))
