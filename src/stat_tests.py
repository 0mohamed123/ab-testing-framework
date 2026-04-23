import numpy as np
from scipy import stats


def z_test(control, treatment):
    n_c, n_t = len(control), len(treatment)
    mean_c, mean_t = np.mean(control), np.mean(treatment)
    var_c, var_t = np.var(control, ddof=1), np.var(treatment, ddof=1)

    se = np.sqrt(var_c/n_c + var_t/n_t)
    z_stat = (mean_t - mean_c) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {'z_stat': z_stat, 'p_value': p_value, 'se': se}


def t_test(control, treatment):
    t_stat, p_value = stats.ttest_ind(control, treatment)
    return {'t_stat': t_stat, 'p_value': p_value}


def chi_square_test(control_conversions, control_total,
                    treatment_conversions, treatment_total):
    table = np.array([
        [control_conversions, control_total - control_conversions],
        [treatment_conversions, treatment_total - treatment_conversions]
    ])
    chi2, p_value, dof, _ = stats.chi2_contingency(table)
    return {'chi2': chi2, 'p_value': p_value, 'dof': dof}


def confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - margin, mean + margin


def effect_size_cohens_d(control, treatment):
    mean_diff = np.mean(treatment) - np.mean(control)
    pooled_std = np.sqrt((np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2)
    return mean_diff / pooled_std if pooled_std > 0 else 0


def required_sample_size(baseline_rate, min_detectable_effect,
                          alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    p1 = baseline_rate
    p2 = baseline_rate + min_detectable_effect
    p_avg = (p1 + p2) / 2
    n = (z_alpha * np.sqrt(2 * p_avg * (1 - p_avg)) +
         z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2)))**2 / (p2-p1)**2
    return int(np.ceil(n))