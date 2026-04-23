import sys
sys.path.append('../src')

import numpy as np
import pytest
from ab_test import ABTest
from stat_tests import (z_test, t_test, confidence_interval,
                        effect_size_cohens_d, required_sample_size)


@pytest.fixture
def significant_data():
    np.random.seed(42)
    control = np.random.normal(0.10, 0.02, 1000)
    treatment = np.random.normal(0.15, 0.02, 1000)
    return control, treatment


@pytest.fixture
def nonsignificant_data():
    np.random.seed(42)
    control = np.random.normal(0.10, 0.02, 1000)
    treatment = np.random.normal(0.101, 0.02, 1000)
    return control, treatment


def test_ab_test_significant(significant_data):
    c, t = significant_data
    test = ABTest("test", alpha=0.05)
    test.add_control(c)
    test.add_treatment(t)
    result = test.analyze()
    assert result['significant'] == True
    assert result['winner'] == 'treatment'


def test_ab_test_nonsignificant(nonsignificant_data):
    np.random.seed(42)
    c = np.random.normal(0.10, 0.02, 100)
    t = np.random.normal(0.101, 0.02, 100)
    test = ABTest("test", alpha=0.05)
    test.add_control(c)
    test.add_treatment(t)
    result = test.analyze()
    assert result['significant'] == False


def test_lift_calculation(significant_data):
    c, t = significant_data
    test = ABTest("test")
    test.add_control(c)
    test.add_treatment(t)
    result = test.analyze()
    assert result['lift'] > 0


def test_control_stats(significant_data):
    c, t = significant_data
    test = ABTest("test")
    test.add_control(c)
    test.add_treatment(t)
    result = test.analyze()
    assert result['control']['n'] == 1000
    assert abs(result['control']['mean'] - 0.10) < 0.01


def test_z_test():
    np.random.seed(42)
    c = np.random.normal(0.10, 0.02, 1000)
    t = np.random.normal(0.15, 0.02, 1000)
    result = z_test(c, t)
    assert result['p_value'] < 0.05
    assert 'z_stat' in result


def test_confidence_interval():
    np.random.seed(42)
    data = np.random.normal(0.5, 0.1, 1000)
    lo, hi = confidence_interval(data)
    assert lo < 0.5 < hi


def test_effect_size():
    np.random.seed(42)
    c = np.random.normal(1.0, 0.5, 100)
    t = np.random.normal(2.0, 0.5, 100)
    d = effect_size_cohens_d(c, t)
    assert d > 0


def test_sample_size():
    n = required_sample_size(0.10, 0.02)
    assert n > 0
    assert isinstance(n, int)


def test_no_data_raises():
    test = ABTest("test")
    with pytest.raises(ValueError):
        test.analyze()


def test_add_single_value():
    test = ABTest("test")
    test.add_control(0.5)
    test.add_treatment(0.6)
    assert len(test.control) == 1
    assert len(test.treatment) == 1