# A/B Testing Framework

![Language](https://img.shields.io/badge/Language-Python-blue)
![Tests](https://img.shields.io/badge/Tests-10%20passing-green)
![Stats](https://img.shields.io/badge/Stats-Z--test%20%7C%20T--test%20%7C%20Chi--Square-orange)

Statistical A/B testing framework with Z-test, T-test, Chi-Square,
confidence intervals, effect size, and sample size calculator.

## Demo Output

    A/B Test: CTR Experiment
    Control   (n=1000): mean=0.1004  std=0.0196
    Treatment (n=1000): mean=0.1214  std=0.0199

    Lift: +20.95%
    Effect size (Cohen's d): 1.0639
    Z-test: z=23.7888, p=0.0000
    T-test: t=-23.7888, p=0.0000
    Significant: True (alpha=0.05)
    Winner: TREATMENT

## Quick Start

    git clone https://github.com/0mohamed123/ab-testing-framework.git
    cd ab-testing-framework
    pip install numpy scipy

    cd src
    python ab_test.py

    cd ../tests
    python -m pytest test_ab_test.py -v

## Usage

    from ab_test import ABTest

    test = ABTest("CTR Experiment", alpha=0.05)
    test.add_control(control_data)
    test.add_treatment(treatment_data)
    result = test.report()

    # Sample size calculator
    from stat_tests import required_sample_size
    n = required_sample_size(baseline_rate=0.10, min_detectable_effect=0.02)

## Statistical Tests

| Test | Use Case |
|------|----------|
| Z-test | Large samples, known variance |
| T-test | Small/large samples, unknown variance |
| Chi-Square | Conversion rates (binary outcomes) |
| Cohen's d | Effect size measurement |
| Confidence Interval | Uncertainty quantification |
| Sample Size Calculator | Pre-experiment planning |

## Test Results

    10 passed | 0 failed

    Tests cover: significant detection, non-significant detection,
    lift calculation, control stats, z-test, confidence interval,
    effect size, sample size, error handling, single value input

## Technologies

- Python 3.12
- NumPy
- SciPy (statistical tests)
- pytest (10 tests)