import numpy as np
from datetime import datetime
from stat_tests import (z_test, t_test, chi_square_test,
                        confidence_interval, effect_size_cohens_d,
                        required_sample_size)


class ABTest:
    def __init__(self, name, metric='mean', alpha=0.05):
        self.name = name
        self.metric = metric
        self.alpha = alpha
        self.control = []
        self.treatment = []
        self.created_at = datetime.now().isoformat()

    def add_control(self, data):
        self.control.extend(data if hasattr(data, '__iter__') else [data])
        return self

    def add_treatment(self, data):
        self.treatment.extend(data if hasattr(data, '__iter__') else [data])
        return self

    def analyze(self):
        if not self.control or not self.treatment:
            raise ValueError("Need data in both control and treatment groups")

        c = np.array(self.control)
        t = np.array(self.treatment)

        result = {
            'name': self.name,
            'alpha': self.alpha,
            'control': {
                'n': len(c),
                'mean': round(float(np.mean(c)), 4),
                'std': round(float(np.std(c, ddof=1)), 4),
                'ci': confidence_interval(c, 1 - self.alpha)
            },
            'treatment': {
                'n': len(t),
                'mean': round(float(np.mean(t)), 4),
                'std': round(float(np.std(t, ddof=1)), 4),
                'ci': confidence_interval(t, 1 - self.alpha)
            },
            'lift': round((np.mean(t) - np.mean(c)) / np.mean(c) * 100, 2),
            'effect_size': round(effect_size_cohens_d(c, t), 4),
        }

        z = z_test(c, t)
        t_res = t_test(c, t)
        result['z_test'] = {'z_stat': round(z['z_stat'], 4),
                            'p_value': round(z['p_value'], 4)}
        result['t_test'] = {'t_stat': round(t_res['t_stat'], 4),
                            'p_value': round(t_res['p_value'], 4)}

        result['significant'] = result['t_test']['p_value'] < self.alpha
        result['winner'] = 'treatment' if (result['significant'] and
                           np.mean(t) > np.mean(c)) else 'control'
        return result

    def report(self):
        r = self.analyze()
        print(f"\n{'='*55}")
        print(f"  A/B Test: {r['name']}")
        print(f"{'='*55}")
        print(f"  Control   (n={r['control']['n']}): "
              f"mean={r['control']['mean']:.4f} "
              f"std={r['control']['std']:.4f}")
        print(f"  Treatment (n={r['treatment']['n']}): "
              f"mean={r['treatment']['mean']:.4f} "
              f"std={r['treatment']['std']:.4f}")
        print(f"\n  Lift: {r['lift']:+.2f}%")
        print(f"  Effect size (Cohen's d): {r['effect_size']:.4f}")
        print(f"\n  Z-test: z={r['z_test']['z_stat']:.4f}, "
              f"p={r['z_test']['p_value']:.4f}")
        print(f"  T-test: t={r['t_test']['t_stat']:.4f}, "
              f"p={r['t_test']['p_value']:.4f}")
        print(f"\n  Significant: {r['significant']} (alpha={r['alpha']})")
        print(f"  Winner: {r['winner'].upper()}")
        print(f"{'='*55}\n")
        return r


def demo():
    np.random.seed(42)

    print("Demo 1: Significant difference")
    test1 = ABTest("CTR Experiment", alpha=0.05)
    test1.add_control(np.random.normal(0.10, 0.02, 1000))
    test1.add_treatment(np.random.normal(0.12, 0.02, 1000))
    test1.report()

    print("Demo 2: No significant difference")
    test2 = ABTest("Revenue Experiment", alpha=0.05)
    test2.add_control(np.random.normal(50, 10, 500))
    test2.add_treatment(np.random.normal(51, 10, 500))
    test2.report()

    print("Sample size calculator:")
    n = required_sample_size(baseline_rate=0.10,
                             min_detectable_effect=0.02)
    print(f"  Required n per group: {n}")


if __name__ == '__main__':
    demo()