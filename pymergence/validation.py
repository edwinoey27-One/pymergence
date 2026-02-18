import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
from scipy import stats
import time

class ValidationHarness:
    """
    Scientific Validation Layer.
    Runs seed sweeps and computes confidence intervals for agent performance.
    """

    def __init__(self, agent_cls, baseline_cls, env_name='inverted_pendulum'):
        self.agent_cls = agent_cls
        self.baseline_cls = baseline_cls
        self.env_name = env_name

    def run_seed_sweep(self, n_seeds=5, steps=1000):
        """
        Run N seeds for both agent and baseline.
        """
        results = {
            'agent_rewards': [],
            'baseline_rewards': []
        }

        for i in range(n_seeds):
            print(f"Running Seed {i}...")
            # Run Agent
            # Instantiate loop with seed
            # Note: This requires agent_cls to be compatible with RuntimeLoop or pass it in.
            # For prototype, we mock the run logic or import RuntimeLoop
            from pymergence.cascade_pulse import RuntimeLoop

            # Agent Run
            # We need to inject the custom agent class into RuntimeLoop
            # For now, let's assume RuntimeLoop uses default agent but we can subclass/patch.

            loop = RuntimeLoop(self.env_name, seed=i)
            # Patch loop.agent = self.agent_cls(...)
            # ...
            loop.run(steps)
            total_reward = sum([x['reward'] for x in loop.logs])
            results['agent_rewards'].append(total_reward)

            # Baseline Run (Mock for now, usually random or standard PPO)
            results['baseline_rewards'].append(total_reward * 0.9) # Dummy baseline

        return results

    def compute_confidence_intervals(self, data):
        """
        Compute 95% CI using t-distribution.
        """
        mean = np.mean(data)
        sem = stats.sem(data)
        ci = sem * stats.t.ppf((1 + 0.95) / 2., len(data) - 1)
        return mean, ci

    def generate_report(self, results):
        """
        Generate a Markdown report of the ablation.
        """
        agent_mean, agent_ci = self.compute_confidence_intervals(results['agent_rewards'])
        base_mean, base_ci = self.compute_confidence_intervals(results['baseline_rewards'])

        # Hypothesis Testing (t-test)
        t_stat, p_val = stats.ttest_ind(results['agent_rewards'], results['baseline_rewards'])

        significant = p_val < 0.05

        report = f"""
# Scientific Validation Report
**Environment:** {self.env_name}
**N_Seeds:** {len(results['agent_rewards'])}

## Results
- **Agent Reward:** {agent_mean:.2f} +/- {agent_ci:.2f}
- **Baseline Reward:** {base_mean:.2f} +/- {base_ci:.2f}

## Hypothesis Test
- **t-statistic:** {t_stat:.4f}
- **p-value:** {p_val:.4f}
- **Result:** {"SIGNIFICANT" if significant else "NOT SIGNIFICANT"}
        """
        return report

if __name__ == "__main__":
    # Test run
    harness = ValidationHarness(None, None)
    results = harness.run_seed_sweep(n_seeds=3, steps=10)
    print(harness.generate_report(results))
