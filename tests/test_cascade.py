import jax
import jax.numpy as jnp
import pytest
from pymergence.integration.cascade_pulse import CascadePulse
from pymergence.lab.validation import ValidationHarness

def test_cascade_pulse_init():
    key = jax.random.PRNGKey(0)
    agent = CascadePulse(key, obs_size=4, action_size=1)

    obs = jnp.ones(4)
    action = agent.act(obs)
    assert action.shape == (1,)

    val = agent.critic_score(obs)
    assert val.shape == (1,)

def test_validation_harness():
    # Mock
    harness = ValidationHarness(None, None)
    results = harness.run_seed_sweep(n_seeds=2, steps=5)

    assert len(results['agent_rewards']) == 2
    assert len(results['baseline_rewards']) == 2

    report = harness.generate_report(results)
    assert "Scientific Validation Report" in report
