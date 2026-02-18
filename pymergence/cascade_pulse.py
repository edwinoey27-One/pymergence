import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import polars as pl
import orjson
import time
from typing import Any, Tuple, Dict, List

# Import our core components
from pymergence.jax_core import StochasticMatrix, Partition
from pymergence.brax_bridge import BraxDataLoader

class CascadePulse(eqx.Module):
    """
    The main runtime substrate for the SODC Agentic System.
    Implements the loop: Observe -> Plan -> Act -> Critic -> Memory -> Update.
    """
    actor: eqx.Module
    critic: eqx.Module
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    opt_state: optax.OptState

    # Internal state - list needs to be initialized in init
    # Equinox Modules are immutable Pytrees. Lists are not valid Pytree nodes unless static.
    # However, for mutable state like memory buffer, we usually keep it outside the Module
    # or treat it as an array buffer.
    # For this prototype, we'll keep memory outside or use eqx.field(static=False) but lists are tricky.

    # Correction: Remove memory from Module fields if it's mutable python list.
    # Keep it in the RuntimeLoop instead.

    def __init__(self, key, obs_size, action_size):
        key1, key2 = jax.random.split(key)
        self.actor = eqx.nn.MLP(obs_size, action_size, width_size=64, depth=2, key=key1)
        self.critic = eqx.nn.MLP(obs_size, 1, width_size=64, depth=2, key=key2)

        self.optimizer = optax.adam(1e-3)
        self.opt_state = self.optimizer.init(eqx.filter(self, eqx.is_array))

    def act(self, obs, key=None):
        """
        Policy: pi(a|s)
        For continuous control, we output mean of action dist.
        """
        action_mean = self.actor(obs)
        if key is not None:
            noise = jax.random.normal(key, action_mean.shape) * 0.1
            return jnp.tanh(action_mean + noise)
        return jnp.tanh(action_mean)

    def critic_score(self, obs):
        """Value: V(s)"""
        return self.critic(obs)

    def update(self, batch: Dict[str, jax.Array]):
        """
        Update agent weights based on experience.
        """
        def loss_fn(model):
            obs = batch['obs']
            # vmap critic over batch
            values = jax.vmap(model.critic)(obs)
            return -jnp.mean(values)

        loss, grads = eqx.filter_value_and_grad(loss_fn)(self)
        updates, new_opt_state = self.optimizer.update(grads, self.opt_state, self)
        new_agent = eqx.apply_updates(self, updates)

        # Return new instance with updated opt_state
        # We need to manually set the new opt_state on the new agent?
        # Equinox modules are immutable.
        # Ideally CascadePulse holds opt_state.
        # We can return (new_agent, new_opt_state) tuple if managing externally,
        # or use eqx.tree_at to update it inside.

        new_agent = eqx.tree_at(lambda m: m.opt_state, new_agent, new_opt_state)
        return new_agent, loss

class RuntimeLoop:
    """
    Manages the environment interaction loop.
    """
    def __init__(self, env_name, seed=0):
        self.loader = BraxDataLoader(env_name)
        from brax import envs
        self.env = envs.create(env_name=env_name)
        self.key = jax.random.PRNGKey(seed)
        self.state = self.env.reset(rng=self.key)
        self.step_fn = jax.jit(self.env.step)

        obs_size = self.env.observation_size
        act_size = self.env.action_size

        self.agent = CascadePulse(self.key, obs_size, act_size)
        self.logs = [] # Memory kept here

    def step(self):
        # 1. Observe
        obs = self.state.obs

        # 2. Plan/Act
        self.key, subkey = jax.random.split(self.key)
        action = self.agent.act(obs, subkey)

        # 3. Step Physics
        next_state = self.step_fn(self.state, action)
        reward = next_state.reward

        # 4. Memory
        transition = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_state.obs,
            "time": time.time()
        }
        self.logs.append(transition)

        self.state = next_state
        return transition

    def run(self, steps=1000):
        print(f"Running Cascade Pulse for {steps} steps...")
        for i in range(steps):
            self.step()

        return self.save_logs()

    def save_logs(self, filepath="logs.json"):
        if not self.logs:
            return None

        obs_stack = jnp.stack([x['obs'] for x in self.logs])
        obs_cpu = jax.device_get(obs_stack)

        df = pl.DataFrame(obs_cpu)
        meta = {"env": "brax", "steps": len(self.logs)}

        with open(filepath, "wb") as f:
            f.write(orjson.dumps(meta))

        return df

if __name__ == "__main__":
    loop = RuntimeLoop('inverted_pendulum')
    loop.run(100)
