# Roadmap: SODC Agentic Science Architecture

This roadmap defines the transition of `pymergence` from a functional analysis toolkit to a **Scientific Agentic System**. The goal is to build a rigorous, falsifiable engine for causal emergence in agentic control.

## Core Philosophy
- **Scientific Standard:** Every claim of "emergence" or "improvement" must be backed by falsifiable hypotheses, ablations, and statistical validation.
- **3-Track Architecture:**
    1.  **Production Agentic Core:** High-performance, practical control loop (JAX/Brax/Equinox).
    2.  **Scientific Validation Layer:** Rigorous testing harness (Ablations, Confidence Intervals).
    3.  **Quantum Experimental Branch:** Forward-looking research (PennyLane/Qiskit) isolated from production stability.

---

## Track 1: Production Agentic Core (The Substrate)
**Substrate:** `pymergence/cascade_pulse.py`
**Tech Stack:** Brax + JAX + Equinox + Optax + Polars + Safetensors

### Milestones
- [ ] **M1.1: The Agent Loop**
    - Implement `Goal -> Plan -> Act -> Critic -> Memory` cycle.
    - Ensure zero-copy data flow from Brax physics to JAX agent.
- [ ] **M1.2: Causal Integration**
    - Embed `StochasticMatrix` (Causal Emergence) into the `Critic` step.
    - Agent optimizes for `Reward + lambda * EffectiveInformation`.
- [ ] **M1.3: Persistence**
    - Real-time logging to Polars/Parquet.
    - Checkpointing via Safetensors.

### Pass/Fail Metrics
- **Latency:** Loop frequency > 60Hz (simulated time) on CPU/GPU.
- **Stability:** No memory leaks over 1M steps.

---

## Track 2: Scientific Validation Layer (The Standard)
**Substrate:** `pymergence/validation.py`

### Hypotheses to Validate
1.  **H1 (Adaptation):** Agents maximizing Causal Emergence adapt faster to OOD physics changes (e.g., gravity change) than baseline RL.
2.  **H2 (Controllability):** Integrated Information ($\Phi$) correlates positively with control authority (empowerment).
3.  **H3 (Structure):** Active Inference objectives lead to sparser, more interpretable causal graphs than reward-only objectives.

### Milestones
- [ ] **M2.1: The Harness**
    - Automated seed sweeps (N=10+).
    - Parallel execution of Baseline vs. Experimental agents.
- [ ] **M2.2: Statistical Rigor**
    - Automatic calculation of 95% Confidence Intervals.
    - Bootstrap hypothesis testing.

### Pass/Fail Metrics
- **Reproducibility:** Results must hold across 3 random seeds with p < 0.05.
- **Reporting:** Every run generates an automated PDF/Markdown report.

---

## Track 3: Quantum Experimental Branch (The Frontier)
**Substrate:** `pymergence/prototypes/quantum_native_agent.py`
**Tech Stack:** PennyLane + CUDA Quantum (if avail)

### Milestones
- [ ] **M3.1: Hybrid Controller**
    - Replace the Actor network with a Variational Quantum Circuit (VQC).
    - Match parameter count to classical baseline.
- [ ] **M3.2: Scaling Benchmark**
    - Measure performance at 4, 8, 12, 16 qubits.
    - Identify "Quantum Advantage" threshold (if any) for causal discovery.

### Pass/Fail Metrics
- **Feasibility:** Training runs without crashing for >1k steps.
- **Honesty:** Explicitly report where Classical > Quantum (no hype).

---

## Academic Anchors
1.  **Brax (Differentiable Physics):** [Freeman et al., 2021](https://arxiv.org/abs/2106.13281)
2.  **O-Information (Higher-Order Interactions):** [Rosas et al., 2019](https://arxiv.org/abs/1902.11239)
3.  **Causal Emergence (Effective Information):** [Hoel et al., 2013](https://pubmed.ncbi.nlm.nih.gov/24248356/)
4.  **Active Inference:** [Friston et al., 2019](https://link.springer.com/article/10.1007/s00422-019-00805-w)
