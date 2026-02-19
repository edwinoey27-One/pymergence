# Fork Innovation Roadmap

This roadmap defines the experimental and specialized features unique to the `SODC` fork.

## 1. Agentic Integration (SODC Core)
- **Goal:** Full integration with SODC Genesis Pulse.
- **Components:** `cascade_pulse.py`, `sodc_adapter.py`.
- **Status:** Implemented V1.
- **Next Steps:**
    - Live training loops with real-time feedback.
    - Integration with LLM planners.

## 2. Geometric Causal Emergence
- **Goal:** Analyzing causal structures in continuous physical/molecular systems.
- **Components:** `molecular.py`, `equivariant.py`.
- **Status:** Implemented basic Lennard-Jones simulation and invariant partitions.
- **Next Steps:**
    - Scale to 3D proteins.
    - Integrate `e3nn` more deeply for vector field inputs.

## 3. Quantum Causal Emergence
- **Goal:** Finding emergent structure in quantum channels.
- **Components:** `quantum.py`.
- **Status:** Implemented circuit optimization loop.
- **Next Steps:**
    - Run on real hardware (IBM Quantum via Qiskit).
    - Analyze larger systems (12+ qubits) using tensor networks.

## 4. Distributed Scaling
- **Goal:** Massive-scale coarse-graining search.
- **Components:** `distributed/manager.py` (Ray).
- **Status:** Implemented actor pattern.
- **Next Steps:**
    - Multi-node cluster deployment.
    - Asynchronous genetic algorithms for partition search.
