# SNRPG - Spiking Neural Reservoir Policy Gradient

---

## Code Overview

### `main.py`
The entry point for the simulation. It initializes the environment, agents, and runs the training loop. This script orchestrates:

- Creation of the environment (`DSA_Period`).
- Initialization of agent(s).
- Execution of training over a specified number of episodes.
- Periodic logging of performance metrics (success rates, collisions, rewards, etc.).
- Saving final results to `.hkl` files.

### `DSA_env.py`
Implements the dynamic spectrum access environment, including:
- Channel-state updates for primary users (PUs).
- Spectrum sensing and interference checks for secondary users (SUs).
- Computation of data rates (SINR) and collisions.
- Reward calculations based on spectral efficiency, interference penalties, and other criteria.

### `Agent.py`
Contains the RL agent implementations. Agents support both **Q-learning** and **Policy Gradient** methods and integrate various neural models and encoding schemes. Key functionalities:
- **choose_action**: Uses an epsilon-greedy or policy-based mechanism to select channels/power levels.
- **store_transition**: Caches state transitions for replay or policy-based updates.
- **learn**: Trains either via approximate gradient-based methods (for spiking networks) or standard backprop for MLP. DEQN or DRQN code not included in this repository.

### `Models.py`
Defines various neural network models including:
- **Linear Regression** (`LinReg`)
- **Multi-Layer Perceptron** (`MLP`)
- **Spiking Neural Network (SNN)** models, such as `Three_Layer_SNN`, `LIFNeuron`, etc.

Here, we leverage **surrogate gradients** to enable training SNNs with backprop-like approaches. The file also includes specialized modules for spiking behavior, e.g., custom activation and reset logic.

### `config.py`
A centralized configuration file where key parameters (scenario type, RL method, neural model options, hyperparameters, etc.) are set. Uses environment variables or command-line-like arguments to modify:

- **SCENARIO**: `DSS` (dynamic switching), `SSSD` (discrete spatiotemporal), `SSSC` (continuous).
- **RLTYPE**: `DQN` or `PG`.
- **REGRESSOR**: `LinReg`, `MLP`, `SNN`, `SurrGrad`, etc.
- **LSM**: Boolean to enable Liquid State Machine-based reservoir computation.
- **Learning rates, batch size, exploration schedules**, etc.

### `NeuralEncBenchmark/`
A collection of neural encoding utilities, including:
- **TTFS** and **ISI** encoders.
- Surrogate gradient training modules (`surrogate_model.py`, `surrogate_train.py`).

This subdirectory provides the building blocks for spike-based neural computations, offering various ways to encode and process real-valued signals into spike trains.

### `PCRITICAL/`
Contains the implementation of **homeostatic plasticity** (P-Critical) and **small-world reservoir** topologies. Specifically:
- `pcritical.py`: P-Critical mechanism for keeping the reservoir near-critically stable.
- `topologies.py`: Construction of small-world connectivity for LSM-like recurrent structures.
- `readout.py`: Example readout layers that extract final signals from the reservoir.

These modules allow the reservoir to self-modulate synaptic scaling and remain in an optimal dynamic range during training.

### Results and Logging
Simulation results (e.g., SU/PU throughput, reward history, and other performance metrics) are saved as `.hkl` files in the `RESULT_PATH` directory. The training progress, including loss and reward averages, is printed to the console. By default, each run is tagged with a unique configuration string (e.g., `DQN_SNN_...`) to keep logs organized.

---

## Configuration

The main configuration parameters are defined in `config.py`. These include (but are not limited to):

- **SCENARIO**  
  Choose between channel switching (e.g., `'DSS'`) or spatiotemporal (e.g., `'SSSD'`, `'SSSC'`) scenarios.

- **RLTYPE**  
  Select the reinforcement learning method: `'PG'` for Policy Gradient or `'DQN'` for Q-Learning.

- **REGRESSOR**  
  Specify the neural model/regressor, such as `'LinReg'`, `'MLP'`, `'SNN'`, or `'SurrGrad'`.

- **POWER LEVELS**  
  Define the set of transmit power levels for SUs, e.g., `[50, 275, 500]`.

- **LSM and P-Critical Options**  
  Set parameters for liquid state machine usage (`USE_LSM`) and specify minicolumn/macrocolumn shapes, spectral radius normalization, alpha (homeostasis strength), etc.

- **Other Hyperparameters**  
  Learning rate (`LR`), hidden layer size, batch size, exploration scheduling, and other training parameters.

Configuration can also be **overridden** via environment variables or command-line arguments.
