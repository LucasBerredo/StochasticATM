# StochasticATM

Collaborative hackathon project focused on applying Neural Stochastic Differential Equations (Neural SDEs) to Air Traffic Management (ATM).

The repository combines three connected workstreams:

- Data generation from flight trajectories.
- Trajectory prediction with Neural SDE models.
- Fuel-aware route optimization with a Genetic Algorithm plus SDE-based evaluation.

The current status is research and prototyping (not production).

## Project Goals

- Build a reusable baseline for ATM modeling with Neural Differential Equations.
- Keep modules interoperable so generated data can be consumed by prediction and optimization pipelines.
- Demonstrate uncertainty-aware behavior (fan charts, Monte Carlo trajectories).
- Provide a codebase that can scale with more compute and larger datasets.

## Repository Structure

```text
.
├── Data/
│   ├── Datasets/
│   │   ├── dataset_total_unido.csv
│   │   ├── flight_data.pt
│   │   └── flight_data_test.pt
│   ├── Generation/
│   │   └── generate-data-2.py
│   └── Normalization/
│       └── data_normalization.ipynb
├── Fuel Optimization/
│   ├── route_optimizer.py
│   └── route_optimizerb.py
├── Prediction model/
│   └── sde_model.ipynb
├── setup.ipynb
├── fitness_convergence.png
├── route_optimization_result.png
├── LICENCE.txt
└── README.md
```

## Technical Overview

### 1) Data Generation

Main script: [Data/Generation/generate-data-2.py](Data/Generation/generate-data-2.py)

What it does:

- Authenticates with OpenSky API (OAuth2 client credentials).
- Pulls historical arrivals for a configured airport list.
- Downloads tracks per aircraft.
- Computes derived physics features:
	- Horizontal speed.
	- Vertical rate.
	- Fuel flow estimate via OpenAP (A320 profile).
	- Cumulative fuel burn.
- Appends consolidated rows to a CSV dataset.

Inputs:

- [credentials.json](credentials.json) with OpenSky client credentials.

Output:

- Trajectory CSV (by default set inside the script).

### Data Normalization

Main notebook: [Data/Normalization/data_normalization.ipynb](Data/Normalization/data_normalization.ipynb)

What it is used for:

- Preparing raw trajectory features before model training.
- Standardizing variable scales so SDE training is numerically stable.
- Exporting normalized representations that are later consumed in prediction experiments.

Recommended usage in the pipeline:

1. Generate or update raw trajectory data.
2. Run the normalization notebook.
3. Train and evaluate models in [Prediction model/sde_model.ipynb](Prediction%20model/sde_model.ipynb).

### 2) Prediction Model (Neural SDE)

Main notebook: [Prediction model/sde_model.ipynb](Prediction%20model/sde_model.ipynb)

What it does:

- Loads tensor datasets from [Data/Datasets/flight_data.pt](Data/Datasets/flight_data.pt) and [Data/Datasets/flight_data_test.pt](Data/Datasets/flight_data_test.pt).
- Defines Neural SDE architectures for:
	- 3 variables (Fuel Flow, Speed, Altitude).
	- 6 variables (full state).
- Trains with trajectory reconstruction loss (MSE).
- Evaluates train/test MSE.
- Visualizes uncertainty with fan charts and multi-panel plots.

### 3) Fuel Optimization

Main scripts:

- [Fuel Optimization/route_optimizer.py](Fuel%20Optimization/route_optimizer.py)
- [Fuel Optimization/route_optimizerb.py](Fuel%20Optimization/route_optimizerb.py)

What they do:

- Build smooth B-spline routes between start and end points.
- Simulate weather penalty fields.
- Use a Genetic Algorithm (GA) to optimize control points.
- Evaluate route fitness with an SDE-based fuel model.
- Export plots:
	- Route result map.
	- Convergence chart.
	- Fuel uncertainty chart (in the extended variant).

## Installation

For installation and environment setup, use `setup.ipynb`.

Recommended steps:

1. Open `setup.ipynb`.
2. Run all cells in order.
3. Verify that the required dependencies are installed before running the other notebooks/scripts.

Notes:

- Torch installation may vary by platform (CPU/CUDA/MPS).
- If a dependency is missing during execution, install it in the same environment used by the notebook.

## Quick Start

Before running any module, complete the installation workflow in `setup.ipynb`.

### A) Run data generation

1. Add valid OpenSky credentials to [credentials.json](credentials.json).
2. Run:

```bash
python Data/Generation/generate-data-2.py
```

### B) Run trajectory prediction notebook

1. Open [Prediction model/sde_model.ipynb](Prediction%20model/sde_model.ipynb).
2. Execute cells in order.
3. Adjust training flags and hyperparameters inside the notebook as needed.

### C) Run route optimization

Baseline variant:

```bash
python "Fuel Optimization/route_optimizer.py"
```

Extended variant (with extra training/evaluation flow):

```bash
python "Fuel Optimization/route_optimizerb.py"
```

## Outputs and Artifacts

Typical artifacts include:

- [route_optimization_result.png](route_optimization_result.png)
- [fitness_convergence.png](fitness_convergence.png)
- Fuel fan chart images generated by optimization scripts
- Model and evaluation plots from [Prediction model/sde_model.ipynb](Prediction%20model/sde_model.ipynb)

## Reproducibility

- The notebook and scripts include deterministic seeds in key sections.
- Results can still vary because:
	- SDE simulation is stochastic.
	- Hardware backend and numeric kernels may differ.
	- Data extraction windows can change over time.

## Limitations

- Hackathon scope and limited compute budget.
- Not designed as a production ATM decision system.
- Model training quality depends strongly on data quality and trajectory coverage.
- API limits can affect data extraction throughput.

## Roadmap

- Add pinned dependency files (requirements.txt or pyproject.toml).
- Add CLI wrappers and config files for all modules.
- Add unit/integration tests for data and modeling pipelines.
- Add experiment tracking and model versioning.
- Add containerized execution and reproducible environments.

## Team and Contributions

Work was collaborative and ideas were shared across modules.

- Dataset generation: Edgar Fernandez
- Prediction model: Lucas Berredo
- Fuel optimization: Juan Boudon

## License

Copyright Lucas Berredo, Juan Boudon, Edgar Fernandez, 2026.

Licensed under EUPL. See [LICENCE.txt](LICENCE.txt).