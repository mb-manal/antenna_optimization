This repository contains a Python implementation to optimize the layout of antennas for radio interferometry using a random mutation-based optimization algorithm. The goal is to minimize the three metrics (side lobe level, excentricity, full width at half maximum) by iteratively adjusting antenna positions.

# Optimization objective

Each antenna configuration is evaluated using this formula:
Score = w_sll × |SLL - target_sll| + w_fwhm × (FWHM_x + FWHM_y) + w_ecc × Eccentricity

Where:
- `w_sll`, `w_fwhm`, `w_ecc` are adjustable weights.
- `target_sll` = -15 dB (default)


# Tested weight scenarios

The script runs the optimization for different trade-offs between circularity, sharpness, and sidelobe suppression:

weight_scenarios = [
    {"w_sll": 1.0, "w_fwhm": 0.1, "w_ecc": 0.6},
    {"w_sll": 1.0, "w_fwhm": 0.05, "w_ecc": 0.1},
    {"w_sll": 1.0, "w_fwhm": 0.5, "w_ecc": 0.001}
]

# Dependecies 

- numpy

- matplotlib

- scipy

- argosim, accessible via : 

conda env 

conda activate argosim

# Plots and outputs 

For each weight scenario, the script generates:

Score evolution

Side Lobe Level (SLL) per generation

FWHM_x and FWHM_y

Beam eccentricity

Antenna layout before and after optimization

# Notebooks overview 

All the core experiments and results are available as Jupyter notebooks in the notebooks/ folder:

Genetic Algorithm – Main Method

notebooks/Genetic_algo.ipynb

→ Main method used for antenna optimization. Applies selection, mutation, and elitism with distance constraints.

Particle Swarm Optimization (PSO)

notebooks/Particle_Swarm_Optimization.ipynb

→ Alternative global optimization method used for comparison with GA.

Random Forest (Machine Learning)

notebooks/Random_Forest.ipynb

→ Regression model trained to predict beam metrics from antenna positions. Used as a fast approximation or surrogate model.

# How to use 

1. Clone the repository:

git clone https://github.com/your-username antenna_optimization.git

cd antenna_optimization

2. Run the notebooks via Jupyter or VSCode:

jupyter notebook notebooks/Genetic_algo.ipynb
