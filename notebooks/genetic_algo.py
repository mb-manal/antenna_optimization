import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from argosim import antenna_utils, imaging_utils, metrics_utils, plot_utils

# Global parameters
d_min = 300.0
fov_size = (0.03, 0.03)
im_size = (256, 256)



def compute_beam(antenna_pos):
    """
    Compute the normalized synthetic beam from antenna positions.
    Returns: beam.
    """
    b_enu = antenna_utils.get_baselines(antenna_pos)
    track, _ = antenna_utils.uv_track_multiband(
        b_ENU=b_enu, track_time=3, n_times=10, f=1e9, df=1e8, n_freqs=10
    )
    mask, _ = imaging_utils.grid_uv_samples(track, sky_uv_shape=im_size, fov_size=fov_size)
    beam = imaging_utils.uv2sky(mask)
    beam = np.abs(beam)
    beam = beam / np.max(beam)  
    return beam

def compute_metrics(beam):
    """
    Compute beam metrics (SLL in dB, FWHM tuple, eccentricity.
    Returns: metrics (dict).
    """
    fit = metrics_utils.fit_elliptical_beam(beam.copy(), threshold_ratio=0.2)
    metrics = metrics_utils.compute_beam_metrics(beam)
    return metrics

def objective(metrics, w_sll=1, w_fwhm=1, w_ecc=0.5):
    """
    Compute score from metrics only.
    score = w_sll*SLL + w_fwhm*(FWHM_x+FWHM_y) + w_ecc*eccentricity.
    """
    try:

        sll_score  = w_sll  * metrics["sll_db"]
        fwhm_score = w_fwhm * (metrics["fwhm"][0] + metrics["fwhm"][1])
        ecc_score  = w_ecc  * metrics["eccentricity"]

        score = sll_score + fwhm_score + ecc_score
        return score
    except Exception:
        
        return 1e6


def min_distance(config, d_min=d_min):
    """True if all antenna pairs are separated by at least d_min."""
    for i in range(len(config)):
        for j in range(i+1, len(config)):
            if np.linalg.norm(config[i] - config[j]) < d_min:
                return False
    return True

def mutate_config(base_config, mutation_rate=0.2, d_min=100, sigma=50, max_trials=10):
    """
    Gaussian mutation (std = sigma) of a subset of antennas.
    Retries until the minimum distance constraint is satisfied.
    """
    for _ in range(max_trials):
        config = base_config.copy()
        for i in range(len(config)):
            if np.random.rand() < mutation_rate:
                config[i] += np.random.normal(0, sigma, size=3)
                config[i][2] = 0
        if min_distance(config, d_min=d_min):
            return config
    return base_config


def run_optimization(case_name, init_config,
                     n_generations=50, n_mutations=20,
                     mutation_rate=0.2, d_min=100, sigma=50,
                     w_sll=1, w_fwhm=1, w_ecc=0.5):
    """
    Run the optimization for a given initial configuration and 
    return (results_dict, best_config, history).
    """
    t0 = time.time()
    current_config = init_config.copy()
    try:
        best_beam = compute_beam(current_config)
        best_metrics = compute_metrics(best_beam)
        best_score = objective(best_metrics, w_sll, w_fwhm, w_ecc)
    except Exception as e:
        raise RuntimeError(f"Initial compute_beam/compute_metrics failed: {e}")

    history = {
        "sll":     [best_metrics["sll_db"]],
        "fwhm_x":  [best_metrics["fwhm"][0]],
        "fwhm_y":  [best_metrics["fwhm"][1]],
        "ecc":     [best_metrics["eccentricity"]],
        "configs": [current_config.copy()],
    }

    for gen in range(n_generations):
        candidates = []

        for _ in range(n_mutations):
            mutated = mutate_config(current_config, mutation_rate=mutation_rate, d_min=d_min, sigma=sigma)
            try:
                beam = compute_beam(mutated)
                metrics = compute_metrics(beam)
                score = objective(metrics, w_sll, w_fwhm, w_ecc)

                candidates.append((score, mutated, beam, metrics))
            except Exception:
                continue
        
        if not candidates:
            history["configs"].append(current_config.copy())
            history["sll"].append(best_metrics["sll_db"])
            history["fwhm_x"].append(best_metrics["fwhm"][0])
            history["fwhm_y"].append(best_metrics["fwhm"][1])
            history["ecc"].append(best_metrics["eccentricity"])
            continue

        candidates.sort(key=lambda x: x[0])
        best_candidate = candidates[0]

        if best_candidate[0] < best_score:
            best_score, current_config, best_beam, best_metrics = best_candidate

        history["configs"].append(current_config.copy())
        history["sll"].append(best_metrics["sll_db"])
        history["fwhm_x"].append(best_metrics["fwhm"][0])
        history["fwhm_y"].append(best_metrics["fwhm"][1])
        history["ecc"].append(best_metrics["eccentricity"])

    results = {
        "Case": case_name,
        "Score": float(best_score),
        "SLL_dB": float(best_metrics["sll_db"]),
        "FWHM_x": float(best_metrics["fwhm"][0]),
        "FWHM_y": float(best_metrics["fwhm"][1]),
        "FWHM_sum": float(best_metrics["fwhm"][0] + best_metrics["fwhm"][1]),
        "Eccentricity": float(best_metrics["eccentricity"]),
        "Generations": n_generations,
        "Mut_per_gen": n_mutations,
        "Sigma": sigma,
        "Weights": f"w_sll={w_sll}, w_fwhm={w_fwhm}, w_ecc={w_ecc}",
        "Runtime_s": round(time.time() - t0, 3),
    }
    return results, current_config, history
