import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from argosim import antenna_utils, imaging_utils, metrics_utils, plot_utils

# # Parameters 
# n_generations = 50
# n_antennas = 30
# space_size = 4000
# mutation_rate = 1
# n_mutations = 30
d_min = 300.0
# sigma = 0.5 * d_min   
fov_size = (0.03, 0.03)
im_size = (256, 256)

OUT = Path("../outputs"); OUT.mkdir(parents=True, exist_ok=True)


def objective(antenna_pos, w_sll=1, w_fwhm=1, w_ecc=0.5):
    """
    Evaluate a configuration: 
    score = w_sll*SLL + w_fwhm*(FWHM_x+FWHM_y) + w_ecc*eccentricity
    (we directly minimize SLL: no target_sll and no abs()).
    """
    try:
        b_enu = antenna_utils.get_baselines(antenna_pos)
        track, _ = antenna_utils.uv_track_multiband(
            b_ENU=b_enu, track_time=3, n_times=10, f=1e9, df=1e8, n_freqs=10
        )
        mask, _ = imaging_utils.grid_uv_samples(track, sky_uv_shape=im_size, fov_size=fov_size)
        beam = imaging_utils.uv2sky(mask)
        beam = np.abs(beam) / np.max(np.abs(beam))
        fit = metrics_utils.fit_elliptical_beam(beam.copy(), threshold_ratio=0.2)
        metrics = metrics_utils.compute_beam_metrics(beam)

        sll_score  = w_sll  * metrics["sll_db"]
        fwhm_score = w_fwhm * (metrics["fwhm"][0] + metrics["fwhm"][1])
        ecc_score  = w_ecc  * metrics["eccentricity"]

        score = sll_score + fwhm_score + ecc_score
        return score, beam, fit, metrics
    except Exception:
        return 1e6, None, None, None

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

    best_score, best_beam, best_fit, best_metrics = objective(current_config, w_sll, w_fwhm, w_ecc)
    if (best_metrics is None) or (not np.isfinite(best_score)):
        raise RuntimeError("Initial objective() failed (metrics=None or score=inf).")

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
            score, beam, fit, metrics = objective(mutated, w_sll, w_fwhm, w_ecc)

            if (metrics is not None) and np.isfinite(score):
                candidates.append((score, mutated, beam, fit, metrics))


        candidates.sort(key=lambda x: x[0])
        best_candidate = candidates[0]

        
        if best_candidate[0] < best_score:
            best_score, current_config, best_beam, best_fit, best_metrics = best_candidate
        

        history["configs"].append(current_config.copy())
        history["sll"].append(best_metrics["sll_db"])
        history["fwhm_x"].append(best_metrics["fwhm"][0])
        history["fwhm_y"].append(best_metrics["fwhm"][1])
        history["ecc"].append(best_metrics["eccentricity"])

    elapsed = time.time() - t0
    np.save(OUT / f"{case_name.lower()}_best_config.npy", current_config)

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
        "Time_s": round(elapsed, 3),
    }
    return results, current_config, history

