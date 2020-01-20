from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import opponent_process_ploscompbiol2020 as op
from opponent_process_ploscompbiol2020.utils import pol2cart, initialize_simulation


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# --- General ---

# virtual_world = 'canberra'

# Vertical coordinate, keeping it constant at 2 cm
ground_distance = 0.02      # in meters

# Resolution of the views (in px across azimuth)
views_resolution = 72       # 72 = 5 deg/px
# views_resolution = 36     # 36 = 10 deg/px

# Location of the nest in the world
setup_loc = (10, 0)
# setup_loc = (20, -20)

normalisation_params = (0.0, 0.2)

attractive_views_only = False

SAVE_PATH = Path("./")

# --- Learning Walks ---

# Initial angle for the Learning Walk spiral
LW_first_theta = 180        # in degrees
# LW_first_theta = np.random.uniform(0, 360)

LW_nb_views = 25
LW_spiral_rotations = 2


# --- Training Route ---

# Starting distance from the nest
TR_start_dist = -10         # in meters
TR_steps_length = 0.2       # in meters


# --- Release points ---

# Initial angle for the release points
REL_first_theta = 0.0       # in degrees
# REL_first_theta = np.random.uniform(0, 360)

REL_nb_rep = 36

# Steps (or time) for the agent to walk
REL_nb_steps = 320

REL_step_length = 0.2      # in meters
# REL_step_length = 0.05


# --- Oscillator parameters : Randomness ---

mu = 0.0
sigma = 10.0


# --- Parameters to explore ---

lw_dist_list = [0.5, 2, 8]
baseline_list = [0, 45, 90, 135, 180]
gain_list = [0, 10, 100, 1000, 10000, 100000]
rel_distances_list = [0, 1, 2, 4, 8, 16, 32]


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Import 3D world for generating snapshots in
world_mesh, simulation_engine = initialize_simulation(virtual_world)


# nb_tests does not include the LW dist because we save a file for each LW dist
nb_tests = len(baseline_list) * len(gain_list) * len(rel_distances_list) * REL_nb_rep

lw_positional_thetas = np.deg2rad(LW_first_theta) + np.linspace(0.0,
                                                                LW_spiral_rotations * 2 * np.pi,
                                                                LW_nb_views,
                                                                endpoint=False)

r_thetas = np.deg2rad(REL_first_theta) + np.linspace(0.0,
                                                     2 * np.pi,
                                                     REL_nb_rep,
                                                     endpoint=False)


# ----------------------------------
#     Lauch parameter exploration
# ----------------------------------

for lw, curlw_dist in enumerate(lw_dist_list):

    # ------------------- (Re)generate LW and TR coords (for current LW size) --------------
    radii = np.linspace(0.1, curlw_dist, LW_nb_views)
    lw_x, lw_y = pol2cart(radii, lw_positional_thetas)
    lw_thetas = lw_positional_thetas - np.pi

    tr_x = np.arange(TR_start_dist, -curlw_dist, TR_steps_length)
    tr_y = np.zeros_like(tr_x)
    tr_thetas = np.zeros_like(tr_x)

    all_x = np.hstack((tr_x, lw_x)) + setup_loc[0]
    all_y = np.hstack((tr_y, lw_y)) + setup_loc[1]
    all_z = np.ones_like(all_x) * ground_distance
    all_thetas = np.hstack((tr_thetas, lw_thetas))

    mem_coords = np.stack((all_x, all_y), axis=1)

    # ------------------- (Re)generate memory views (for current LW size) --------------
    print("Learning attractive views...")
    mem_viewbank_attr = op.get_views_resized(all_x, all_y, all_z,
                                             np.rad2deg(all_thetas),
                                             simulation_engine,
                                             resolution=views_resolution)

    if not attractive_views_only:
        print("Learning repulsive views...")
        mem_viewbank_rep = op.get_views_resized(all_x, all_y, all_z,
                                                np.rad2deg(all_thetas + np.pi),
                                                simulation_engine,
                                                resolution=views_resolution)
    else:
        mem_viewbank_rep = None

    # ------------------- Launch systematic sampling (for current LW size) --------------
    dist_to_nest = np.zeros((len(baseline_list), len(gain_list), len(rel_distances_list), REL_nb_rep))

    k = 1
    for b, baseline in enumerate(baseline_list):

        for g, gain in enumerate(gain_list):

            for d, reldist in enumerate(rel_distances_list):

                r_x, r_y = pol2cart(reldist, r_thetas)
                release_coords = np.stack((r_x, r_y), axis=1) + setup_loc

                for a, relangl in enumerate(r_thetas):

                    print(f'LW dist {lw + 1}/{len(lw_dist_list)}, {nb_tests - k} tests remaining')

                    curr_coords = (release_coords[a, 0], release_coords[a, 1], ground_distance)

                    X, Y, _, _, _ = op.oscillator(curr_coords,
                                                  mem_viewbank_attr, mem_viewbank_rep,
                                                  REL_nb_steps,
                                                  baseline, gain,
                                                  normalisation_params,
                                                  REL_step_length,
                                                  mu, sigma,
                                                  simulation_engine)

                    dist_to_nest[b, g, d, a] = np.sqrt((Y[-1] - setup_loc[1]) ** 2 + (X[-1] - setup_loc[0]) ** 2)

                    k += 1

    # ------------------- Save result for current LW size --------------
    rep = 'rep' if mem_viewbank_rep is not None else ''
    np.savez_compressed(SAVE_PATH / f'parm_explo_attr{rep}_72px_LW{curlw_dist}m.npz',
                        baseline_list=baseline_list,
                        gain_list=gain_list,
                        rel_distances_list=rel_distances_list,
                        dist_to_nest=dist_to_nest)


# ------------------- Plot parameters heatmap -------------------

errors = np.median(dist_to_nest, axis=-1)

gain_list_log10 = np.log10(gain_list)
gain_list_log10[gain_list_log10 == -np.inf] = 0

baseline_to_plot = 90
baseline_idx = int(np.where(np.array(baseline_list) == baseline_to_plot)[0])
x_list = gain_list_log10
y_list = rel_distances_list
toplot = errors[baseline_idx, :, :].squeeze()
plotting.heatmap(x_list, y_list, toplot)

gain_to_plot = 100
gain_idx = int(np.where(np.array(gain_list) == gain_to_plot)[0])
x_list = baseline_list
y_list = rel_distances_list
toplot = errors[:, gain_idx, :].squeeze()
plotting.heatmap(x_list, y_list, toplot)

rel_distances_to_plot = 4
rel_distances_idx = int(np.where(np.array(rel_distances_list) == rel_distances_to_plot)[0])
x_list = gain_list_log10
y_list = baseline_list
toplot = errors[:, :, rel_distances_idx].squeeze()
plotting.heatmap(x_list, y_list, toplot)

plt.show()

