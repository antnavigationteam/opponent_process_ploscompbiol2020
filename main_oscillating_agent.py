import numpy as np
import matplotlib.pyplot as plt

import opponent_process as op
from opponent_process.utils import pol2cart, initialize_simulation
from opponent_process.plotting import pathangles


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# --- General ---
virtual_world = 'canberra'

# Vertical coordinate, keeping it constant at 2 cm
# ground_distance = 0.02      # in meters

# Resolution of the views (in px across azimuth)
views_resolution = 72       # 72 = 5 deg/px
# views_resolution = 36     # 36 = 10 deg/px

# Location of the nest in the world
# setup_loc = (10, 0)
setup_loc = (20, -20)

recompute_normalisation_params = False
normalisation_params = (0.0, 0.2)       # ignored if above option is True

attractive_views_only = False


# --- Learning Walks ---
# Initial angle for the Learning Walk spiral
LW_first_theta = 180        # in degrees
# LW_first_theta = np.random.uniform(0, 360)

LW_max_radius = 2           # in meters
# LW_max_radius = 0.1       # Small Learning walks
LW_nb_views = 25
LW_spiral_rotations = 2
LW_max_noise_angl = 0       # in degrees
# LW_max_noise_angl = 135   # in degrees


# --- Training Route ---

# Starting distance from the nest
TR_start_dist = -10         # in meters
TR_steps_length = 0.2       # in meters


# --- Release points ---

# Initial angle for the release points
REL_first_theta = 0.0       # in degrees
# REL_first_theta = np.random.uniform(0, 360)

REL_distance = 5
REL_nb_rep = 10

# Steps (or time) for the agent to walk
REL_nb_steps = 150
# REL_nb_steps = 600       # for smaller steps, allow more steps


REL_step_length = 0.2      # in meters
# REL_step_length = 0.05   # for Small Learning Walks, smaller steps are safer

# --- Oscillator parameters : Randomness, Gain and Baseline ---

# Randomness
mu = 0.0
sigma = 10.0

oscillator_gain = 250
# oscillator_gain = np.random.uniform(0, 10) * 10e10  # Infinite gain

turn_baseline = 90


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Import 3D world for generating snapshots in
world_mesh, simulation_engine = initialize_simulation(virtual_world)

# Define base coordinates for the Learning Walk
lw_positioal_thetas = np.deg2rad(LW_first_theta) + np.linspace(0.0,
                                                               LW_spiral_rotations * 2 * np.pi,
                                                               LW_nb_views, endpoint=False)
radii = np.linspace(0.1, LW_max_radius, LW_nb_views)            # Learning Walk is a spiral, so radii are in a linspace
lw_x, lw_y = pol2cart(radii, lw_positioal_thetas)
lw_thetas = lw_positioal_thetas - np.pi                         # The orientation of the views (towards the nest)


# Add optional angular noise
lw_ang_noise = np.random.uniform(-LW_max_noise_angl, LW_max_noise_angl, LW_nb_views)
lw_thetas += np.deg2rad(lw_ang_noise)


# Define base coordinates for the Training Route
tr_x = np.arange(TR_start_dist, -LW_max_radius, TR_steps_length)
tr_y = np.zeros_like(tr_x)
tr_thetas = np.zeros_like(tr_x)


# Put all coordinates together and recenter to another location
all_x = np.hstack((tr_x, lw_x)) + setup_loc[0]
all_y = np.hstack((tr_y, lw_y)) + setup_loc[1]
all_z = np.ones_like(all_x) * ground_distance
all_thetas = np.hstack((tr_thetas, lw_thetas))

mem_coords = np.stack((all_x, all_y), axis=1)
nb_views_total = len(all_x)


# Optionally display the generated coordinates
# pathangles(all_thetas, all_x, all_y, queue_length=0.1, jump=1, color='k')

#  Acquire the views
print("Learning attractive views...")
mem_viewbank_attr = op.get_views_resized(all_x, all_y, all_z,
                                         np.rad2deg(all_thetas),
                                         simulation_engine,
                                         resolution=views_resolution)

if not attractive_views_only:
    print("Learning repulsive views...")
    mem_viewbank_rep = op.get_views_resized(all_x, all_y, all_z,
                                            np.rad2deg(all_thetas + np.pi),  # +pi because repulsive views look away from nest
                                            simulation_engine,
                                            resolution=views_resolution)
else:
    mem_viewbank_rep = None


#  Optionally recompute the normalisation parameters
if recompute_normalisation_params:

    # Do a "pirouette" to get min and max values used for online normalisation of the familiarity

    # Get the best familiarity value:
    # On a point very close to the beginning of the training route
    viewbank_pirouette_familiar = op.get_views_resized(all_x[0] + 0.1,  # just 10 cm away from route (in X)
                                                       all_y[0] + 0.1,  # and also 10 cm in Y
                                                       all_z[0],
                                                       list(range(360)),  # and we sample all around for the rIDF
                                                       simulation_engine,
                                                       resolution=views_resolution)

    rIDF_familiar = op.image_difference(viewbank_pirouette_familiar, mem_viewbank_attr)
    bestmatch_score = rIDF_familiar.min()

    # Get the worst familiarity values:
    # We compare the memorised views with views from 4 different distant positions and take the worst match overall
    list_x_pirouette_unfamiliar = np.array([25, 39, 18, 20])
    list_y_pirouette_unfamiliar = np.array([-25, -33, 19, 4])

    distant_locs = np.array([[25, 25],
                             [39, -33],
                             [18, 19],
                             [20, 4],
                             ])

    rIDF_unfamiliar = np.zeros((distant_locs.shape[0], 360))

    for i, loc in enumerate(distant_locs):
        viewbank_pirouette_unfamiliar = op.get_views_resized(loc[0],
                                                             loc[1],
                                                             ground_distance,
                                                             list(range(360)),
                                                             simulation_engine,
                                                             resolution=views_resolution)

        rIDF_unfamiliar[i] = op.image_difference(viewbank_pirouette_unfamiliar, mem_viewbank_attr)

    worstmatch_score = rIDF_unfamiliar.max()

    normalisation_params = (np.round(bestmatch_score, decimals=1), np.round(worstmatch_score, decimals=1))


# Get release locations coordinates
r_thetas = np.deg2rad(REL_first_theta) + np.linspace(0.0, 2 * np.pi, REL_nb_rep, endpoint=False)
r_x, r_y = pol2cart(REL_distance, r_thetas)
release_coords = np.stack((r_x, r_y), axis=1) + setup_loc

# plt.scatter(mem_coords[:, 0], mem_coords[:, 1])
# plt.scatter(release_coords[:, 0], release_coords[:, 1])
# plt.axis('equal')


# ----------------------------------
#          Release agent
# ----------------------------------

X = np.zeros((REL_nb_rep, REL_nb_steps))
Y = np.zeros((REL_nb_rep, REL_nb_steps))
theta_osc = np.zeros((REL_nb_rep, REL_nb_steps))
turns = np.zeros((REL_nb_rep, REL_nb_steps))
raw_drive = np.zeros((REL_nb_rep, REL_nb_steps))

for p in range(REL_nb_rep):
    curr_coords = (*release_coords[p, :], ground_distance)

    X[p, :], Y[p, :], theta_osc[p, :], turns[p, :], raw_drive[p, :] = op.oscillator(curr_coords,
                                                                                    mem_viewbank_attr,
                                                                                    mem_viewbank_rep,
                                                                                    REL_nb_steps,
                                                                                    turn_baseline, oscillator_gain,
                                                                                    normalisation_params,
                                                                                    REL_step_length,
                                                                                    mu, sigma,
                                                                                    simulation_engine)
    print(f'{p + 1}/{REL_nb_rep}')

    plt.scatter(*release_coords[p, :], color='red')
    plt.plot(X[p, :], Y[p, :], color='red')

plt.grid()
plt.xlabel('Distance (meters)')
plt.ylabel('Distance (meters)')
plt.axis('equal')
plt.scatter(mem_coords[:, 0], mem_coords[:, 1])
plt.suptitle('{}steps-{}baseline-{}gain'.format(REL_step_length, turn_baseline, oscillator_gain))
plt.scatter(0 + setup_loc[0], 0 + setup_loc[1], color='green')
plt.show()

# Distance to the goal at the end
dist_to_nest = np.sqrt((Y[-1] - setup_loc[1]) ** 2 + (X[-1] - setup_loc[1]) ** 2)

# dist_to_nest of the selected conditions (for example with or without LW angular noise) can then be compared as in
# the paper (boxplots)

