import numpy as np

import opponent_process as op
from opponent_process.utils import pol2cart, initialize_simulation


# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


# --- General ---
# virtual_world = 'canberra'

# Vertical coordinate, keeping it constant at 2 cm
ground_distance = 0.02      # in meters

# Resolution of the views (in px across azimuth)
views_resolution = 72       # 72 = 5 deg/px

# Location of the nest in the world
setup_loc = (20, -20)

attractive_views_only = False


# --- Learning Walks ---
# Initial angle for the Learning Walk spiral
LW_first_theta = 180        # in degrees

LW_max_radius = 2           # in meters
LW_nb_views = 25
LW_spiral_rotations = 2
LW_max_noise_angl = 0       # in degrees


# --- Training Route ---

# Starting distance from the nest
TR_start_dist = -10         # in meters
TR_steps_length = 0.2       # in meters


# --- Sampling coordinates ---

xy_range = 2           # in meters
xy_grid_spacing = 0.5   # in meters
theta_spacing = 5       # in degrees

# For the unidirectional sampling
fixed_direction = 90    # in degrees

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# Import 3D world for generating snapshots in
world_mesh, simulation_engine = initialize_simulation(virtual_world)


# Define base coordinates for the Learning Walk
lw_positioal_thetas = np.deg2rad(LW_first_theta) + np.linspace(0.0, LW_spiral_rotations * 2 * np.pi,
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


# Define base coordinates for the systematic sampling
xy_coords, thetas = op.get_sampling_coordinates(setup_loc, xy_range, xy_grid_spacing, theta_spacing)


# -------------------------------------
#          Launch 360deg Sampling
# -------------------------------------

rIDFs = op.grid_unfam(simulation_engine,
                      xy_coords, thetas,
                      mem_viewbank_attr, mem_viewbank_rep,
                      resolution=views_resolution)
# np.save('unfamiliarities_sampling360', rIDFs)


## Compute for each x,y position the following metrics :
# - Mean familiarity all around
# - Most familiar direction
# - Directionality of the familiarity (i.e. how much the rIDF is concentrated on one direction, or spread all around)
all_metrics = op.compute_metrics(rIDFs)


attr_metrics = all_metrics[:, :, 0]
attrrep_metrics = all_metrics[:, :, 1]

# Arrows directions correspond to the most familiar direction
# Arrows lengths = Color values (= the resultant vector, i.e. the directionality of the apparent familiarity)
arrows_dir = attr_metrics[:, :, 1]
color_vals = attr_metrics[:, :, 2]
arrows_len = attr_metrics[:, :, 2]

plotting.mapplot(color_vals, arrows_dir, arrows_len,
                 mem_coords=mem_coords,
                 colorlims=(0.0, 0.5),
                 colormap='viridis',
                 setup_loc=setup_loc,
                 world=world_mesh)

# TODO: Fix this bug with the mapplot

# ----------------------------------------------
#          Launch Unidirectional Sampling
# ----------------------------------------------

rIDFs_unidir = op.grid_unfam(simulation_engine,
                             xy_coords, np.deg2rad(fixed_direction),
                             mem_viewbank_attr, mem_viewbank_rep,
                             resolution=views_resolution)
# np.save('unfamiliarities_samplingUni', rIDFs_unidir)

unfams_attr = rIDFs_unidir[:, :, 0]
unfams_rep = rIDFs_unidir[:, :, 1]

unfams_diff = unfams_attr - unfams_rep

plotting.mapplot(unfams_attr,
                 mem_coords=mem_coords,
                 colorlims=(0, 0.2),
                 colormap='plasma_r',
                 setup_loc=setup_loc,
                 world=world_mesh)

plotting.mapplot(unfams_diff,
                 mem_coords=mem_coords,
                 colorlims=(-0.025, 0.025),
                 colormap='plasma_r',
                 setup_loc=setup_loc,
                 world=world_mesh)


