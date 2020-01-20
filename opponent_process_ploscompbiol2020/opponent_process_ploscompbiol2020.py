import numpy as np
from itertools import repeat, cycle
from skimage.transform import resize

from opponent_process_ploscompbiol2020.utils import pol2cart, circ_r, circmean


def get_views_resized(x, y, z, th, simulation_engine, resolution=360):
    """
    Outputs one or several downsampled and cropped snapshot of given coordinates
    """

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    z = np.atleast_1d(z)
    th = np.atleast_1d(th)

    nb_views = max(x.shape[0], y.shape[0], z.shape[0], th.shape[0])

    x = list(repeat(x, nb_views)) if x.shape[0] < nb_views else x
    y = list(repeat(y, nb_views)) if y.shape[0] < nb_views else y
    z = list(repeat(z, nb_views)) if z.shape[0] < nb_views else z
    th = list(repeat(th, nb_views)) if th.shape[0] < nb_views else th

    scalefactor = resolution / simulation_engine.viewport_size[0]
    width = int(np.floor(simulation_engine.viewport_size[0] * scalefactor))
    height = int(np.floor(simulation_engine.viewport_size[1] * scalefactor))

    all_views = np.zeros((nb_views, int(height / 2), width))

    for v in range(nb_views):
        view = simulation_engine.snapshot_of(x=x[v], y=y[v], z=z[v],
                                          theta_degrees=th[v],
                                          extract_channel=2,
                                          save=False)

        view_lowres = resize(view, (height, width))

        all_views[v] = view_lowres[:int(height / 2), :]

    return all_views.squeeze()


def image_difference(cur_viewbank, mem_viewbank):
    """ Compares 360 views of current position to the memory bank,
     returns the one that best matches """

    if mem_viewbank.ndim < 3:
        mem_viewbank = mem_viewbank[None, :, :]

    if cur_viewbank.ndim < 3:
        cur_viewbank = cur_viewbank[None, :, :]

    nb_mems = mem_viewbank.shape[0]
    nb_curs = cur_viewbank.shape[0]

    diff = np.zeros((nb_mems, nb_curs))

    for cur_view in range(nb_curs):

        for cur_mem in range(nb_mems):

            diff_view = cur_viewbank[cur_view] - mem_viewbank[cur_mem]
            diff[cur_mem, cur_view] = np.mean(np.abs(diff_view))

    bestmatch = np.min(diff, axis=0)

    return bestmatch


def normalize_unfamiliarity(val, minimum, maximum, mean=0.0):
    """ Normalize unfamiliarity values, typically [-0.5, 0.5] """

    amplitude = (maximum - minimum)

    shifted_val = val - minimum

    normalized = shifted_val/amplitude

    normalized += (mean - 0.5)

    return normalized


def oscillator(start_coords, mem_viewbank_attr, mem_viewbank_rep, steps, baseline, oscillator_gain, norm_minmax, step_length, mu, sigma, simulation_engine):
    """ Oscillating moving agent """

    # Get the resolution (nb of azimutal px) of the memorised view,
    # automatically get current view at this resolution

    resolution = mem_viewbank_attr.shape[-1]

    randomness = np.random.normal(mu, sigma, steps)

    theta_osc = np.zeros(steps)
    x = np.zeros(steps)
    y = np.zeros(steps)
    turn_effective = np.zeros(steps)
    raw_overall_drive = np.zeros(steps)

    theta_osc[0] = np.random.randint(0, 360)
    x[0], y[0], z = start_coords

    osc = cycle([1, -1])
    for i in range(steps):

        # Alternate between oscillator states (left-right-left-right-...)
        oscil_state = next(osc)

        view = get_views_resized(x[i], y[i], z, theta_osc[i] % 360, simulation_engine, resolution=resolution)

        unfam_attr = image_difference(view, mem_viewbank_attr)

        # Neuron between 0 (very familiar) and 1 (very unfamiliar)
        unfam_attr_norm = normalize_unfamiliarity(unfam_attr, *norm_minmax, mean=0.5)

        if mem_viewbank_rep is not None:
            unfam_rep = image_difference(view, mem_viewbank_rep)

            # Neuron between 0 (very familiar) and 1 (very unfamiliar)
            unfam_rep_norm = normalize_unfamiliarity(unfam_rep, *norm_minmax, mean=0.5)

        else:
            unfam_rep_norm = 0.5

        raw_overall_drive[i] = unfam_attr_norm - unfam_rep_norm

        turn_intensity = baseline + (raw_overall_drive[i] * oscillator_gain)

        turn_intensity = np.clip(turn_intensity, 0, 180)

        turn_effective[i] = turn_intensity * oscil_state + randomness[i]

        if i < (steps-1):     # Just to prevent the last step

            # Rotate according to oscillator state
            newtheta = (theta_osc[i] + turn_effective[i]) % 360

            x_displacement, y_displacement = pol2cart(step_length, np.deg2rad(newtheta))

            theta_osc[i + 1] = newtheta
            x[i + 1] = x[i] + x_displacement
            y[i + 1] = y[i] + y_displacement

    return x, y, theta_osc, turn_effective, raw_overall_drive


def get_sampling_coordinates(location, xy_range, xy_spacing, theta_spacing):

    start_x = location[0] - xy_range / 2
    end_x = location[0] + xy_range / 2
    start_y = location[1] - xy_range / 2
    end_y = location[1] + xy_range / 2

    X = np.arange(start_x, end_x, xy_spacing)
    Y = np.arange(start_y, end_y, xy_spacing)
    XY = np.stack((X, Y), axis=1)
    thetas = np.arange(0, 360, theta_spacing)

    return XY, thetas


def compare_rIDFs(ridf_attr, ridf_rep):

    dset = np.zeros((2, 3))

    thetas = np.linspace(0, 2 * np.pi, len(ridf_attr), endpoint=False)

    ridf_attr = ridf_attr - ridf_attr.min()
    ridf_rep = ridf_rep - ridf_rep.min()

    ridf_both = ridf_attr - ridf_rep
    ridf_both = ridf_both - ridf_both.min()

    # We take the circular mean of all the rotIDF
    mean_attr = circmean(np.deg2rad(thetas), weights=ridf_attr)
    mean_both = circmean(np.deg2rad(thetas), weights=ridf_both)

    resultant_vector_attr = circ_r(np.deg2rad(thetas), w=ridf_attr)
    resultant_vector_both = circ_r(np.deg2rad(thetas), w=ridf_both)

    # We must use this method instead of np.argmin, because argmin only returns the first instance:

    # Get the corresponding theta
    angles_where_min_attr = thetas[np.where(ridf_attr == 0)[0]]
    angles_where_min_both = thetas[np.where(ridf_both == 0)[0]]

    # Just to make sure
    theta_min_attr = np.rad2deg(circmean(np.deg2rad(angles_where_min_attr)))
    theta_min_both = np.rad2deg(circmean(np.deg2rad(angles_where_min_both)))

    dset[0, :] = mean_attr, theta_min_attr, resultant_vector_attr
    dset[1, :] = mean_both, theta_min_both, resultant_vector_both

    return dset


def compute_metrics(unfam_values):

    dset = np.zeros((*unfam_values.shape[:3], 3))

    for i in range(unfam_values.shape[0]):
        for j in range(unfam_values.shape[1]):
            ridf_attr, ridf_rep = unfam_values[i, j]

            dset[i, j, :, :] = compare_rIDFs(ridf_attr, ridf_rep)

    return dset


def grid_unfam(simulation_engine, xy_coords, thetas, mem_viewbank_attr, mem_viewbank_rep, resolution=72, z_offset=0.02):

    X, Y = np.split(xy_coords, 2, axis=1)
    T = np.atleast_1d(thetas)

    unfam_values = np.zeros((len(X), len(Y), 2, len(T)))

    i = 1
    for j, x in enumerate(X):
        for k, y in enumerate(Y):

            print(f'Position {i}/{len(X)*len(Y)}')

            # For current x,y coord, get familiarity all around (i.e. get the rIDFs)
            ridf_attr = np.zeros(len(T))
            ridf_rep = np.zeros(len(T))

            for l, t in enumerate(T):
                view = get_views_resized(x, y, z_offset, t, simulation_engine, resolution=resolution)
                ridf_attr[l] = image_difference(view, mem_viewbank_attr)
                ridf_rep[l] = image_difference(view, mem_viewbank_rep)

            unfam_values[j, k] = ridf_attr, ridf_rep

            i += 1

    return unfam_values.squeeze()