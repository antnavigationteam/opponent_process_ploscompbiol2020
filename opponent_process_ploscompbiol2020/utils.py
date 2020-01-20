import numpy as np


def pol2cart(rho, theta):
    """Converts polar coordinates to cartesian coordinates."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


def circmean(data, weights=None, axis=None):

    if weights is None:
        # if no specific weighting has been specified, assume no binning
        weights = np.ones_like(data)
    else:
        if not np.all(weights.shape == data.shape):
            print('Input dimensions do not match!')

    all_cos = np.sum(weights * np.cos(data), axis) / np.sum(weights, axis)
    all_sin = np.sum(weights * np.sin(data), axis) / np.sum(weights, axis)

    theta = np.arctan2(all_sin, all_cos)

    return theta


def circ_r(alpha, w=None, d=0, axis=0):
    #   r = circ_r(alpha, w, d)
    #   Computes mean resultant vector length for circular data.
    #
    #   Input:
    #     alpha	sample of angles in radians
    #     [w		number of incidences in case of binned angle data]
    #     [d    spacing of bin centers for binned data, if supplied
    #           correction factor is used to correct for bias in
    #           estimation of r, in radians (!)]
    #           (per default do not apply correct for binned data)
    #     [axis  compute along this dimension, default is 1]
    #
    #     If dim argument is specified, all other optional arguments can be
    #     left empty: circ_r(alpha, [], [], dim)
    #
    #   Output:
    #     r		mean resultant length
    #
    # PHB 7/6/2008
    #
    # References:
    #   Statistical analysis of circular data, N.I. Fisher
    #   Topics in circular statistics, S.R. Jammalamadaka et al.
    #   Biostatistical Analysis, J. H. Zar
    #
    # Circular Statistics Toolbox for Matlab

    # By Philipp Berens, 2009
    # berens@tuebingen.mpg.de - www.kyb.mpg.de/~berens/circStat.html

    if w is None:
        # if no specific weighting has been specified
        # assume no binning has taken place
        w = np.ones_like(alpha)
    else:
        if np.all(w.shape) != np.all(alpha.shape):
            print('Input dimensions do not match')

    # compute weighted sum of cos and sin of angles
    r = np.sum(np.multiply(w, np.exp(1j * alpha)), axis)

    # obtain length
    r = np.divide(abs(r), np.sum(w, axis))

    # for data with known spacing, apply correction factor to correct for bias
    # in the estimation of r (see Zar, p. 601, equ. 26.16)
    if d != 0:
        c = d / 2 / np.sin(d / 2)
        r = c * r

    return r


def subsample_vertices(world, level=1, xlims=None, ylims=None, zmin=None):
    vertices = world['vertices']['pos']
    shape = vertices.shape[0]

    r = shape % level

    subsampled = np.mean(vertices[:shape - r, :].reshape(-1, level, 3), axis=1)

    zmin = np.asanyarray(zmin)
    xlims = np.asanyarray(xlims)
    ylims = np.asanyarray(ylims)

    if None not in zmin:
        subsampled = subsampled[subsampled[:, 2] > zmin]

    if None not in xlims:
        subsampled = subsampled[(subsampled[:, 0] > xlims[0]) & (subsampled[:, 0] < xlims[1])]

    if None not in ylims:
        subsampled = subsampled[(subsampled[:, 1] > ylims[0]) & (subsampled[:, 1] < ylims[1])]

    return subsampled


def initialize_simulation(virtual_world_name):
    # TODO: Reactivate this once the simulation package is public
    raise NotImplementedError("The simulation implementation depends on a package that is not yet public!"
                              "\nPlease consider using your own solution in the meantime.")
    # ...
    # ...
    # return world_mesh, simulation_engine