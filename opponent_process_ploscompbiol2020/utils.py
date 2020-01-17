import numpy as np


def pol2cart(rho, theta):
    """Converts polar coordinates to cartesian coordinates."""
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y


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