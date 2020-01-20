import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from opponent_process_ploscompbiol2020.utils import pol2cart, subsample_vertices


def pathangles(theta, X, Y, queue_length=1.0, jump=1, color='k', figsize=(16, 6)):
    """Plots ants' body orientation (takes one value every 'jump' points).
        Examples:   if jump = 1, all points are considered
                    if jump = 2, every other point is considered"""

    fig, ax = plt.subplots(figsize=figsize)

    x_theta, y_theta = pol2cart(queue_length, np.pi + theta)
    x_dir = X + x_theta
    y_dir = Y + y_theta

    X = X[::jump]
    X_dir = x_dir[::jump]
    Y = Y[::jump]
    Y_dir = y_dir[::jump]

    xarray = np.array([X, X_dir])
    yarray = np.array([Y, Y_dir])

    ax.scatter(X, Y, marker='.', color=color)
    ax.plot(xarray, yarray, color=color)
    plt.axis('equal')

    return fig, ax


def heatmap(xx, yy, zz):

    if len(xx) >= len(yy):
        X = np.tile(xx, zz.shape[0])
        Y = np.repeat(yy, zz.shape[1])
    else:
        X = np.repeat(xx, zz.shape[1])
        Y = np.tile(yy, zz.shape[0])

    Z = zz.ravel()

    xgrid = np.linspace(0, np.max(xx), 1000, endpoint=True)
    ygrid = np.linspace(0, np.max(yy), 1000, endpoint=True)
    interpolated_values = griddata((X, Y), Z, (xgrid[None, :], ygrid[:, None]), method='cubic')

    levels = np.arange(0, 32, 2)
    cs = plt.contourf(xgrid, ygrid, interpolated_values, levels, cmap=cm.get_cmap('RdYlBu_r', len(levels) - 1), extend='both')
    plt.colorbar(cs, orientation='vertical', shrink=0.5, format='%.2f')


def add_arrows(x, y, directions, lengths, mpl_ax, arrows_scale=1):

    # Make a grid from these coord vectors
    XX, YY = np.meshgrid(x, y)

    angles = np.deg2rad(directions)

    # We convert from polar to cartesian coordinates
    UU = np.multiply(np.cos(angles), lengths)
    VV = np.multiply(np.sin(angles), lengths)

    # If the cell fires in no direction at all on a given position, the mean vector
    # is non existent, so it has no direction and no length, hence the NaNs.
    UU = np.nan_to_num(UU)  # We can simply replace by 0.0 and use minlength=0.0 in the quiver function
    VV = np.nan_to_num(VV)

    mpl_ax.quiver(XX.T, YY.T, UU, VV,
                  units='dots', angles='xy', scale=0.5 * arrows_scale, scale_units='xy',
                  color='k', alpha=0.8, width=1, pivot='tail',
                  headaxislength=4, headlength=4, headwidth=4, minlength=0.0)
                  # TODO: make these values aware of arrows_scale
    return mpl_ax


def mapplot(*args, colorlims=None, colormap='plasma_r', setup_loc=(0.0, 0.0), world=None, mem_coords=None, figsize=(12, 12)):

    if len(args) == 3:
        vals, directions, lengths = args
        is_quiver = True
    elif len(args) == 1:
        vals = args[0]
        is_quiver = False
    else:
        raise AttributeError("You must pass either 1 (map only), or 3 arrays (map + quiver).")

    # ---- Recover viewport coordinates ----

    # Get dimensions (in integer, no units)
    x_len = vals.shape[0]
    y_len = vals.shape[1]
    # And move limits to have the origin in the center
    xlims = np.array([-x_len/2, x_len/2]) + setup_loc[0]
    ylims = np.array([-y_len/2, y_len/2]) + setup_loc[1]

    # Generate X and Y coord vectors.
    # Each pixel's center (and not its bottom left corner) corresponds to a position in the world,
    # so we need to shift the whole generated image 1 unit up, and scale it by 1 pixel
    X = np.arange(xlims[0], xlims[1])
    Y = np.arange(ylims[0], ylims[1]) + 1.0

    plot_lim_X = np.array([X[0], X[-1]])
    plot_lim_Y = np.array([Y[0], Y[-1]])

    img_lim_Xsc = np.array([X[0] - 0.5, X[-1] + 0.5])
    img_lim_Ysc = np.array([Y[0] - 0.5, Y[-1] + 0.5])

    # ---- Make the plot ----

    fig, ax = plt.subplots(figsize=figsize)

    if colorlims is None:
        colorlims = (vals.min(), vals.max())

    # We're dealing with a data array, not an actual image, so as numpy arrays are row-major but images are col-major,
    # we need to transpose it (hence vals.T) and start from the bottom left corner (origin='lower').
    im = ax.imshow(vals.T, origin='lower',
                   vmin=colorlims[0], vmax=colorlims[1],
                   extent=(*img_lim_Xsc, *img_lim_Ysc), aspect='equal',
                   cmap=colormap)

    if is_quiver:
        ax = add_arrows(X, Y, directions, lengths, ax)

    # Add color bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.set_ylabel('Familiarity', rotation=90, va="bottom")

    if world is not None:
        # Subsample the number of vertices to simplify the plot.
        # In order to have a correct idea of the average world elevation, consider an area that is larger
        # than the actually sampled area (for example 150% of the sampled area)
        # (i.e. take a large enough area, but no borders)
        avg_over = 1.5
        vertices = subsample_vertices(world, xlims=plot_lim_X * avg_over, ylims=plot_lim_Y * avg_over, level=10)

        X_elev, Y_elev, Z_elev = np.hsplit(vertices, 3)
        X_elev, Y_elev, Z_elev = X_elev.squeeze(), Y_elev.squeeze(), Z_elev.squeeze()

        ax.tricontour(X_elev, Y_elev, Z_elev, colors='k', alpha=0.2)

    ax.set_xlim(*plot_lim_X)
    ax.set_ylim(*plot_lim_Y)

    if mem_coords is not None:
        ax.scatter(mem_coords[:, 0], mem_coords[:, 1], marker='.', color='w')

    return fig, ax
