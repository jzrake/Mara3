#!/usr/bin/env python3



import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt



def plot_single_file(filename):
    fig, (axes, cb_axes) = plt.subplots(nrows=2, ncols=3, figsize=[15, 8], gridspec_kw={'height_ratios': [19, 1]})
    h5f = h5py.File(filename, 'r')

    vx    = h5f['x_vertices'][...]
    vy    = h5f['y_vertices'][...]
    sigma = h5f['sigma'][...]
    vr    = h5f['radial_velocity'][...]
    vp    = h5f['phi_velocity'][...]

    X, Y = np.meshgrid(vx, vy)

    m0 = axes[0].pcolormesh(Y, X, sigma.T)
    m1 = axes[1].pcolormesh(Y, X, vr.T)
    m2 = axes[2].pcolormesh(Y, X, vp.T)

    axes[0].set_title(r'$\Sigma$')
    axes[1].set_title(r'$v_r$')
    axes[2].set_title(r'$v_\theta$')

    for m, cax in zip([m0, m1, m2], cb_axes):
        fig.colorbar(m, cax=cax, orientation='horizontal')

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        if ax is not axes[0]:
            ax.set_yticks([])

    axes[0].set_ylabel(r'$\log_{10}(r)$')

    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.1, hspace=0.0)
    fig.suptitle(filename)

    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')

    args = parser.parse_args()

    for filename in args.filenames:
        plot_single_file(filename)

    plt.show()
