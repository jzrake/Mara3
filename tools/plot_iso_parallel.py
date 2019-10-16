#!/usr/bin/env python3




import numpy as np
import h5py
import matplotlib.pyplot as plt




def plot_single_block(ax, h5_verts, h5_values, edges=False, **kwargs):
    X = h5_verts[...][:,:,0]
    Y = h5_verts[...][:,:,1]
    Z = h5_values[...]

    if edges:
        Xb = X[::X.shape[0]//2, ::X.shape[1]//2]
        Yb = Y[::Y.shape[0]//2, ::Y.shape[1]//2]
        Zb = np.zeros_like(Xb + Yb)
        ax.pcolormesh(Xb, Yb, Zb, edgecolor=(1.0, 0.5, 0.5, 1.0))

    return ax.pcolormesh(X, Y, Z, **kwargs)




def plot_single_file(
    fig,
    filename,
    depth=0,
    edges=False):

    ax, cax = fig.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [19, 1]})
    h5f = h5py.File(filename, 'r')

    for block_index in h5f['vertices']:

        if int(block_index[0]) < depth:
            continue

        verts = h5f['vertices'][block_index]

        if len(verts) == 0:
            continue

        ls = h5f['conserved'][block_index][:,:][:,:,0]
        m0 = plot_single_block(ax, verts, ls, edges=edges, cmap='inferno', vmin=0.0, vmax=1.0)

    fig.colorbar(m0, cax=cax, orientation='horizontal')

    ax.set_title(r'$\rho$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')

    return fig




if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    args = parser.parse_args()

    for filename in args.filenames:
        fig = plt.figure()
        plot_single_file(fig, filename, edges=True)

    plt.show()
