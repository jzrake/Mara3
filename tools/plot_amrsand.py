#!/usr/bin/env python3



import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt



def plot_single_block(ax, h5_verts, h5_values):
    X = h5_verts[...][:,:,0]
    Y = h5_verts[...][:,:,1]
    Z = h5_values[...]
    # Z = X**2 + Y**2
    m0 = ax.pcolormesh(X, Y, Z, edgecolor='none', vmin=0.0, vmax=1.0)



def plot_single_file(filename):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=[15, 8], gridspec_kw={'height_ratios': [1]})
    h5f = h5py.File(filename, 'r')

    for verts, values in zip(h5f['vertices'], h5f['conserved']):
        plot_single_block(axes, h5f['vertices'][verts], h5f['conserved'][values])

    axes.set_aspect('equal')

    return fig



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')

    args = parser.parse_args()

    for filename in args.filenames:
        plot_single_file(filename)

    plt.show()
