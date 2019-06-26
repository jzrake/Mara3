#!/usr/bin/env python3



import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt



def get_ranges(args):
    return dict(
        sigma_range=eval(args.sigma, dict(default=[ -2.0, 0.0])),
        vr_range   =eval(args.vr,    dict(default=[ -0.5, 0.5])),
        vp_range   =eval(args.vp,    dict(default=[  0.0, 2.0])))



def plot_single_block(ax, h5_verts, h5_values, **kwargs):
    X = h5_verts[...][:,:,0]
    Y = h5_verts[...][:,:,1]
    Z = h5_values[...]

    Xb = X[::X.shape[0]//2, ::X.shape[1]//2]
    Yb = Y[::Y.shape[0]//2, ::Y.shape[1]//2]
    Zb = np.zeros_like(Xb + Yb)
    ax.pcolormesh(Xb, Yb, Zb, edgecolor=(1.0, 1.0, 1.0, 0.3))
    m0 = ax.pcolormesh(X, Y, Z, **kwargs)
    return m0



def plot_single_file(
    fig,
    filename,
    depth=8,
    sigma_range=[None, None],
    vr_range=[None, None],
    vp_range=[None, None]):

    axes, cb_axes = fig.subplots(nrows=2, ncols=3, gridspec_kw={'height_ratios': [19, 1]})
    h5f = h5py.File(filename, 'r')

    for block_index in h5f['vertices']:

        if int(block_index[0]) <= depth: continue

        verts = h5f['vertices'][block_index]
        ls = np.log10(h5f['sigma'][block_index])
        vr = h5f['radial_velocity'][block_index]
        vp = h5f['phi_velocity'][block_index]
        m0 = plot_single_block(axes[0], verts, ls, cmap='inferno', vmin=sigma_range[0], vmax=sigma_range[1])
        m1 = plot_single_block(axes[1], verts, vr, cmap='viridis', vmin=vr_range[0], vmax=vr_range[1])
        m2 = plot_single_block(axes[2], verts, vp, cmap='plasma',  vmin=vp_range[0], vmax=vp_range[1])

    for m, cax in zip([m0, m1, m2], cb_axes):
        fig.colorbar(m, cax=cax, orientation='horizontal')

    axes[0].set_title(r'$\log_{10} \Sigma$')
    axes[1].set_title(r'$v_r$')
    axes[2].set_title(r'$v_\phi$')
    axes[0].set_xlabel(r'$x$')
    axes[0].set_ylabel(r'$y$')

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xticks([])
        if ax is not axes[0]:
            ax.set_yticks([])

    return fig



def make_movie(args):
    from matplotlib.animation import FFMpegWriter

    dpi = 200
    res = 768

    writer = FFMpegWriter(fps=10)
    fig = plt.figure(figsize=[15, 8])

    with writer.saving(fig, args.output, dpi):
        for filename in args.filenames:
            print(filename)
            plot_single_file(fig, filename, depth=args.depth, **get_ranges(args))
            writer.grab_frame()
            fig.clf()



def raise_figure_windows(args):
    for filename in args.filenames:
        print(filename)
        fig = plt.figure(figsize=[16, 6])
        plot_single_file(fig, filename, depth=args.depth, **get_ranges(args))
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--movie", action='store_true')
    parser.add_argument("--output", "-o", default="output.mp4")
    parser.add_argument("--sigma", default="default", type=str)
    parser.add_argument("--depth", default=0, type=int)
    parser.add_argument("--vr", default="default", type=str)
    parser.add_argument("--vp", default="default", type=str)

    args = parser.parse_args()

    if args.movie:
        make_movie(args)
    else:
        raise_figure_windows(args)
