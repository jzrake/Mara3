#!/usr/bin/env python3



import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt



def get_ranges(args):
    return dict(
        sigma_range=eval(args.sigma, dict(auto=[None, None], fiducial=[  0.0, 1.0])),
        vr_range   =eval(args.vr,    dict(auto=[None, None], fiducial=[-1e-2, 0.0])),
        vp_range   =eval(args.vp,    dict(auto=[None, None], fiducial=[  0.0, 2.0])))



def plot_single_file(
    fig,
    filename,
    sigma_range=[None, None],
    vr_range=[None, None],
    vp_range=[None, None]):

    axes, cb_axes = fig.subplots(nrows=2, ncols=3, gridspec_kw={'height_ratios': [19, 1]})
    h5f = h5py.File(filename, 'r')

    vx    = h5f['x_vertices'][...]
    vy    = h5f['y_vertices'][...]
    sigma = h5f['sigma'][...]
    vr    = h5f['radial_velocity'][...]
    vp    = h5f['phi_velocity'][...]

    X, Y = np.meshgrid(vx, vy)

    m0 = axes[0].pcolormesh(Y, X, sigma.T, vmin=sigma_range[0], vmax=sigma_range[1], cmap='inferno')
    m1 = axes[1].pcolormesh(Y, X, vr.T, vmin=vr_range[0], vmax=vr_range[1], cmap='viridis')
    m2 = axes[2].pcolormesh(Y, X, vp.T, vmin=vp_range[0], vmax=vp_range[1], cmap='plasma')

    axes[0].set_title(r'$\Sigma$')
    axes[1].set_title(r'$v_r$')
    axes[2].set_title(r'$v_\phi$')

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



def make_movie(args):
    from matplotlib.animation import FFMpegWriter

    dpi = 200
    res = 768

    writer = FFMpegWriter(fps=10)
    fig = plt.figure(figsize=[15, 8])

    with writer.saving(fig, args.output, dpi):
        for filename in args.filenames:
            print(filename)
            plot_single_file(fig, filename, **get_ranges(args))
            writer.grab_frame()
            fig.clf()



def raise_figure_windows(args):
    for filename in args.filenames:
        print(filename)
        fig = plt.figure(figsize=[15, 8])
        plot_single_file(fig, filename, **get_ranges(args))
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--movie", action='store_true')
    parser.add_argument("--output", default="output.mp4")
    parser.add_argument("--sigma", default="fiducial", type=str)
    parser.add_argument("--vr", default="fiducial", type=str)
    parser.add_argument("--vp", default="fiducial", type=str)

    args = parser.parse_args()

    if args.movie:
        make_movie(args)
    else:
        raise_figure_windows(args)
