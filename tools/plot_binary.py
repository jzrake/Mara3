#!/usr/bin/env python3




import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt




def moving_average(a, window_size=10):
    """
    @brief      Return the moving average of an array, with the given window
                size.
    
    @param      a            The array
    @param      window_size  The window size to use
    
    @return     The window-averaged array
    """
    n = window_size
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n




def plot_moving_average(ax, x, y, window_size=100, avg_only=False, c=None, **kwargs):
    """
    @brief      Wrapper for ax.plot, where the exact values of x and y are
                plotted with a lower alpha, and the moving averages are plotted
    
    @param      ax           The axis instance to plot on
    @param      x            The x values
    @param      y            The y values
    @param      window_size  The window size to use in the moving average
    @param      avg_only     Plot only the moving average if True
    @param      c            The color
    @param      kwargs       Keyword args passed to the plot of moving averages
    
    @return     The result of ax.plot
    """

    if not avg_only:
        ax.plot(x, y, c=c, lw=1.0, alpha=0.5)
    return ax.plot(moving_average(x, window_size), moving_average(y, window_size), **kwargs)




def get_ranges(args):
    """
    @brief      Return the vmin and vmax keywords for fields.
    
    @param      args  The arguments (argparse result)
    
    @return     The vmin/vmax values
    """
    return dict(
        sigma_range=eval(args.sigma, dict(default=[ -2.0, 0.0])),
        vr_range   =eval(args.vr,    dict(default=[ -0.5, 0.5])),
        vp_range   =eval(args.vp,    dict(default=[  0.0, 2.0])))




def plot_single_block(ax, h5_verts, h5_values, edges=False, **kwargs):
    X = h5_verts[...][:,:,0]
    Y = h5_verts[...][:,:,1]
    Z = h5_values[...]

    if edges:
        Xb = X[::X.shape[0]//2, ::X.shape[1]//2]
        Yb = Y[::Y.shape[0]//2, ::Y.shape[1]//2]
        Zb = np.zeros_like(Xb + Yb)
        ax.pcolormesh(Xb, Yb, Zb, edgecolor=(1.0, 0.0, 1.0, 0.3))

    return ax.pcolormesh(X, Y, Z, **kwargs)




def plot_single_file_with_vel(
    fig,
    filename,
    depth=0,
    edges=False,
    sigma_range=[None, None],
    vr_range=[None, None],
    vp_range=[None, None]):

    axes, cb_axes = fig.subplots(nrows=2, ncols=3, gridspec_kw={'height_ratios': [19, 1]})
    h5f = h5py.File(filename, 'r')

    for block_index in h5f['vertices']:

        if int(block_index[0]) < depth: continue

        verts = h5f['vertices'][block_index]
        ls = np.log10(h5f['sigma'][block_index])
        vr = h5f['radial_velocity'][block_index]
        vp = h5f['phi_velocity'][block_index]
        m0 = plot_single_block(axes[0], verts, ls, edges=edges, cmap='inferno', vmin=sigma_range[0], vmax=sigma_range[1])
        m1 = plot_single_block(axes[1], verts, vr, edges=edges, cmap='viridis', vmin=vr_range[0], vmax=vr_range[1])
        m2 = plot_single_block(axes[2], verts, vp, edges=edges, cmap='plasma',  vmin=vp_range[0], vmax=vp_range[1])

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




def plot_single_file_sigma_only(
    fig,
    filename,
    depth=0,
    edges=False,
    sigma_range=[None, None],
    vr_range=[None, None],
    vp_range=[None, None]):

    ax, cax = fig.subplots(nrows=2, ncols=1, gridspec_kw={'height_ratios': [19, 1]})
    h5f = h5py.File(filename, 'r')

    for block_index in h5f['vertices']:

        if int(block_index[0]) < depth:
            continue

        verts = h5f['vertices'][block_index]
        ls = np.log10(h5f['sigma'][block_index])
        m0 = plot_single_block(ax, verts, ls, edges=edges, cmap='inferno', vmin=sigma_range[0], vmax=sigma_range[1])

    fig.colorbar(m0, cax=cax, orientation='horizontal')

    ax.set_title(r'$\log_{10} \Sigma$')
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$y$')
    ax.set_aspect('equal')
    ax.set_xticks([])

    return fig




def make_movie_impl(args, plot_fn, figsize=[16, 6]):
    from matplotlib.animation import FFMpegWriter

    dpi = 200
    res = 768

    writer = FFMpegWriter(fps=10)
    fig = plt.figure(figsize=figsize)

    with writer.saving(fig, args.output, dpi):
        for filename in args.filenames:
            print(filename)
            fig = plot_fn(fig, filename, edges=args.edges, depth=args.depth, **get_ranges(args))
            writer.grab_frame()
            fig.clf()




def raise_figure_windows_impl(args, plot_fn, figsize=[16, 6]):
    for filename in args.filenames:
        print(filename)
        fig = plt.figure(figsize=figsize)
        plot_fn(fig, filename, edges=args.edges, depth=args.depth, **get_ranges(args))
        fig.suptitle(filename)
    plt.show()




def make_movie(args):
    if args.with_vel:
        make_movie_impl(args, plot_single_file_with_vel, figsize=[16, 6])
    else:
        make_movie_impl(args, plot_single_file_sigma_only, figsize=[10, 10])




def raise_figure_windows(args):
    if args.with_vel:
        raise_figure_windows_impl(args, plot_single_file_with_vel, figsize=[16, 6])
    else:
        raise_figure_windows_impl(args, plot_single_file_sigma_only, figsize=[10, 10])




def unzip_time_series(h5_time_series):
    ts = h5_time_series[:]
    return {k:[s[i] for s in ts] for i, k in enumerate(ts.dtype.names)}




def time_series(args):

    fig = plt.figure(figsize=[15, 8])
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(args.filenames)))

    for c, fname in zip(colors, args.filenames):
        h5f = h5py.File(fname, 'r')
        ts = unzip_time_series(h5f['time_series'])

        t  = np.array([s / 2 / np.pi for s in ts['time']])
        Md = np.array([s for s in ts['disk_mass']])
        Me = np.array([s for s in ts['mass_ejected']])
        M1 = np.array([s[0] for s in ts['mass_accreted_on']])
        M2 = np.array([s[1] for s in ts['mass_accreted_on']])

        L1 = np.array([s[0] for s in ts['integrated_torque_on']])
        L2 = np.array([s[1] for s in ts['integrated_torque_on']])
        E1 = np.array([s[0] for s in ts['work_done_on']])
        E2 = np.array([s[1] for s in ts['work_done_on']])

        Mdot1 = np.diff(M1) / np.diff(t)
        Mdot2 = np.diff(M2) / np.diff(t)
        Ldot1 = np.diff(L1) / np.diff(t)
        Ldot2 = np.diff(L2) / np.diff(t)

        Mdot = Mdot1 + Mdot2
        Ldot = Ldot1 + Ldot2
        steady = np.where(t[:-1] > args.saturation_time)

        ax1.plot(t, M1, c='g', lw=1, ls='-',  label=r'$M_1$')
        ax1.plot(t, M2, c='r', lw=2, ls='--', label=r'$M_2$')
        ax1.plot(t, Md, c='g', label=r'$M_{\rm disk}$')
        ax1.plot(t, Me, c='b', label=r'$M_{\rm ejected}$')
        ax1.plot(t, M1 + M2 + Md + Me, c='orange', lw=3, label=r'$M_{\rm tot}$')

        plot_moving_average(ax2, t[:-1], Mdot / Md[:-1], window_size=args.window_size, avg_only=args.avg_only, c=c, lw=2, label=fname)
        plot_moving_average(ax3, t[:-1], Ldot / Mdot,    window_size=args.window_size, avg_only=args.avg_only, c=c, lw=2, label=fname)

        # ax2.axhline(np.mean((Mdot / Md[:-1])[steady]), lw=1.0, c=c, ls='--')
        # ax3.axhline(np.mean((Ldot / Mdot)   [steady]), lw=1.0, c=c, ls='--')
        ax2.axhline(np.mean(Mdot[steady]) / np.mean(Md[:-1][steady]), lw=1.0, c=c, ls='--')
        ax3.axhline(np.mean(Ldot[steady]) / np.mean(Mdot   [steady]), lw=1.0, c=c, ls='--')

        try:
            ax2.axvline(t[steady][0], c='k', ls='--', lw=0.5)
            ax3.axvline(t[steady][0], c='k', ls='--', lw=0.5)
        except:
            print("Warning: no data points are available after the saturation time (try with e.g. --saturation-time=50)")

    # ax1.set_yscale('log')
    ax1.legend()
    ax2.legend()
    ax2.set_ylabel(r'$\dot M / M_{\rm disk}$')
    ax2.set_yscale('log')
    ax3.set_xlabel("Orbits")
    ax3.set_ylabel(r'$\dot L / \dot M$')
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--movie", action='store_true')
    parser.add_argument("--time-series", '-t', action='store_true')
    parser.add_argument("--avg-only", action='store_true')
    parser.add_argument("--saturation-time", type=float, default=150.0)
    parser.add_argument("--window-size", type=int, default=1000)
    parser.add_argument("--with-vel", action='store_true')
    parser.add_argument("--output", "-o", default="output.mp4")
    parser.add_argument("--sigma", default="default", type=str)
    parser.add_argument("--depth", default=0, type=int)
    parser.add_argument("--edges", action='store_true')
    parser.add_argument("--vr", default="default", type=str)
    parser.add_argument("--vp", default="default", type=str)

    args = parser.parse_args()

    if args.time_series:
        time_series(args)
    elif args.movie:
        make_movie(args)
    else:
        raise_figure_windows(args)
