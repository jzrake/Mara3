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
        sigma_range=eval(args.sigma, dict(default=[ -6.5,-4.5])),
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
    fig = plt.figure(figsize=[15, 9])
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    colors = plt.cm.viridis(np.linspace(0.3, 0.7, len(args.filenames)))

    for c, fname in zip(colors, args.filenames):
        h5f = h5py.File(fname, 'r')
        ts = unzip_time_series(h5f['time_series'])

        t  = np.array([s / 2 / np.pi for s in ts['time']])
        Md = np.array([s for s in ts['disk_mass']])
        Me = np.array([s for s in ts['mass_ejected']])
        M1 = np.array([s[0] for s in ts['mass_accreted_on']])
        M2 = np.array([s[1] for s in ts['mass_accreted_on']])

        Ld = np.array([s for s in ts['disk_angular_momentum']])
        Le = np.array([s for s in ts['angular_momentum_ejected']])
        L1 = np.array([s[0] for s in ts['integrated_torque_on']])
        L2 = np.array([s[1] for s in ts['integrated_torque_on']])
        K1 = np.array([s[0] for s in ts['angular_momentum_accreted_on']])
        K2 = np.array([s[1] for s in ts['angular_momentum_accreted_on']])

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
        ax1.plot(t, Me, c='b', label=r'$\Delta M_{\rm buffer}$')
    
        if args.show_total:
            ax1.plot(t, Md, c='g', label=r'$M_{\rm disk}$')
            ax1.plot(t, M1 + M2 + Md + Me, c='orange', lw=3, label=r'$M_{\rm tot}$')
        else:
            ax1.plot(t, Md - Md[0], c='g', label=r'$\Delta M_{\rm disk}$')

        ax2.plot(t, L1, c='g', lw=2, ls='-',  label=r'$L_{\rm grav, 1}$')
        ax2.plot(t, L2, c='r', lw=2, ls='-',  label=r'$L_{\rm grav, 2}$')
        ax2.plot(t, K1, c='g', lw=1, ls='--', label=r'$L_{\rm acc, 1}$')
        ax2.plot(t, K2, c='r', lw=1, ls='--', label=r'$L_{\rm acc, 2}$')
        ax2.plot(t, Le, c='b', label=r'$\Delta L_{\rm buffer}$')

        if args.show_total:
            ax2.plot(t, Ld, c='g', label=r'$L_{\rm disk}$')
            ax2.plot(t, L1 + L2 + K1 + K2 + Ld + Le, c='orange', lw=3, label=r'$L_{\rm tot}$')
        else:
            ax2.plot(t, Ld - Ld[0], c='g', label=r'$\Delta L_{\rm disk}$')

        plot_moving_average(ax3, t[:-1], Mdot / Md[:-1], window_size=args.window_size, avg_only=args.avg_only, c=c, lw=2, label=fname)
        plot_moving_average(ax4, t[:-1], Ldot / Mdot,    window_size=args.window_size, avg_only=args.avg_only, c=c, lw=2, label=fname)

        # ax2.axhline(np.mean((Mdot / Md[:-1])[steady]), lw=1.0, c=c, ls='--')
        # ax3.axhline(np.mean((Ldot / Mdot)   [steady]), lw=1.0, c=c, ls='--')
        ax3.axhline(np.mean(Mdot[steady]) / np.mean(Md[:-1][steady]), lw=1.0, c=c, ls='--')
        ax4.axhline(np.mean(Ldot[steady]) / np.mean(Mdot   [steady]), lw=1.0, c=c, ls='--')

        try:
            ax3.axvline(t[steady][0], c='k', ls='--', lw=0.5)
            ax4.axvline(t[steady][0], c='k', ls='--', lw=0.5)
        except:
            print("Warning: no data points are available after the saturation time (try with e.g. --saturation-time=50)")

    # ax1.set_yscale('log')
    ax1.legend()
    ax2.legend()
    ax3.set_ylabel(r'$\dot M / M_{\rm disk}$')
    ax3.set_yscale('log')
    ax4.set_xlabel("Orbits")
    ax4.set_ylabel(r'$\dot L / \dot M$')
    plt.show()




def time_series_specific_torques(args):
    import numpy as np
    import h5py
    import matplotlib.pyplot as plt


    def smooth(f):
        return np.array([np.mean(f[i0-1000:i0+1000]) for i0 in range(1000, len(f) - 1000)])


    fig = plt.figure(figsize=[15, 9])
    ax1 = fig.add_subplot(1, 1, 1)


    for filename in args.filenames:
        h5f = h5py.File(filename, 'r')
        q = h5f['run_config']['mass_ratio'][()]
        e = h5f['run_config']['eccentricity'][()]

        if e != 0.0:
           raise NotImplementedError("specific torque calculation not implemented for non-zero eccentricity")

        a2 = 1 / (1 + q)
        a1 = 1 - a2
        M2 = q / (1 + q)
        M1 = 1 - M2
        L1 = M1 * a1**2
        L2 = M2 * a2**2

        time = h5f['time_series']['time']
        La1 = L1 + h5f['time_series']['angular_momentum_accreted_on'][:,0]
        La2 = L2 + h5f['time_series']['angular_momentum_accreted_on'][:,1]
        Lg1 = L1 + h5f['time_series']['integrated_torque_on'][:,0]
        Lg2 = L2 + h5f['time_series']['integrated_torque_on'][:,1]
        Ma1 = M1 + h5f['time_series']['mass_accreted_on'][:,0]
        Ma2 = M2 + h5f['time_series']['mass_accreted_on'][:,1]
        Mg1 = M1 + np.zeros_like(Ma1)
        Mg2 = M2 + np.zeros_like(Ma2)

        delta_l_grav_1 = (np.diff(Lg1) * Mg1[1:] - Lg1[1:] * np.diff(Mg1)) / Mg1[1:]**2
        delta_l_grav_2 = (np.diff(Lg2) * Mg2[1:] - Lg2[1:] * np.diff(Mg2)) / Mg2[1:]**2
        delta_l_accr_1 = (np.diff(La1) * Ma1[1:] - La1[1:] * np.diff(Ma1)) / Ma1[1:]**2
        delta_l_accr_2 = (np.diff(La2) * Ma2[1:] - La2[1:] * np.diff(Ma2)) / Ma2[1:]**2
        delta_M = np.diff(Ma1 + Ma2 + Mg1 + Mg2)

        orbits = time[1:] / 2 / np.pi

        sat = np.where(orbits > 150)
        plot_moving_average(ax1, orbits, delta_l_grav_1 / delta_M, window_size=args.window_size, avg_only=True, label='Grav 1 (average = {:.3f})'.format(np.mean(delta_l_grav_1[sat] / delta_M[sat])))
        plot_moving_average(ax1, orbits, delta_l_grav_2 / delta_M, window_size=args.window_size, avg_only=True, label='Grav 2 (average = {:.3f})'.format(np.mean(delta_l_grav_2[sat] / delta_M[sat])))
        plot_moving_average(ax1, orbits, delta_l_accr_1 / delta_M, window_size=args.window_size, avg_only=True, label='Accr 1 (average = {:.3f})'.format(np.mean(delta_l_accr_1[sat] / delta_M[sat])))
        plot_moving_average(ax1, orbits, delta_l_accr_2 / delta_M, window_size=args.window_size, avg_only=True, label='Accr 2 (average = {:.3f})'.format(np.mean(delta_l_accr_2[sat] / delta_M[sat])))

    ax1.set_xlabel("Orbits")
    ax1.set_ylabel(r'Specific torque per accreted mass $dl / dM \ (\Omega a^2 \dot{M} / M$')
    ax1.legend()
    plt.show()




def time_series_orbital_elements(args):
    fname = args.filenames[0]
    h5f = h5py.File(fname, 'r')
    ts = unzip_time_series(h5f['time_series'])


    fig = plt.figure(figsize=[15, 9])
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    orbits = h5f['time_series']['time'] / 2 / np.pi
    a_acc  = h5f['time_series']['orbital_elements_acc' ]['elements']['separation']
    a_grav = h5f['time_series']['orbital_elements_grav']['elements']['separation']
    e_acc  = h5f['time_series']['orbital_elements_acc' ]['elements']['eccentricity']
    e_grav = h5f['time_series']['orbital_elements_grav']['elements']['eccentricity']
    M_disk = h5f['time_series']['disk_mass']

    # dx_acc  = h5f['time_series']['orbital_elements_acc' ]['cm_position_x']
    # dx_grav = h5f['time_series']['orbital_elements_grav']['cm_position_x']

    ax1.plot(orbits, a_acc  / M_disk * M_disk[0], label='Accretion')
    ax1.plot(orbits, a_grav / M_disk * M_disk[0], label='Gravitational')
    ax2.plot(orbits, e_acc  / M_disk * M_disk[0], label='Accretion')
    ax2.plot(orbits, e_grav / M_disk * M_disk[0], label='Gravitational')
    # ax2.plot(orbits, 0.0008 * orbits**0.5, c='k', ls='--')
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    ax1.set_ylabel(r'Separation')
    ax2.set_ylabel(r'Eccentricity')
    ax2.set_xlabel("Orbits")
    ax1.legend()
    plt.show()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("filenames", nargs='+')
    parser.add_argument("--movie", action='store_true')
    parser.add_argument("--time-series", '-t', action='store_true')
    parser.add_argument("--orbital-elements", '-e', action='store_true')
    parser.add_argument("--specific-torques", '-s', action='store_true')
    parser.add_argument("--avg-only", action='store_true')
    parser.add_argument("--show-total", action='store_true')
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
    elif args.orbital_elements:
        time_series_orbital_elements(args)
    elif args.specific_torques:
        time_series_specific_torques(args)
    elif args.movie:
        make_movie(args)
    else:
        raise_figure_windows(args)
