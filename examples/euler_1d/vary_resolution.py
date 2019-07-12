suite = {
    'subprog': '',
    'comment': 'Variation of resolution',
    'exe': 'examples/euler_1d/euler_1d',
    'job_params': dict(hours=12, nodes=1),
    'mara_opts': dict(
        tfinal=0.5,
        delta_t_diagnostic=0.01,
        cfl_number=0.1,
        cd = 0,
        rhoL = 1.,
        rhoR = 0.1,
        pL = 1.,
        pR = 0.125),
     'runs': {
        'res_00050': dict(resolution=50),
        'res_00100': dict(resolution=100),
        'res_00200': dict(resolution=200),
        'res_00400': dict(resolution=400),
        'res_00800': dict(resolution=800),
        'res_01600': dict(resolution=1600),
    },
}
