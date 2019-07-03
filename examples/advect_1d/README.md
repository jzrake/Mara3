# 1-D ADVECTION PROBLEM

Simulate a Gaussian traveling wave on a 1-D grid. After creating the output files for a range of resolutions, the diffusion of the wave is quantified.

## Getting Started

After compiling Mara3 (check the README in the root directory for instructions), the executable:

examples/advect_1d/advect_1d

should exist. Run a suite of parameter settings by opening a terminal, changing into the root directory (Mara3)
and running:

./tools/run_suite.py examples/advect_1d/vary_resolution.py --submit

This will take a few minutes. Upon completion, this folder should contain .hdf5 files with output at various resolutions:

examples/advect_1d/vary_resolution/res*/diagnostics*.h5

Then open the jupyter notebook: advect_1d.ipynb and run, creating plots.

### Prerequisites

Python 3.5.4 and standard modules.

## Authors

* **Magdalena Siwek**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
