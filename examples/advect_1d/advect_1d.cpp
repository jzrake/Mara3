/**
 ==============================================================================
 Copyright 2019, Magdalena Siwek and Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/




#include <algorithm>
#include <cstdio>
#include <cmath>
#include <iostream>
#include <sstream>
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "core_hdf5.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"
#include "app_serialize.hpp"





/**
 * @brief      A data structure that stores the state of the whole simulation.
 */
struct state_t
{
    int iteration;
    double time;

    nd::shared_array<double,1> vertices;
    nd::shared_array<double,1> concentration;

    int diagnostics_count;
    double diagnostics_last;
};




/**
 * @brief      A data structure holding numerical parameters used by the solver.
 */
struct solver_data_t
{
    double cfl_number;
    double pulse_width;
    double wavespeed;
};




/**
 * @brief      Map a value x on the real line to the interval [-1, 1]
 *
 * @param[in]  x     The value to map
 *
 * @return     The mapped value
 */
double wrap(double x)
{
    if (x < -1)
        return wrap(x + 2);
    if (x > +1)
        return wrap(x - 2);
    return x;
}




/**
 * @brief      Return a gaussian function at point x with standard deviation
 *             sigma.
 *
 * @param[in]  x      The point
 * @param[in]  sigma  The standard deviation
 *
 * @return     N(x; sigma)
 */
double gaussian(double x, double sigma)
{
    return std::exp(-x * x / sigma / sigma);
}




/**
 * @brief      Return the average value of adjacent cells
 *
 * @param[in]  array      The 1d array (length N) to average
 *
 * @tparam     ArrayType  The type of the array
 *
 * @return     A new array of length N - 1
 */
template<typename ArrayType>
auto midpoint(ArrayType array)
{
    auto L = array | nd::select_axis(0).from(0).to(1).from_the_end();
    auto R = array | nd::select_axis(0).from(1).to(0).from_the_end();
    return (L + R) * 0.5;
}




/**
 * @brief      Return the adjacent difference of a 1d array
 *
 * @param[in]  array      The 1d array (length N) to difference
 *
 * @tparam     ArrayType  The type of the array
 *
 * @return     A new array of length N - 1
 */
template<typename ArrayType>
auto difference(ArrayType array)
{
    auto L = array | nd::select_axis(0).from(0).to(1).from_the_end();
    auto R = array | nd::select_axis(0).from(1).to(0).from_the_end();
    return R - L;
}




/**
 * @brief      Return an array of concentrations at a time t + dt
 *
 * @param[in]  state  The state at time t
 * @param[in]  dt     The time step size
 * @param[in]  a      The wavespeed
 *
 * @return     The new array of concentrations
 */
auto advance_concentration(const state_t& state, double dt, double a)
{
    // U is the concentration, extended by one zone at each end
    auto U = state.concentration | nd::extend_periodic_on_axis(0);

    // F is the flux in each cell
    auto F = U * a;

    auto FL = F | nd::select_axis(0).from(0).to(1).from_the_end(); // flux on left
    auto FR = F | nd::select_axis(0).from(1).to(0).from_the_end(); // flux on right
    auto Fhat = a < 0.0 ? FR : FL;
    auto Fhat_diff = difference(Fhat);
    auto dx = difference(state.vertices);

    return state.concentration - Fhat_diff * dt / dx | nd::to_shared();
}




/**
 * @brief      Return the exact solution for a traveling pulse at the given
 *             time.
 *
 * @param[in]  t            The time
 * @param[in]  solver_data  The solver data, containing wave speed and pulse
 *                          width
 *
 * @return     An array of concentrations that is the exact solution at time t
 */
auto solution_at_time(double t, const solver_data_t& solver_data)
{
    auto a = solver_data.wavespeed;
    auto sigma = solver_data.pulse_width;

    auto evaluate_gaussian_for_t = [a, t, sigma] (auto x)
    {
        return gaussian(wrap(x - a * t), sigma);
    };
    return evaluate_gaussian_for_t;
};




/**
 * @brief      Calculate the L2 error.
 *
 * @param[in]  state        The simulation state
 * @param[in]  solver_data  The solver data to use
 *
 * @return     The L2 error.
 *
 * @note       https://en.wikipedia.org/wiki/Norm_(mathematics)
 */
auto compute_l2_error(state_t state, const solver_data_t& solver_data)
{
    // Array of cell widths
    auto dx = difference(state.vertices);
    // Array of cell midpoint coordinates
    auto xc = midpoint(state.vertices);

    auto n = 2; // We're getting thre L2 norm
    auto f = solution_at_time(state.time, solver_data);
    auto F = nd::map(f);
    auto v = F(xc);
    auto u = state.concentration;

    auto integral = (u - v)
    | nd::map([n] (double x) { return std::pow(x, n); })
    | nd::multiply(dx)
    | nd::sum();

    return std::pow(integral, 1.0 / n);
}




/**
 * @brief      Create the solver data.
 *
 * @param[in]  run_config  The run configuration to use
 *
 * @return     The solver data
 */
solver_data_t create_solver_data(mara::config_t run_config)
{
    return {
        run_config.get_double("cfl_number"),
        run_config.get_double("pulse_width"),
        run_config.get_double("wavespeed"),
    };
}




/**
 * @brief      Create an initial simulation state.
 *
 * @param[in]  run_config  The run configuration to use
 *
 * @return     The initial state
 */
state_t create_initial_state(mara::config_t run_config)
{
    auto num_cells = run_config.get_int("resolution");
    auto vertices = nd::linspace(-1, 1, num_cells + 1);
    auto cell_centers = midpoint(vertices);
    auto concentration = cell_centers | nd::map(solution_at_time(0.0, create_solver_data(run_config)));

    return{
        0,    // iteration
        0.0,  // time
        vertices.shared(),
        concentration.shared(),
        0,   // diagnostics_count
        0.0  // diagnostics_last
    };
}




/**
 * @brief      Return the next state of the simulation.
 *
 * @param[in]  state        The state at time t
 * @param[in]  solver_data  The solver data to use
 *
 * @return     The state at time t + dt
 */
state_t next(const state_t& state, const solver_data_t& solver_data)
{
    double a = solver_data.wavespeed;


    // CFL number, which is set to ensure cells aren't "emptied" in one
    // timesteps due to poorly chosen dt or grid spacing. This parameter must be
    // in the range [0, 1].
    double cfl = solver_data.cfl_number;
    auto diff_vert = difference(state.vertices);
    double min_spacing = difference(state.vertices) | nd::min();
    double dt = cfl * min_spacing / a;


    return {
        state.iteration + 1,
        state.time + dt,
        state.vertices,
        advance_concentration(state, dt, a),
        state.diagnostics_count,
        state.diagnostics_last,
    };
}




/**
 * @brief      Write diagnostic info to a file.
 *
 * @param[in]  state       The simulation state
 * @param[in]  run_config  The run config
 *
 * @return     A state, updated to reflect that diagnostics were written
 */
state_t write_diagnostics(const state_t& state, mara::config_t run_config)
{
    double diagnostics_interval =run_config.get_double("delta_t_diagnostic");
    auto outdir = run_config.get_string("outdir");

    if (state.time - state.diagnostics_last >= diagnostics_interval || state.time == 0.0)
    {
        auto fname = mara::filesystem::join({outdir, mara::create_numbered_filename("diagnostics", state.diagnostics_count, "h5")});
        auto h5f = h5::File(fname, "w");
        h5f.write("x", midpoint(state.vertices).shared());
        h5f.write("u", state.concentration.shared());
        h5f.write("t", state.time);
        h5f.write("L2", compute_l2_error(state, create_solver_data(run_config)));

        auto root = h5f.open_group("/");
        mara::write(root, "run_config", run_config);

        auto next_state = state;
        next_state.diagnostics_count += 1;
        next_state.diagnostics_last += state.time == 0.0 ? 0.0 : diagnostics_interval;

        std::printf("writing %s\n", fname.data());
        return next_state;
    }
    return state;
}




/**
 * @brief      The main function
 *
 * @param[in]  argc  The number of arguments from the command line
 * @param      argv  An array of strings with the executable invocation
 *
 * @return     An exit status
 */
int main(int argc, const char* argv[])
{
    auto cfg_template = mara::make_config_template()
    .item("resolution", 100)
    .item("tfinal", 10.0)
    .item("outdir", "data")
    .item("pulse_width", 1.0)
    .item("cfl_number", 0.8)
    .item("wavespeed", 1.0)
    .item("delta_t_diagnostic", 0.2);

    auto run_config = cfg_template.create().update(mara::argv_to_string_map(argc,argv));
    mara::pretty_print(std::cout, "config", run_config);
    mara::filesystem::require_dir(run_config.get_string("outdir"));

    auto solver_data = create_solver_data(run_config);
    auto state = create_initial_state(run_config);

    while (state.time < run_config.get_double("tfinal"))
    {
        state = write_diagnostics(state, run_config);
        state = next(state, solver_data);

        std::printf("[%04d] t=%lf L2=%lf\n", state.iteration, state.time, compute_l2_error(state, solver_data));
    }
    write_diagnostics(state, run_config);

    return 0;
}
