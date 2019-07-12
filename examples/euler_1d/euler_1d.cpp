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
#include "physics_euler.hpp"





/**
 * @brief      A data structure that stores the state of the whole simulation.
 */
struct state_t
{
    int iteration;
    double time;
    nd::shared_array<double,1> vertices;
    nd::shared_array<mara::euler::conserved_density_t, 1> conserved;
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
    double gamma;
};




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
        run_config.get_double("gamma"),
    };
}




/**
 * @brief      Create an initial simulation state.
 *
 * @param[in]  run_config  The run configuration to use
 *
 * @return     The initial state
 */
state_t create_initial_state(mara::config_t run_config, const solver_data_t& solver_data)
{
    auto nc    = run_config.get_int("resolution");
    auto cd    = run_config.get_double("cd");
    auto rhoL  = run_config.get_double("rhoL");
    auto rhoR  = run_config.get_double("rhoR");
    auto pL    = run_config.get_double("pL");
    auto pR    = run_config.get_double("pR");
    auto gamma = run_config.get_double("gamma");

    auto xv = nd::linspace(-1, 1, nc + 1);
    auto xc = xv | nd::midpoint_on_axis(0);

    auto Pl = mara::euler::primitive_t()
    .with_mass_density(rhoL)
    .with_velocity_1(0.0)
    .with_gas_pressure(pL);

    auto Pr = mara::euler::primitive_t()
    .with_mass_density(rhoR)
    .with_velocity_1(0.0)
    .with_gas_pressure(pR);

    auto primitive = xc | nd::map([Pl, Pr, cd] (auto x) { return x < cd ? Pl : Pr; });
    auto conserved = primitive | nd::map([gamma] (auto p) { return p.to_conserved_density(gamma); });

    return{
        0,    // iteration
        0.0,  // time
        xv.shared(),
        conserved.shared(),
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

    double gamma = solver_data.gamma;
    auto nh = mara::unit_vector_t::on_axis(0);

    auto max_wavespeed_of_primitive = [nh, gamma] (auto p)
    {
        return std::max(
            std::abs(p.wavespeeds(nh, gamma).m.value),
            std::abs(p.wavespeeds(nh, gamma).p.value));
    };


    //first recover primitive variables
    auto primitive = state.conserved |
    nd::map([gamma] (auto cons)
    {
        return mara::euler::recover_primitive(cons, gamma, 0);
    });


    //maximum wavespeed in the entire domain
    auto vmax = primitive
    | nd::map(max_wavespeed_of_primitive)
    | nd::max();


    //spacing between cell vertices
    double min_spacing = state.vertices | nd::difference_on_axis(0) | nd::min();


    // CFL number, which is set to ensure cells aren't "emptied" in one
    // timestep due to poorly chosen dt or grid spacing. This parameter must be
    // in the range [0, 1].
    double cfl = solver_data.cfl_number;
    auto dt = mara::make_time(cfl * min_spacing / vmax);
    auto riemann = [gamma,nh] (auto pl, auto pr) { return mara::euler::riemann_hlle(pl, pr, nh, gamma); };
    auto dx = state.vertices | nd::difference_on_axis(0) | nd::map([] (auto x) { return mara::make_length(x); });
    auto Fhat = primitive | nd::extend_zero_gradient(0) | nd::zip_adjacent2_on_axis(0) | nd::apply(riemann);
    auto Fhat_diff = Fhat | nd::difference_on_axis(0);
    auto next_conserved = state.conserved - Fhat_diff * dt / dx;

    return {
        state.iteration + 1,
        state.time + dt.value,
        state.vertices,
        next_conserved.shared(),
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
    auto diagnostics_interval = run_config.get_double("delta_t_diagnostic");
    auto outdir               = run_config.get_string("outdir");


    if (state.time - state.diagnostics_last >= diagnostics_interval || state.time == 0.0)
    {
        auto fname = mara::filesystem::join({outdir, mara::create_numbered_filename("diagnostics", state.diagnostics_count, "h5")});
        auto h5f = h5::File(fname, "w");
        auto root = h5f.open_group("/");

        h5f.write("x", state.vertices | nd::midpoint_on_axis(0) | nd::to_shared());
        h5f.write("t", state.time);
        h5f.write("conserved", state.conserved);
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
    .item("resolution", 1000)
    .item("tfinal", 1.0)
    .item("outdir", "examples/euler_1d/data")
    .item("pulse_width", 1.0)
    .item("cfl_number", 0.1)
    .item("delta_t_diagnostic", 0.01)
    .item("cd", 0.)
    .item("rhoL", 1.)
    .item("rhoR", 0.1)
    .item("pL", 1.)
    .item("pR", 0.125)
    .item("vL", 0.)
    .item("vR", 0.)
    .item("gamma", 1.4);

    auto run_config = cfg_template.create().update(mara::argv_to_string_map(argc,argv));
    mara::pretty_print(std::cout, "config", run_config);
    mara::filesystem::require_dir(run_config.get_string("outdir"));

    auto solver_data = create_solver_data(run_config);
    auto state = create_initial_state(run_config, solver_data);

    while (state.time < run_config.get_double("tfinal"))
    {
        state = write_diagnostics(state, run_config);
        state = next(state, solver_data);
        //std::printf("[%04d] t=%lf\n", state.iteration, state.time);
    }
    write_diagnostics(state, run_config);

    return 0;
}
