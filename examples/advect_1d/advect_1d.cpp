/**
 ==============================================================================
 Copyright 2019, Magdalena Siwek

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


#include <cstdio>
#include "core_ndarray.hpp"
#include <cmath>
#include "core_ndarray_ops.hpp"
#include "core_hdf5.hpp"
#include "app_serialize.hpp"
#include <sstream>
#include <algorithm>
#include "app_config.hpp"
#include <iostream>
#include "app_filesystem.hpp"

double wavespeed = 0;

struct state_t
{
    int iteration;
    double time;
    nd::shared_array<double,1> vertices;
    nd::shared_array<double,1> concentration;

    int diagnostics_count;
    double diagnostics_last;
};


struct solver_data_t
{
    double cfl_number;
    double pulse_width;
    double wavespeed;
};



// ====================Function to calculate midpoints of array cells===========
template<typename ArrayType>
auto midpoint(ArrayType array)
{
    auto L = array | nd::select_axis(0).from(0).to(1).from_the_end();
    auto R = array | nd::select_axis(0).from(1).to(0).from_the_end();
    return (L + R) * 0.5;
}

// ====================Function to calculate array cell sizes===================
template<typename ArrayType>
auto difference(ArrayType array)
{
    auto L = array | nd::select_axis(0).from(0).to(1).from_the_end();
    auto R = array | nd::select_axis(0).from(1).to(0).from_the_end();
    return R - L;
}

auto advance_concentration(const state_t& state, double dt, double a)
{
    // U is the concentration, extend here on each side to later fit the array
    auto U = state.concentration | nd::extend_periodic_on_axis(0);

    // Flux below
    auto F = U * a; //F: flux in each cell
    //now obtain intercell fluxes

    auto FL = F | nd::select_axis(0).from(0).to(1).from_the_end(); //flux on left
    auto FR = F | nd::select_axis(0).from(1).to(0).from_the_end(); //flux on right
    auto Fhat = a < 0 ? FR : FL; // using the “ternary" operator
    auto Fhat_diff = difference(Fhat);
    auto dx = difference(state.vertices);

    return state.concentration - Fhat_diff * dt / dx | nd::to_shared();
}

double wrap(double x)
{
	if (x < -1)
		return wrap(x + 2);
	if (x > +1)
		return wrap(x - 2);
	return x;
}

double gaussian(double x, double sigma)
{
    return std::exp(-(x * x) / sigma / sigma);
}

auto solution_at_time = [] (double t, const solver_data_t& solver_data)
{
    auto a = solver_data.wavespeed;
    auto sigma = solver_data.pulse_width;

    auto evaluate_gaussian_for_t = [a, t, sigma] (auto x)
    {
        return gaussian(wrap(x - a * t), sigma);
    };
    return evaluate_gaussian_for_t;
};

auto compute_l2_error(state_t state, const solver_data_t& solver_data)
{
    //size of cells
    auto dx = difference(state.vertices);

    //midpoints of cells
    auto xc = midpoint(state.vertices);
    auto n = 2; // we're getting thre L2 norm
    auto f = solution_at_time(state.time, solver_data); //comment on this later
    auto F = nd::map(f);
    auto v = F(xc);
    auto u = state.concentration;

    auto integral = (u - v)
    | nd::map([n] (double x) { return std::pow(x, n); })
    | nd::multiply(dx)
    | nd::sum();

    return std::pow(integral, 1.0 / n);
}

solver_data_t create_my_solver_data(mara::config_t run_config)
{
    return {
        run_config.get_double("cfl_number"),
        run_config.get_double("pulse_width"),
        run_config.get_double("wavespeed"),
    };
}

state_t create_my_initial_state(mara::config_t run_config)
{
    auto num_cells = run_config.get_int("resolution");
    auto vertices = nd::linspace(-1, 1, num_cells + 1);
    auto cell_centers = midpoint(vertices); // has size num_cells
    auto concentration = cell_centers | nd::map(solution_at_time(0.0, create_my_solver_data(run_config)));

    return{
        0, // iteration ...
        0.0,
        vertices.shared(),
        concentration.shared(),
        0, // diagnostics_count
        0.0 // diagnostics_last };
};
}

state_t next(const state_t& state, const solver_data_t& solver_data)
{

    //wave speed
    double a = solver_data.wavespeed;
    //CFL number, which is set to ensure cells aren't "emptied"
    //in one timesteps due to poorly chosen dt or grid spacing
    //cfl number is supposed to be 0 < cfl < 1
    double cfl = solver_data.cfl_number;
    auto diff_vert = difference(state.vertices);
    double min_spacing = difference(state.vertices) | nd::min();
    double dt = cfl * min_spacing / a;


    return{
        state.iteration + 1,
        state.time + dt,
        state.vertices,
        //state.concentration
        advance_concentration(state,dt,a),
        state.diagnostics_count,
        state.diagnostics_last,
        //in advance_concentration, put in the physics (M_i = M...)
    };
}

state_t write_diagnostics(const state_t& state, mara::config_t run_config)
{
    double diagnostics_interval =run_config.get_double("delta_t_diagnostic"); //diagnostic-field-interval (dfi)
    auto outdir = run_config.get_string("outdir");

    if (state.time - state.diagnostics_last >= diagnostics_interval - 1e-8)
    {
        auto fname = mara::filesystem::join({outdir, mara::create_numbered_filename("diagnostics", state.diagnostics_count, "h5")});
        auto h5f = h5::File(fname, "w");
        h5f.write("x", midpoint(state.vertices).shared());
        h5f.write("u", state.concentration.shared());
        h5f.write("L2", compute_l2_error(state, create_my_solver_data(run_config)));
        h5f.write("t", state.time);

        auto root = h5f.open_group("/");
        mara::write(root, "run_config", run_config);

        auto next_state = state;
        next_state.diagnostics_count += 1;
        next_state.diagnostics_last += diagnostics_interval;

        std::printf("writing %s\n", fname.data());
        return next_state;
    }

    if (state.time == 0.)
    {
        auto fname = mara::filesystem::join({outdir, mara::create_numbered_filename("diagnostics", state.diagnostics_count, "h5")});
        auto h5f = h5::File(fname, "w");
        h5f.write("x", midpoint(state.vertices).shared());
        h5f.write("u", state.concentration.shared());
        h5f.write("L2", compute_l2_error(state, create_my_solver_data(run_config)));
        h5f.write("t", state.time);

        auto root = h5f.open_group("/");
        mara::write(root, "run_config", run_config);

        auto next_state = state;
        next_state.diagnostics_count += 1;
        next_state.diagnostics_last += diagnostics_interval;

        std::printf("writing %s\n", fname.data());
        return next_state;
    }

    return state;
}


int main(int argc, const char* argv[])
{
    auto cfg_template = mara::make_config_template()
    .item("resolution", 100)
    .item("tfinal", 10.0)
    .item("outdir", "my_data")
    .item("pulse_width", 1.0)
    .item("cfl_number", 1.0)
    .item("wavespeed", 1.0)
    .item("delta_t_diagnostic", 0.2);

    auto run_config = cfg_template.create().update(mara::argv_to_string_map(argc,argv));
    mara::pretty_print(std::cout, "config", run_config);
    mara::filesystem::require_dir(run_config.get_string("outdir"));

    auto solver_data = create_my_solver_data(run_config);
    auto state = create_my_initial_state(run_config);
    state = write_diagnostics(state, run_config);

    while (state.time < run_config.get_double("tfinal"))
    {
        state = write_diagnostics(state, run_config);
        state = next(state, solver_data);

        std::printf("[%04d] t=%lf L2=%lf\n", state.iteration, state.time, compute_l2_error(state, solver_data));
    }

    return 0;
}
