/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

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
#include "app_compile_opts.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




#include <cmath>
#include <iostream>
#include "ndmpi.hpp"
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "app_config.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_subprogram.hpp"
#include "physics_iso2d.hpp"




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)
    .item("tfinal", 1.0)
    .item("N", 256)
    .item("SofteningRadius", 0.1)
    .item("MachNumber", 10.0)
    .item("ViscousAlpha", 0.1)
    .item("BinarySeparation", 1.0)
    .item("CounterRotate", 0);
}

namespace binary
{
    struct solution_state_t
    {
        double time = 0.0;
        int iteration = 0;
        nd::shared_array<mara::unit_length<double>, 1> x_vertices;
        nd::shared_array<mara::unit_length<double>, 1> y_vertices;
        nd::shared_array<mara::iso2d::conserved_t, 2> conserved;
    };
}

using namespace binary;




static auto initial_disk_profile(const mara::config_t& cfg)
{
    return [cfg] (auto x_length, auto y_length)
    {
        auto SofteningRadius  = cfg.get_double("SofteningRadius");
        auto MachNumber       = cfg.get_double("MachNumber");
        auto ViscousAlpha     = cfg.get_double("ViscousAlpha");
        auto BinarySeparation = cfg.get_double("BinarySeparation");
        auto CounterRotate    = cfg.get_int("CounterRotate");
        auto xsi = 10.0;
        auto GM  = 1.0;
        auto x   = x_length.value;
        auto y   = y_length.value;

        // Initial conditions from Tang+ (2017) MNRAS 469, 4258
        // Using time independent potential of single point mass GM
        auto rs             = SofteningRadius;
        auto r0             = BinarySeparation * 2.5;
        auto sigma0         = GM / BinarySeparation / BinarySeparation;
        auto r2             = x * x + y * y;
        auto r              = std::sqrt(r2);
        auto cavity_cutoff  = std::max(std::exp(-std::pow(r / r0, -xsi)), 1e-6);
        auto phi            = -GM * std::pow(r2 + rs * rs, -0.5);    
        auto ag             = -GM * std::pow(r2 + rs * rs, -1.5) * r;    
        auto cs2            = std::pow(MachNumber, -2) * -phi;
        auto cs2_deriv      = std::pow(MachNumber, -2) * ag;
        auto sigma          = sigma0 * std::pow((r + rs) / r0, -0.5) * cavity_cutoff;
        auto sigma_deriv    = sigma0 * std::pow((r + rs) / r0, -1.5) * -0.5 / r0;
        auto dp_dr          = cs2 * sigma_deriv + cs2_deriv * sigma;
        auto omega2         = r < r0 ? GM / (4 * r0) : -ag / r + dp_dr / (sigma * r);        
        auto vq             = (CounterRotate ? -1 : 1) * r * std::sqrt(omega2);
        auto h0             = r / MachNumber;
        auto nu             = ViscousAlpha * std::sqrt(cs2) * h0; // ViscousAlpha * cs * h0
        auto vr             = -(3.0 / 2.0) * nu / (r + rs); // radial drift velocity (CHECK)
        auto vx             = vq * (-y / r) + vr * (x / r);
        auto vy             = vq * ( x / r) + vr * (y / r);

        return mara::iso2d::primitive_t()
        .with_sigma(sigma)
        .with_velocity_x(vx)
        .with_velocity_y(vy);
    };
}




//=============================================================================
template<typename VertexArrayType>
static auto cell_surface_area(const VertexArrayType& xv, const VertexArrayType& yv)
{
    auto dx = xv | nd::difference_on_axis(0);
    auto dy = yv | nd::difference_on_axis(0);
    return nd::cartesian_product(dx, dy) | nd::apply(std::multiplies<>());
}

static auto gravitational_source_term(const solution_state_t& state)
{
    // IMPLEMENT GRAV SOURCE TERMS
    auto xc = state.x_vertices | nd::midpoint_on_axis(0);
    auto yc = state.y_vertices | nd::midpoint_on_axis(0);

    return nd::cartesian_product(xc, yc) | nd::apply([] (auto x, auto y)
    {
        return mara::iso2d::conserved_per_area_t() / mara::make_time(1.0);
    });
}




//=============================================================================
static void write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("x_vertices", state.x_vertices);
    group.write("y_vertices", state.y_vertices);
    group.write("conserved", state.conserved);
}

static auto read_solution(h5::Group&& group)
{
    return solution_state_t(); // IMPLEMENT READING SOLUTION FROM CHECKPOINT
}

static auto new_solution(const mara::config_t& cfg)
{
    auto nx = cfg.get_int("N");
    auto ny = cfg.get_int("N");

    auto xv = nd::linspace(-6, 6, nx + 1) | nd::map(mara::make_length<double>);
    auto yv = nd::linspace(-6, 6, ny + 1) | nd::map(mara::make_length<double>);
    auto xc = xv | nd::midpoint_on_axis(0);
    auto yc = yv | nd::midpoint_on_axis(0);

    auto U = nd::cartesian_product(xc, yc)
    | nd::apply(initial_disk_profile(cfg))
    | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area))
    | nd::multiply(cell_surface_area(xv, yv));

    auto state = solution_state_t();
    state.time = 0.0;
    state.iteration = 0;
    state.x_vertices = xv | nd::to_shared();
    state.y_vertices = yv | nd::to_shared();
    // state.conserved = U | nd::to_shared();
    return state;
}

static auto create_solution(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");
    return restart.empty()
    ? new_solution(run_config)
    : read_solution(h5::File(restart, "r").open_group("solution"));
}

static auto next_solution(const solution_state_t& state)
{
    auto next_state = state;

    auto dA = cell_surface_area(state.x_vertices, state.y_vertices);
    auto u0 = state.conserved;
    auto p0 = u0 / dA | nd::map(mara::iso2d::recover_primitive) | nd::to_shared();

    auto sg = gravitational_source_term(state) * dA;

    // IMPLEMENT INTERCELL FLUXES
    // auto lx = p0 | extend_periodic_on_axis(0) | intercell_flux_on_axis(0) | nd::multiply(dy) | nd::difference_on_axis(0);
    // auto ly = p0 | extend_periodic_on_axis(1) | intercell_flux_on_axis(1) | nd::multiply(dx) | nd::difference_on_axis(1);
    
    // IMPLEMENT TIME STEP CALCULATION
    auto dt = mara::make_time(0.1);

    auto u1 = u0 + (/*lx + ly + */ sg) * dt;

    next_state.iteration += 1;
    next_state.time += dt.value;
    next_state.conserved = u1 | nd::to_shared();
    return next_state;
}




//=============================================================================
static auto new_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create("write_checkpoint");
    schedule.mark_as_due("write_checkpoint");
    return schedule;
}

static auto create_schedule(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");
    return restart.empty()
    ? new_schedule(run_config)
    : mara::read_schedule(h5::File(restart, "r").open_group("schedule"));
}

static auto next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time)
{
    auto next_schedule = schedule;
    auto cpi = run_config.get<double>("cpi");

    if (time - schedule.last_performed("write_checkpoint") >= cpi)
    {
        next_schedule.mark_as_due("write_checkpoint", cpi);
    }
    return next_schedule;
}




//=============================================================================
static auto new_run_config(const mara::config_string_map_t& args)
{
    return config_template().create().update(args);
}

static auto create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);
    return args.count("restart")
    ? config_template()
            .create()
            .update(mara::read_config(h5::File(args.at("restart"), "r").open_group("run_config")))
            .update(args)
    : new_run_config(args);
}




//=============================================================================
struct app_state_t
{
    solution_state_t solution_state;
    mara::schedule_t schedule;
    mara::config_t run_config;
};

static void write_checkpoint(const app_state_t& state)
{
    auto count = state.schedule.num_times_performed("write_checkpoint");
    auto file = h5::File(mara::create_numbered_filename("chkpt", count, "h5"), "w");
    write_solution(file.require_group("solution"), state.solution_state);
    mara::write_schedule(file.require_group("schedule"), state.schedule);
    mara::write_config(file.require_group("run_config"), state.run_config);

    std::printf("write checkpoint: %s\n", file.filename().data());
}

static auto create_app_state(mara::config_t run_config)
{
    auto state = app_state_t();
    state.run_config     = run_config;
    state.solution_state = create_solution(run_config);
    state.schedule       = create_schedule(run_config);
    return state;
}

static auto next(const app_state_t& state)
{
    auto next_state = state;
    next_state.solution_state = next_solution(state.solution_state);
    next_state.schedule       = next_schedule(state.schedule, state.run_config, state.solution_state.time);
    return next_state;
}

static auto simulation_should_continue(const app_state_t& state)
{
    auto time = state.solution_state.time;
    auto tfinal = state.run_config.get<double>("tfinal");
    return time < tfinal;
}

static auto run_tasks(const app_state_t& state)
{
    auto next_state = state;

    if (state.schedule.is_due("write_checkpoint"))
    {
        write_checkpoint(state);
        next_state.schedule.mark_as_completed("write_checkpoint");
    }
    return next_state;
}




//=============================================================================
static void print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.x_vertices.size() * solution.y_vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration, solution.time, kzps);
}




//=============================================================================
class subprog_binary : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        mpi::Session mpi_session;

        mpi::printf_master("initialized on %d mpi processes\n", mpi::comm_world().size());

        auto run_config = create_run_config(argc, argv);
        auto perf = mara::perf_diagnostics_t();
        auto state = create_app_state(run_config);

        mara::pretty_print(mpi::cout_master(), "config", run_config);
        state = run_tasks(state);

        while (simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(mara::compose(run_tasks, next), state);
            print_run_loop_message(state.solution_state, perf);
        }

        run_tasks(next(state));
        return 0;
    }

    std::string name() const override
    {
        return "binary";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_binary()
{
    return std::make_unique<subprog_binary>();
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
