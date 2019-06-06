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
#if MARA_COMPILE_SUBPROGRAM_BOILERPLATE




#include <cmath>
#include <iostream>
#include "core_mpi.hpp"
#include "core_hdf5.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "app_config.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_subprogram.hpp"




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)
    .item("tfinal", 1.0)
    .item("N", 256);
}

namespace boilerplate
{
    struct solution_state_t
    {
        double time = 0.0;
        int iteration = 0;
        nd::shared_array<double, 1> vertices;
        nd::shared_array<double, 1> solution;
    };

    auto intercell_flux_on_axis(std::size_t axis)
    {
        return [axis] (auto array)
        {
            return array | nd::select_axis(axis).from(0).to(1).from_the_end();        
        };
    }
}

using namespace boilerplate;




//=============================================================================
static void write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("vertices", state.vertices);
    group.write("solution", state.solution);
}

static auto read_solution(h5::Group&& group)
{
    auto state = solution_state_t();
    state.time = group.read<double>("time");
    state.iteration = group.read<int>("iteration");
    state.vertices = group.read<nd::unique_array<double, 1>>("vertices").shared();
    state.solution = group.read<nd::unique_array<double, 1>>("solution").shared();
    return state;
}

static auto new_solution(const mara::config_t& cfg)
{
    auto initial_u = [] (auto x) { return std::sin(2 * M_PI * x); };
    auto nx = cfg.template get<int>("N");
    auto xv = nd::linspace(0.0, 1.0, nx + 1);
    auto xc = xv | nd::midpoint_on_axis(0);
    auto state = solution_state_t();

    state.vertices = xv.shared();
    state.solution = (xc | nd::map(initial_u)).shared();

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
    auto xv = state.vertices;
    auto u0 = state.solution;
    auto nx = xv.shape(0);
    auto dt = 0.25 / nx;
    auto xc = xv | nd::midpoint_on_axis(0);          // nx
    auto dx = xv | nd::difference_on_axis(0);        // nx
    auto ue = u0 | nd::extend_periodic_on_axis(0);   // nx + 2
    auto fc = ue | intercell_flux_on_axis(0);        // nx + 1
    auto lc = (fc | nd::difference_on_axis(0)) / dx; // nx / nx
    auto u1 = u0 - lc * dt;
    auto t1 = state.time + dt;
    auto i1 = state.iteration + 1;
    return solution_state_t { t1, i1, xv, u1.shared() };
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
    auto kzps = solution.vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration, solution.time, kzps);
}




//=============================================================================
class subprog_boilerlate : public mara::sub_program_t
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
        return "boilerplate";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_boilerlate()
{
    return std::make_unique<subprog_boilerlate>();
}

#endif // MARA_COMPILE_SUBPROGRAM_BOILERPLATE
