#include <cmath>
#include <iostream>
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "config.hpp"
#include "serialize.hpp"
#include "schedule.hpp"
#include "performance.hpp"




//=============================================================================
struct solution_state_t
{
    double time = 0.0;
    int iteration = 0;
    nd::shared_array<double, 1> vertices;
    nd::shared_array<double, 1> solution;
};




//=============================================================================
solution_state_t new_solution_state(const mara::config_t& cfg)
{
    auto initial_u = [] (auto x) { return std::sin(2 * M_PI * x); };
    auto nx = cfg.template get<int>("N");
    auto xv = nd::linspace(0, 1, nx + 1);
    auto xc = xv | nd::midpoint_on_axis(0);

    solution_state_t state;
    state.vertices = xv.shared();
    state.solution = (xc | nd::transform(initial_u)).shared();

    return state;    
}

solution_state_t create_solution_state(const mara::config_t& cfg)
{
    return new_solution_state(cfg);
}

solution_state_t next_solution(const solution_state_t& state)
{
    auto xv = state.vertices;
    auto u0 = state.solution;

    auto nx = xv.shape(0);
    auto dt = 0.25 / nx;
    auto xc = xv | nd::midpoint_on_axis(0);          // nx
    auto dx = xv | nd::difference_on_axis(0);        // nx
    auto ue = u0 | nd::extend_periodic_on_axis(0);   // nx + 2
    auto fc = ue | nd::intercell_flux_on_axis(0);    // nx + 1
    auto lc = (fc | nd::difference_on_axis(0)) / dx; // nx / nx
    auto u1 = u0 - lc * dt;
    auto t1 = state.time + dt;
    auto i1 = state.iteration + 1;

    return { t1, i1, xv, u1.shared() };
}




//=============================================================================
mara::schedule_t create_schedule(const mara::config_t& run_config)
{
    mara::schedule_t schedule;
    schedule.create("write_checkpoint");
    schedule.mark_as_due("write_checkpoint");
    return schedule;
}

mara::schedule_t next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time)
{
    auto next_schedule = schedule;
    auto cpi = run_config.get<double>("cpi");

    if (time - schedule.last_performed("write_checkpoint") >= cpi)
    {
        next_schedule.mark_as_due("write_checkpoint", cpi);
    }
    return next_schedule;
}

void write_schedule(h5::Group&& group, const mara::schedule_t& schedule)
{
    for (auto task : schedule)
    {
        auto h5_task = group.require_group(task.first);
        h5_task.write("name", task.second.name);
        h5_task.write("num_times_performed", task.second.num_times_performed + 1);
        h5_task.write("last_performed", task.second.last_performed);
    }
}




//=============================================================================
auto read_restart_config(const mara::config_string_map_t& mapping)
{
    return mara::config_parameter_map_t();
}

auto create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);

    return mara::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)
    .item("tfinal", 1.0)
    .item("N", 256)
    .create()
    .update(read_restart_config(args))
    .update(args);
}

void write_config(h5::Group&& group, mara::config_t run_config)
{
    for (auto item : run_config)
    {
        group.write(item.first, item.second);
    }
}




//=============================================================================
struct app_state_t
{
    solution_state_t solution_state;
    mara::schedule_t schedule;
    mara::config_t run_config;
};




//=============================================================================
void write_checkpoint(const app_state_t& state)
{
    char filename[1024];

    std::snprintf(filename, 1024, "chkpt.%04d.h5", state.schedule.num_times_performed("write_checkpoint"));
    std::printf("write checkpoint: %s\n", filename);

    auto file = h5::File(filename, "w");

    file.write("time", state.solution_state.time);
    file.write("vertices", state.solution_state.vertices);
    file.write("solution", state.solution_state.solution);
    write_config(file.require_group("run_config"), state.run_config);
    write_schedule(file.require_group("schedule"), state.schedule);
}

void print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration, solution.time, kzps);
}




//=============================================================================
auto create_app_state(mara::config_t run_config)
{
    auto state = app_state_t();
    state.run_config = run_config;
    state.solution_state = create_solution_state(run_config);
    state.schedule = create_schedule(run_config);
    return state;
}

auto simulation_should_continue(const app_state_t& state)
{
    auto time = state.solution_state.time;
    auto tfinal = state.run_config.get<double>("tfinal");
    return time < tfinal;
}

auto next(const app_state_t& state)
{
    auto next_state = state;

    next_state.solution_state = next_solution(state.solution_state);
    next_state.schedule = next_schedule(state.schedule, state.run_config, state.solution_state.time);

    return next_state;
}

auto run_tasks(const app_state_t& state)
{
    auto next_state = state;

    if (state.schedule.is_due("write_checkpoint"))
    {
        write_checkpoint(state);
        next_state.schedule.mark_as_completed("write_checkpoint");
    }
    return next_state;
}




template<typename F, typename G>
auto compose(F f, G g)
{
    return [f, g] (auto&&... args)
    {
        return f(g(std::forward<decltype(args)>(args)...));
    };
};




//=============================================================================
int main(int argc, const char* argv[])
{
    auto run_config = create_run_config(argc, argv);
    mara::pretty_print(std::cout, "config", run_config);

    auto perf = mara::perf_diagnostics_t();
    auto state = run_tasks(create_app_state(run_config));

    while (simulation_should_continue(state))
    {
        std::tie(state, perf) = mara::time_execution(compose(run_tasks, next), state);
        print_run_loop_message(state.solution_state, perf);
    }

    run_tasks(next(state));
	return 0;
}
