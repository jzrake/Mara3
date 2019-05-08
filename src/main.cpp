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
auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)
    .item("tfinal", 1.0)
    .item("N", 256);
}




//=============================================================================
struct solution_state_t
{
    double time = 0.0;
    int iteration = 0;
    nd::shared_array<double, 1> vertices;
    nd::shared_array<double, 1> solution;
};

void write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("vertices", state.vertices);
    group.write("solution", state.solution);
}

auto read_solution(h5::Group&& group)
{
    auto state = solution_state_t();
    state.time = group.read<double>("time");
    state.iteration = group.read<int>("iteration");
    state.vertices = group.read<nd::unique_array<double, 1>>("vertices").shared();
    state.solution = group.read<nd::unique_array<double, 1>>("solution").shared();
    return state;
}

auto new_solution(const mara::config_t& cfg)
{
    auto initial_u = [] (auto x) { return std::sin(2 * M_PI * x); };
    auto nx = cfg.template get<int>("N");
    auto xv = nd::linspace(0, 1, nx + 1);
    auto xc = xv | nd::midpoint_on_axis(0);
    auto state = solution_state_t();

    state.vertices = xv.shared();
    state.solution = (xc | nd::transform(initial_u)).shared();

    return state;    
}

auto create_solution(const mara::config_t& run_config)
{
    if (! run_config.get<std::string>("restart").empty())
    {
        auto file = h5::File(run_config.get<std::string>("restart"), "r");
        return read_solution(file.open_group("solution"));
    }
    return new_solution(run_config);
}

auto next_solution(const solution_state_t& state)
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
    return solution_state_t { t1, i1, xv, u1.shared() };
}




//=============================================================================
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

auto read_schedule(h5::Group&& group)
{
    auto schedule = mara::schedule_t();

    for (auto task_name : group)
    {
        auto task = mara::schedule_t::task_t();
        auto h5_task = group.open_group(task_name);
        task.name = task_name;
        task.num_times_performed = h5_task.read<int>("num_times_performed");
        task.last_performed = h5_task.read<double>("last_performed");
        schedule.insert(task);
    }
    return schedule;
}

auto new_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create("write_checkpoint");
    schedule.mark_as_due("write_checkpoint");
    return schedule;
}

auto create_schedule(const mara::config_t& run_config)
{
    if (! run_config.get<std::string>("restart").empty())
    {
        auto file = h5::File(run_config.get<std::string>("restart"), "r");
        return read_schedule(file.open_group("schedule"));
    }
    return new_schedule(run_config);
}

auto next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time)
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
void write_config(h5::Group&& group, mara::config_t run_config)
{
    for (auto item : run_config)
    {
        group.write(item.first, item.second);
    }
}

auto read_config(h5::Group&& group)
{
    auto config = mara::config_parameter_map_t();

    for (auto item_name : group)
    {
        config[item_name] = group.read<mara::config_parameter_t>(item_name);
    }
    return config;
}

auto new_run_config(const mara::config_string_map_t& args)
{
    return config_template().create().update(args);
}

auto create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);

    if (args.count("restart"))
    {
        auto file = h5::File(args.at("restart"), "r");

        return config_template()
        .create()
        .update(read_config(file.open_group("run_config")))
        .update(args);
    }
    return new_run_config(args);
}




//=============================================================================
struct app_state_t
{
    solution_state_t solution_state;
    mara::schedule_t schedule;
    mara::config_t run_config;
};

void write_checkpoint(const app_state_t& state)
{
    char filename[1024];

    std::snprintf(filename, 1024, "chkpt.%04d.h5", state.schedule.num_times_performed("write_checkpoint"));
    std::printf("write checkpoint: %s\n", filename);

    auto file = h5::File(filename, "w");
    write_solution(file.require_group("solution"), state.solution_state);
    write_schedule(file.require_group("schedule"), state.schedule);
    write_config(file.require_group("run_config"), state.run_config);
}

auto create_app_state(mara::config_t run_config)
{
    auto state = app_state_t();
    state.run_config = run_config;
    state.solution_state = create_solution(run_config);
    state.schedule = create_schedule(run_config);
    return state;
}

auto next(const app_state_t& state)
{
    auto next_state = state;
    next_state.solution_state = next_solution(state.solution_state);
    next_state.schedule = next_schedule(state.schedule, state.run_config, state.solution_state.time);
    return next_state;
}

auto simulation_should_continue(const app_state_t& state)
{
    auto time = state.solution_state.time;
    auto tfinal = state.run_config.get<double>("tfinal");
    return time < tfinal;
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




//=============================================================================
template<typename F, typename G>
auto compose(F f, G g)
{
    return [f, g] (auto&&... args)
    {
        return f(g(std::forward<decltype(args)>(args)...));
    };
};

void print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration, solution.time, kzps);
}




//=============================================================================
int main(int argc, const char* argv[])
{
    auto run_config = create_run_config(argc, argv);
    auto perf = mara::perf_diagnostics_t();
    auto state = create_app_state(run_config);

    mara::pretty_print(std::cout, "config", run_config);
    state = run_tasks(state);

    while (simulation_should_continue(state))
    {
        std::tie(state, perf) = mara::time_execution(compose(run_tasks, next), state);
        print_run_loop_message(state.solution_state, perf);
    }

    run_tasks(next(state));
	return 0;
}
