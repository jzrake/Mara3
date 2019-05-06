#include <cmath>
#include <iostream>
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "config.hpp"
#include "serialize.hpp"




//=============================================================================
auto linspace(double x0, double x1, std::size_t count)
{
    auto mapping = [x0, x1, count] (auto index)
    {
        return x0 + (x1 - x0) * index[0] / (count - 1);
    };
    return make_array(mapping, nd::make_shape(count));
}

auto midpoint_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return (
        (array | nd::select_axis(axis).from(0).to(1).from_the_end()) +
        (array | nd::select_axis(axis).from(1).to(0).from_the_end())) * 0.5;
    };
}

auto select_first(std::size_t count, std::size_t axis)
{
    return [count, axis] (auto array)
    {
        auto shape = array.shape();
        auto start = nd::make_uniform_index<shape.size()>(0);
        auto final = shape.last_index();

        final[axis] = start[axis] + count;

        return array | nd::select(nd::make_access_pattern(shape).with_start(start).with_final(final));
    };
}

auto select_final(std::size_t count, std::size_t axis)
{
    return [count, axis] (auto array)
    {
        auto shape = array.shape();
        auto start = nd::make_uniform_index<shape.size()>(0);
        auto final = shape.last_index();

        start[axis] = final[axis] - count;

        return array | nd::select(nd::make_access_pattern(shape).with_start(start).with_final(final));
    };
}

auto difference_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return (
        (array | nd::select_axis(axis).from(1).to(0).from_the_end()) -
        (array | nd::select_axis(axis).from(0).to(1).from_the_end()));
    };
}

auto intercell_flux_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return array | nd::select_axis(axis).from(0).to(1).from_the_end();        
    };
}

auto extend_periodic_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        auto xl = array | select_first(1, 0);
        auto xr = array | select_final(1, 0);
        return xr | nd::concat(array).on_axis(axis) | nd::concat(xl).on_axis(axis);
    };
}




//=============================================================================
struct task_schedule_t
{
public:

    void create(std::string name)
    {
        task_counts[name] = 0;
        task_last_performed[name] = 0.0;
    }

    int count(std::string task_name) const
    {
        return task_counts.at(task_name);
    }

    double last_performed(std::string task_name) const
    {
        return task_last_performed.at(task_name);
    }

    auto increment(std::string task_name, double interval=0) const
    {
        auto result = *this;
        result.task_counts.at(task_name) += 1;
        result.task_last_performed.at(task_name) += interval;
        return result;
    }

private:
    std::map<std::string, int> task_counts;
    std::map<std::string, double> task_last_performed;
};




//=============================================================================
struct solver_diagnostics_t
{
    solver_diagnostics_t() {}

    template<typename SolverDuration>
    solver_diagnostics_t(SolverDuration duration)
    {
        execution_time_ms = 1e-6 * std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    }
    double execution_time_ms = 0;
};




template<typename Function, typename... Args>
auto time_execution(Function&& func, Args&&... args)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto result = std::forward<Function>(func)(std::forward<Args>(args)...);
    auto stop = std::chrono::high_resolution_clock::now();
    return std::make_pair(std::move(result), solver_diagnostics_t(stop - start));
};




//=============================================================================
struct state_t
{
    double time = 0.0;
    int iteration = 0;
    nd::shared_array<double, 1> vertices;
    nd::shared_array<double, 1> solution;

    task_schedule_t schedule;
};




//=============================================================================
template<typename Config>
state_t initial_state(Config cfg)
{
    auto initial_u = [] (auto x) { return std::sin(2 * M_PI * x); };
    auto nx = cfg.template get<int>("N");
    auto xv = linspace(0, 1, nx + 1);
    auto xc = xv | midpoint_on_axis(0);

    state_t state;
    state.vertices = xv.shared();
    state.solution = (xc | nd::transform(initial_u)).shared();

    state.schedule.create("write_checkpoint");
    state.schedule.create("print_run_loop_message");

    return state;
}

state_t next(const state_t& state)
{
    auto xv = state.vertices;
    auto u0 = state.solution;

    auto nx = xv.shape(0);
    auto dt = 0.25 / nx;
    auto xc = xv | midpoint_on_axis(0);          // nx
    auto dx = xv | difference_on_axis(0);        // nx
    auto ue = u0 | extend_periodic_on_axis(0);   // nx + 2
    auto fc = ue | intercell_flux_on_axis(0);    // nx + 1
    auto lc = (fc | difference_on_axis(0)) / dx; // nx / nx
    auto u1 = u0 - lc * dt;
    auto t1 = state.time + dt;
    auto i1 = state.iteration + 1;

    return { t1, i1, xv, u1.shared(), state.schedule };
}




//=============================================================================
auto write_checkpoint(const state_t& state, solver_diagnostics_t diagnostics)
{
    double cpi = 0.01;

    if (state.time - state.schedule.last_performed("write_checkpoint") > cpi)
    {
        char filename[1024];

        std::snprintf(filename, 1024, "chkpt.%04d.h5", state.schedule.count("write_checkpoint"));
        std::printf("write checkpoint: %s\n", filename);

        auto file = h5::File(filename, "w");

        file.write("time", state.time);
        file.write("vertices", state.vertices);
        file.write("solution", state.solution);

        return state.schedule.increment("write_checkpoint", cpi);
    }
    return state.schedule;
}

auto print_run_loop_message(const state_t& state, solver_diagnostics_t diagnostics)
{
    std::printf("[%04d] t=%3.3lf kzps=%3.2lf\n", state.iteration, state.time, state.vertices.size() / diagnostics.execution_time_ms);
    return state.schedule.increment("print_run_loop_message");
}




//=============================================================================
template<typename StringMapping>
auto read_restart_config(const StringMapping& mapping)
{
    return config::parameter_map_t();
}




//=============================================================================
int main(int argc, const char* argv[])
{
    auto config_template = config::make_config_template()
    .item("tfinal", 1.0)
    .item("N", 256);

    auto args = config::argv_to_string_map(argc, argv);
    auto run_config = config_template.create().update(read_restart_config(args)).update(args);

    config::pretty_print(std::cout, "config", run_config);

    auto exec_diagnostics = solver_diagnostics_t();
    auto state = initial_state(run_config);
    state.schedule = write_checkpoint(state, exec_diagnostics);
    state.schedule = print_run_loop_message(state, exec_diagnostics);


    while (state.time < run_config.get<double>("tfinal"))
    {
        std::tie(state, exec_diagnostics) = time_execution(next, state);
        state.schedule = write_checkpoint(state, exec_diagnostics);
        state.schedule = print_run_loop_message(state, exec_diagnostics);
    }

	return 0;
}
