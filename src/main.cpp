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
struct state_t
{
    double time = 0.0;
    int iteration = 0;
    nd::shared_array<double, 1> vertices;
    nd::shared_array<double, 1> solution;
    mara::task_schedule_t schedule;
};




//=============================================================================
state_t initial_state(const config::config_t& cfg)
{
    auto initial_u = [] (auto x) { return std::sin(2 * M_PI * x); };
    auto nx = cfg.template get<int>("N");
    auto xv = nd::linspace(0, 1, nx + 1);
    auto xc = xv | nd::midpoint_on_axis(0);

    state_t state;
    state.vertices = xv.shared();
    state.solution = (xc | nd::transform(initial_u)).shared();

    state.schedule.create("write_checkpoint");

    return state;
}




//=============================================================================
class side_effects_t
{
public:

    //=========================================================================
    side_effects_t(config::config_t run_config) : run_config(run_config) {}

    state_t operator()(state_t state, mara::perf_diagnostics_t diagnostics) const
    {
        state.schedule = print_run_loop_message(state, diagnostics);
        state.schedule = write_checkpoint(state, diagnostics);
        return state;
    }

    mara::task_schedule_t write_checkpoint(const state_t& state, mara::perf_diagnostics_t diagnostics) const
    {
        double cpi = state.iteration == 0 ? 0.0 : run_config.get<double>("cpi");

        if (state.time - state.schedule.last_performed("write_checkpoint") >= cpi)
        {
            char filename[1024];

            std::snprintf(filename, 1024, "chkpt.%04d.h5", state.schedule.num_times_performed("write_checkpoint"));
            std::printf("write checkpoint: %s\n", filename);

            auto file = h5::File(filename, "w");

            file.write("time", state.time);
            file.write("vertices", state.vertices);
            file.write("solution", state.solution);

            auto cfg_group = file.require_group("run_config");

            for (auto item : run_config)
            {
                cfg_group.write(item.first, item.second);
            }
            return state.schedule.increment("write_checkpoint", cpi);
        }
        return state.schedule;
    }

    mara::task_schedule_t print_run_loop_message(const state_t& state, mara::perf_diagnostics_t diagnostics) const
    {
        if (state.iteration > 0)
        {
            auto kzps = state.vertices.size() / diagnostics.execution_time_ms;
            std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", state.iteration, state.time, kzps);
        }
        return state.schedule;
    }

private:
    //=========================================================================
    config::config_t run_config;
};




//=============================================================================
state_t next(const state_t& state)
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

    return { t1, i1, xv, u1.shared(), state.schedule };
}




//=============================================================================
template<typename StringMapping>
auto read_restart_config(const StringMapping& mapping)
{
    // if (mapping.count("restart") && ! mapping.at("restart").empty())
    // {
    //     auto file = h5::File(mapping.at("restart"), "r");
    //     auto cfg_group = file.open_group("run_config");

    //     config::parameter_map_t run_config;

    //     for (auto key : cfg_group)
    //     {
    //         auto dset = cfg_group.open_dataset(key);
    //     }
    // }

    return config::parameter_map_t();
}




//=============================================================================
int main(int argc, const char* argv[])
{
    auto config_template = config::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)
    .item("tfinal", 1.0)
    .item("N", 256);

    auto args = config::argv_to_string_map(argc, argv);
    auto run_config = config_template.create().update(read_restart_config(args)).update(args);


    config::pretty_print(std::cout, "config", run_config);


    auto side_effects = side_effects_t(run_config);
    auto [state, diagnostics] = mara::time_execution(initial_state, run_config);
    state = side_effects(state, diagnostics);


    while (state.time < run_config.get<double>("tfinal"))
    {
        std::tie(state, diagnostics) = mara::time_execution(next, state);
        state = side_effects(state, diagnostics);
    }

	return 0;
}
