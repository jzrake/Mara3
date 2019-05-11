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
#include "physics_srhd.hpp"
#include "core_geometric.hpp"
#include "core_rational.hpp"
#define gamma_law_index (4. / 3)




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)
    .item("tfinal", 1.0)
    .item("N", 256);
}

namespace shockwave
{
    struct solution_state_t
    {
        double time = 0.0;
        mara::rational_value_t iteration = mara::make_rational(0, 1);
        nd::shared_array<double, 1> vertices;
        nd::shared_array<mara::srhd::conserved_t, 1> solution;
    };
}

using namespace shockwave;




//=============================================================================
static auto intercell_flux(std::size_t axis)
{
    return [axis] (auto array)
    {
        using namespace std::placeholders;
        auto L = array | nd::select_axis(axis).from(0).to(1).from_the_end();
        auto R = array | nd::select_axis(axis).from(1).to(0).from_the_end();
        auto nh = mara::unit_vector_t::on_axis_1();
        auto riemann = std::bind(mara::srhd::riemann_hlle, _1, _2, nh, gamma_law_index);
        return nd::zip_arrays(L, R) | nd::apply(riemann);
    };
}

static auto extend_constant(std::size_t axis)
{
    return [axis] (auto array)
    {
        auto xl = array | nd::select_first(1, 0);
        auto xr = array | nd::select_final(1, 0);
        return xl | nd::concat(array).on_axis(axis) | nd::concat(xr).on_axis(axis);
    };
}

template<typename Multiplier>
auto multiply(Multiplier arg)
{
    return std::bind(std::multiplies<>(), std::placeholders::_1, arg);
};

template<typename Multiplier>
auto divide(Multiplier arg)
{
    return std::bind(std::divides<>(), std::placeholders::_1, arg);
};




//=============================================================================
static void write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration.as_integral());
    group.write("vertices", state.vertices);
    group.write("conserved", state.solution);
}

static auto read_solution(h5::Group&& group)
{
    auto state = solution_state_t();
    state.time      = group.read<double>("time");
    state.iteration = group.read<int>("iteration");
    state.vertices  = group.read<nd::unique_array<double, 1>>("vertices").shared();
    state.solution  = group.read<nd::unique_array<mara::srhd::conserved_t, 1>>("conserved").shared();
    return state;
}

static auto new_solution(const mara::config_t& cfg)
{
    using namespace std::placeholders;

    auto initial_p = [] (auto x)
    {
        return mara::srhd::primitive_t()
        .mass_density(x < 0.5 ? 1.0 : 0.100)
        .gas_pressure(x < 0.5 ? 1.0 : 0.125);
    };
    auto to_conserved = std::bind(&mara::srhd::primitive_t::to_conserved_density, _1, gamma_law_index);

    auto nx = cfg.get<int>("N");
    auto xv = nd::linspace(0, 1, nx + 1);
    auto dv = xv | nd::difference_on_axis(0) | nd::map(mara::make_volume<double>);
    auto xc = xv | nd::midpoint_on_axis(0);
    auto state = solution_state_t();

    state.vertices = xv.shared();
    state.solution = xc | nd::map(initial_p) | nd::map(to_conserved) | multiply(dv) | nd::to_shared();

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
    auto dt = mara::make_time(0.25 / state.vertices.shape(0));
    auto du = state.solution
    | divide(state.vertices | nd::difference_on_axis(0) | nd::map(mara::make_volume<double>))
    | nd::map(std::bind(mara::srhd::recover_primitive, std::placeholders::_1, gamma_law_index))
    | extend_constant(0)
    | nd::to_shared()
    | intercell_flux(0)
    | nd::to_shared()
    | nd::difference_on_axis(0)
    | multiply(-dt * mara::make_area(1.0));

    return solution_state_t {
        state.time + dt.value,
        state.iteration + 1,
        state.vertices,
        state.solution + du | nd::to_shared() };
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
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration.as_integral(), solution.time, kzps);
}




//=============================================================================
class subprog_shockwave : public mara::sub_program_t
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

std::unique_ptr<mara::sub_program_t> make_subprog_shockwave()
{
    return std::make_unique<subprog_shockwave>();
}
