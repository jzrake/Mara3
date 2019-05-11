#include <iostream>
#include "ndmpi.hpp"
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "core_geometric.hpp"
#include "core_rational.hpp"
#include "app_config.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_subprogram.hpp"
#include "physics_srhd.hpp"
#define gamma_law_index (4. / 3)




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("cpi", 1.0)    // checkpoint interval
    .item("di", 0.1)     // diagnostics interval
    .item("tfinal", 1.0)
    .item("geometry", "planar")
    .item("N", 256);
}

namespace shockwave
{
    struct solution_state_t
    {
        double time = 0.0;
        mara::rational_number_t iteration = mara::make_rational(0, 1);
        nd::shared_array<double, 1> vertices;
        nd::shared_array<mara::srhd::conserved_t, 1> conserved;
    };

    struct diagnostic_fields_t
    {
        double time = 0.0;
        nd::shared_array<double, 1> mass_density;
        nd::shared_array<double, 1> gas_pressure;
        nd::shared_array<double, 1> radial_gamma_beta;
        nd::shared_array<double, 1> radial_coordinates;
    };
}

using namespace shockwave;




//=============================================================================
template<typename F1, typename F2, typename... Args>
auto call_with_geometry(const mara::config_t& cfg, F1 planar, F2 spherical, Args&&... args)
{
    auto geometry = cfg.get<std::string>("geometry");

    if (geometry == "planar")
    {
        return planar(std::forward<Args>(args)...);
    }
    if (geometry == "spherical")
    {
        return spherical(std::forward<Args>(args)...);
    }
    throw std::invalid_argument("unrecognized geometry '" + geometry + "'");
}




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

static auto extend_zero_gradient(std::size_t axis)
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

template<typename VertexArrayType>
auto face_areas_planar(VertexArrayType vertices)
{
    return mara::make_area(1.0);
}

template<typename VertexArrayType>
auto face_areas_spherical(VertexArrayType vertices)
{
    return vertices | nd::map([] (auto r) { return mara::make_area(r * r); });
}

template<typename VertexArrayType>
auto cell_volumes_planar(VertexArrayType vertices)
{
    auto slab_volume = [] (double x0, double x1)
    {
        return mara::make_volume(x1 - x0);
    };
    return vertices | nd::zip_adjacent2_on_axis(0) | nd::apply(slab_volume);
}

template<typename VertexArrayType>
auto cell_volumes_spherical(VertexArrayType vertices)
{
    auto shell_volume = [] (double r0, double r1)
    {
        return mara::make_volume((std::pow(r1, 3) - std::pow(r0, 3)) / 3);
    };
    return vertices | nd::zip_adjacent2_on_axis(0) | nd::apply(shell_volume);
}

template<typename VertexArrayType>
auto cell_volumes(VertexArrayType vertices, const mara::config_t& cfg)
{
    auto p = [] (auto v) { return cell_volumes_planar(v) | nd::to_shared(); };
    auto s = [] (auto v) { return cell_volumes_spherical(v) | nd::to_shared(); };
    return call_with_geometry(cfg, p, s, vertices);
}

auto make_diagnostic_fields(const solution_state_t& state, const mara::config_t& cfg)
{
    using namespace mara::srhd;
    using namespace std::placeholders;
    auto cons_to_prim = std::bind(recover_primitive, std::placeholders::_1, gamma_law_index);

    auto primitive = state.conserved | divide(cell_volumes(state.vertices, cfg)) | nd::map(cons_to_prim);
    auto result = diagnostic_fields_t();

    result.time               = state.time;
    result.gas_pressure       = primitive | nd::map([] (auto p) { return p.gas_pressure(); }) | nd::to_shared();
    result.mass_density       = primitive | nd::map([] (auto p) { return p.mass_density(); }) | nd::to_shared();
    result.radial_gamma_beta  = primitive | nd::map([] (auto p) { return p.gamma_beta_1(); }) | nd::to_shared();
    result.radial_coordinates = state.vertices | nd::midpoint_on_axis(0) | nd::to_shared();

    return result;
}




//=============================================================================
static void write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("vertices", state.vertices);
    group.write("conserved", state.conserved);
}

static auto read_solution(h5::Group&& group)
{
    auto state = solution_state_t();
    state.time      = group.read<double>("time");
    state.iteration = group.read<mara::rational_number_t>("iteration");
    state.vertices  = group.read<nd::unique_array<double, 1>>("vertices").shared();
    state.conserved = group.read<nd::unique_array<mara::srhd::conserved_t, 1>>("conserved").shared();
    return state;
}

static auto new_solution(const mara::config_t& cfg)
{
    using namespace std::placeholders;

    // auto initial_p = [] (auto x)
    // {
    //     return mara::srhd::primitive_t()
    //     .mass_density(x < 0.5 ? 1.0 : 0.100)
    //     .gas_pressure(x < 0.5 ? 1.0 : 0.125);
    // };

    auto initial_p = [] (auto r)
    {
        return mara::srhd::primitive_t()
        .mass_density(r < 2.0 ? 1.0 : 0.025)
        .gas_pressure(r < 2.0 ? 1.0 : 0.025);
    };
    auto to_conserved = std::bind(&mara::srhd::primitive_t::to_conserved_density, _1, gamma_law_index);

    auto nx = cfg.get<int>("N");
    // auto vertices = nd::linspace(0, 1, nx + 1);
    auto vertices = nd::linspace(1, 10, nx + 1);
    auto dv = cell_volumes(vertices, cfg);
    auto xc = vertices | nd::midpoint_on_axis(0);
    auto state = solution_state_t();

    state.vertices = vertices.shared();
    state.conserved = xc | nd::map(initial_p) | nd::map(to_conserved) | multiply(dv) | nd::to_shared();

    return state;
}

static auto create_solution(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");
    return restart.empty()
    ? new_solution(run_config)
    : read_solution(h5::File(restart, "r").open_group("solution"));
}

static auto next_solution_spherical(const solution_state_t& state)
{
    using namespace mara::srhd;
    using namespace std::placeholders;

    auto source_terms = std::bind(&primitive_t::spherical_geometry_source_terms_radial, _1, _2, gamma_law_index);
    auto cons_to_prim = std::bind(recover_primitive, std::placeholders::_1, gamma_law_index);

    auto dt = mara::make_time(0.025 / state.vertices.shape(0));
    auto dv = cell_volumes_spherical(state.vertices) | nd::to_shared();
    auto da = face_areas_spherical(state.vertices);
    auto rc = state.vertices | nd::midpoint_on_axis(0);

    auto u0 = state.conserved;
    auto p0 = u0 / dv | nd::map(cons_to_prim) | nd::to_shared();
    auto s0 = nd::zip_arrays(p0, rc) | nd::apply(source_terms) | multiply(dv);
    auto l0 = p0 | extend_zero_gradient(0) | intercell_flux(0) | multiply(-da) | nd::difference_on_axis(0);
    auto u1 = u0 + (l0 + s0) * dt;

    return solution_state_t {
        state.time + dt.value,
        state.iteration + 1,
        state.vertices,
        u1 | nd::to_shared() };
}

static auto next_solution_planar(const solution_state_t& state)
{
    auto dt = mara::make_time(0.25 / state.vertices.shape(0));
    auto du = state.conserved
    | divide(cell_volumes_planar(state.vertices))
    | nd::map(std::bind(mara::srhd::recover_primitive, std::placeholders::_1, gamma_law_index))
    | extend_zero_gradient(0)
    | nd::to_shared()
    | intercell_flux(0)
    | nd::to_shared()
    | multiply(face_areas_planar(state.vertices))
    | nd::difference_on_axis(0)
    | multiply(-dt);

    return solution_state_t {
        state.time + dt.value,
        state.iteration + 1,
        state.vertices,
        state.conserved + du | nd::to_shared() };
}

static auto next_solution(const solution_state_t& state, const mara::config_t& cfg)
{
    return call_with_geometry(cfg, next_solution_planar, next_solution_spherical, state);
}




//=============================================================================
static auto new_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create("write_checkpoint");
    schedule.create("write_diagnostics");
    schedule.mark_as_due("write_checkpoint");
    schedule.mark_as_due("write_diagnostics");
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
    auto di = run_config.get<double>("di");

    if (time - schedule.last_performed("write_checkpoint") >= cpi)
    {
        next_schedule.mark_as_due("write_checkpoint", cpi);
    }
    if (time - schedule.last_performed("write_diagnostics") >= di)
    {
        next_schedule.mark_as_due("write_diagnostics", di);
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

static void write_diagnostics(const app_state_t& state)
{
    auto count = state.schedule.num_times_performed("write_diagnostics");
    auto file = h5::File(mara::create_numbered_filename("diagnostics", count, "h5"), "w");
    auto diagnostics = make_diagnostic_fields(state.solution_state, state.run_config);

    file.write("time",               diagnostics.time);
    file.write("gas_pressure",       diagnostics.gas_pressure);
    file.write("mass_density",       diagnostics.mass_density);
    file.write("radial_gamma_beta",  diagnostics.radial_gamma_beta);
    file.write("radial_coordinates", diagnostics.radial_coordinates);

    std::printf("write diagnostics: %s\n", file.filename().data());
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
    next_state.solution_state = next_solution(state.solution_state, state.run_config);
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
    if (state.schedule.is_due("write_diagnostics"))
    {
        write_diagnostics(state);
        next_state.schedule.mark_as_completed("write_diagnostics");
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
