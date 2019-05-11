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
#define cfl_number 0.2




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("tfinal", 1.0)
    .item("cpi", 1.0)    // checkpoint interval
    .item("di", 0.1)     // diagnostics interval
    .item("outer_radius", 100.0)
    .item("nr", 256)
    .item("explosion_pressure", 1.0)
    .item("explosion_density", 1.0)
    .item("density_index", 0.0);
}

namespace sedov
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

using namespace sedov;




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

template<typename VertexArrayType>
auto face_areas(VertexArrayType vertices)
{
    return vertices | nd::map([] (auto r) { return mara::make_area(r * r); });
}

template<typename VertexArrayType>
auto cell_volumes(VertexArrayType vertices)
{
    auto shell_volume = [] (double r0, double r1)
    {
        return mara::make_volume((std::pow(r1, 3) - std::pow(r0, 3)) / 3);
    };
    return vertices | nd::zip_adjacent2_on_axis(0) | nd::apply(shell_volume);
}

auto make_diagnostic_fields(const solution_state_t& state)
{
    using namespace mara::srhd;
    using namespace std::placeholders;
    auto cons_to_prim = std::bind(recover_primitive, std::placeholders::_1, gamma_law_index);

    auto primitive = state.conserved | nd::divide(cell_volumes(state.vertices)) | nd::map(cons_to_prim);
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

    auto initial_p = [cfg] (auto r)
    {
        auto explosion_density  = cfg.get<double>("explosion_density");
        auto explosion_pressure = cfg.get<double>("explosion_pressure");
        auto density_index      = cfg.get<double>("explosion_density");
        auto temperature        = 1e-3;

        return mara::srhd::primitive_t()
        .mass_density(r < 1.0 ? explosion_density  : std::pow(r, -density_index))
        .gas_pressure(r < 1.0 ? explosion_pressure : std::pow(r, -density_index) * temperature);
    };
    auto to_conserved = std::bind(&mara::srhd::primitive_t::to_conserved_density, _1, gamma_law_index);

    auto nr             = cfg.get<int>("nr");
    auto outer_radius   = cfg.get<double>("outer_radius");
    auto radial_decades = std::log10(outer_radius);

    auto vertices = nd::linspace(-0.5, radial_decades, int(radial_decades * nr) + 1)
    | nd::map([] (auto y) { return std::pow(10.0, y); });

    // For linear-spaced radial zones:
    // auto vertices = nd::linspace(1, outer_radius, nr);

    auto dv = cell_volumes(vertices);
    auto xc = vertices | nd::midpoint_on_axis(0);
    auto state = solution_state_t();

    state.time = 0.0;
    state.iteration = 0;
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

static auto next_solution(const solution_state_t& state)
{
    using namespace mara::srhd;
    using namespace std::placeholders;

    auto source_terms = std::bind(&primitive_t::spherical_geometry_source_terms_radial, _1, _2, gamma_law_index);
    // auto source_terms = std::bind(&primitive_t::spherical_geometry_source_terms, _1, _2, M_PI / 2, gamma_law_index);
    auto cons_to_prim = std::bind(recover_primitive, std::placeholders::_1, gamma_law_index);

    auto dr_min = state.vertices | nd::difference_on_axis(0) | nd::read_index(0);
    auto dt = mara::make_time(cfl_number * dr_min);
    auto dv = cell_volumes(state.vertices) | nd::to_shared();
    auto da = face_areas(state.vertices);
    auto rc = state.vertices | nd::midpoint_on_axis(0);

    auto u0 = state.conserved;
    auto p0 = u0 / dv | nd::map(cons_to_prim) | nd::to_shared();
    auto s0 = nd::zip_arrays(p0, rc) | nd::apply(source_terms) | multiply(dv);
    auto l0 = p0 | nd::extend_zero_gradient(0) | intercell_flux(0) | multiply(-da) | nd::difference_on_axis(0);
    auto u1 = u0 + (l0 + s0) * dt;

    return solution_state_t {
        state.time + dt.value,
        state.iteration + 1,
        state.vertices,
        u1 | nd::to_shared() };
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
    auto diagnostics = make_diagnostic_fields(state.solution_state);

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
class subprog_sedov : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto run_config = create_run_config(argc, argv);
        auto perf = mara::perf_diagnostics_t();
        auto state = create_app_state(run_config);

        mara::pretty_print(std::cout, "config", run_config);
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
        return "sedov";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_sedov()
{
    return std::make_unique<subprog_sedov>();
}
