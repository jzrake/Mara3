#include <iostream>
#include "ndmpi.hpp"
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "core_geometric.hpp"
#include "core_rational.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_subprogram.hpp"
#include "physics_srhd.hpp"
#include "physics_euler.hpp"

#define gamma_law_index (4. / 3)
#define cfl_number 0.4




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart",  std::string())   // name of a restart file (create new run if empty)
    .item("outdir",          "data")   // directory to put output files in
    .item("nr",                 256)   // number of radial zones, per decade
    .item("tfinal",             1.0)   // time to stop the simulation
    .item("cpi",                1.0)   // checkpoint interval
    .item("tsi",                0.1)   // time-series interval
    .item("dfi",                0.1)   // diagnostic field interval (useful primitives)
    .item("outer_radius",     100.0)   // outer boundary radius
    .item("explosion_pressure", 1.0)   // gas pressure between 0.5 < r < 1.0
    .item("explosion_density",  1.0)   // mass density between 0.5 < r < 1.0
    .item("density_index",      0.0)   // index n of the power-law ambient density profile, rho = r^(-n)
    .item("newtonian",            0);  // whether to use euler equations instead of srhd
}




//=============================================================================
static double radial_velocity_or_gamma_beta(const mara::srhd::primitive_t& p)
{
    return p.gamma_beta_1();
}
static double radial_velocity_or_gamma_beta(const mara::euler::primitive_t& p)
{
    return p.velocity_1();
}
static auto negate_radial_velocity(const mara::srhd::primitive_t& p)
{
    return p.with_gamma_beta_1(-p.gamma_beta_1());
}
static auto negate_radial_velocity(const mara::euler::primitive_t& p)
{
    return p.with_velocity_1(-p.velocity_1());
}




//=============================================================================
template<typename HydroSystem>
struct SedovProblem
{


    //=========================================================================
    struct solution_state_t
    {
        double time = 0.0;
        mara::rational_number_t iteration = mara::make_rational(0, 1);
        nd::shared_array<double, 1> vertices;
        nd::shared_array<typename HydroSystem::conserved_t, 1> conserved;
    };

    struct diagnostic_fields_t
    {
        double time = 0.0;
        double shock_radius = 1.0;
        nd::shared_array<double, 1> mass_density;
        nd::shared_array<double, 1> gas_pressure;
        nd::shared_array<double, 1> specific_entropy;
        nd::shared_array<double, 1> radial_gamma_beta;
        nd::shared_array<double, 1> radial_coordinates;
    };

    struct app_state_t
    {
        solution_state_t solution_state;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };


    //=========================================================================
    static auto intercell_flux(std::size_t axis);
    static auto extend_reflecting_inner();
    static auto extend_zero_gradient_outer();
    static auto find_shock_radius(const solution_state_t& state);
    static auto make_diagnostic_fields(const solution_state_t& state);


    //=========================================================================
    template<typename VertexArrayType>
    static auto face_areas(VertexArrayType vertices)
    {
        return vertices | nd::map([] (auto r) { return mara::make_area(r * r); });
    }

    template<typename VertexArrayType>
    static auto cell_volumes(VertexArrayType vertices)
    {
        auto shell_volume = [] (double r0, double r1)
        {
            return mara::make_volume((std::pow(r1, 3) - std::pow(r0, 3)) / 3);
        };
        return vertices | nd::zip_adjacent2_on_axis(0) | nd::apply(shell_volume);
    }


    //=============================================================================
    static void write_solution(h5::Group&& group, const solution_state_t& state);
    static auto read_solution(h5::Group&& group);
    static auto new_solution(const mara::config_t& cfg);
    static auto create_solution(const mara::config_t& run_config);
    static auto next_solution(const solution_state_t& state);


    //=============================================================================
    static auto new_schedule(const mara::config_t& run_config);
    static auto create_schedule(const mara::config_t& run_config);
    static auto next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time);


    //=========================================================================
    static void write_checkpoint(const app_state_t& state, std::string outdir);
    static void write_diagnostics(const app_state_t& state, std::string outdir);
    static void write_time_series(const app_state_t& state, std::string outdir);
    static auto create_app_state(mara::config_t run_config);
    static auto next(const app_state_t& state);
    static auto simulation_should_continue(const app_state_t& state);
    static auto run_tasks(const app_state_t& state);


    //=========================================================================
    static void print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf);
    static void prepare_filesystem(const mara::config_t& cfg);
};




//=============================================================================
template<typename HydroSystem>
auto SedovProblem<HydroSystem>::intercell_flux(std::size_t axis)
{
    return [axis] (auto array)
    {
        using namespace std::placeholders;
        auto L = array | nd::select_axis(axis).from(0).to(1).from_the_end();
        auto R = array | nd::select_axis(axis).from(1).to(0).from_the_end();
        auto nh = mara::unit_vector_t::on_axis_1();
        auto riemann = std::bind(HydroSystem::riemann_hlle, _1, _2, nh, gamma_law_index);
        return nd::zip_arrays(L, R) | nd::apply(riemann);
    };
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::extend_reflecting_inner()
{
    return [] (auto array)
    {
        auto xl = array
        | nd::select_first(1, 0)
        | nd::map([] (auto p) { return negate_radial_velocity(p); });
        return xl | nd::concat(array);
    };
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::extend_zero_gradient_outer()
{
    return [] (auto array)
    {
        return array | nd::concat(array | nd::select_final(1, 0));
    };
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::find_shock_radius(const solution_state_t& state)
{
    using namespace std::placeholders;
    auto cons_to_prim = std::bind(HydroSystem::recover_primitive, _1, gamma_law_index);

    auto primitive = state.conserved | nd::divide(cell_volumes(state.vertices)) | nd::map(cons_to_prim);
    auto rc = state.vertices | nd::midpoint_on_axis(0);
    auto s0 = primitive | nd::map(std::bind(&HydroSystem::primitive_t::specific_entropy, _1, gamma_law_index));
    auto ds = s0 | nd::difference_on_axis(0);
    auto shock_index = nd::where(ds == nd::min(ds)) | nd::read_index(0);
    return rc(shock_index);
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::make_diagnostic_fields(const solution_state_t& state)
{
    using namespace std::placeholders;
    auto cons_to_prim = std::bind(HydroSystem::recover_primitive, _1, gamma_law_index);

    auto primitive = state.conserved | nd::divide(cell_volumes(state.vertices)) | nd::map(cons_to_prim);
    auto result = diagnostic_fields_t();

    result.time               = state.time;
    result.specific_entropy   = primitive | nd::map(std::bind(&HydroSystem::primitive_t::specific_entropy, _1, gamma_law_index)) | nd::to_shared();
    result.gas_pressure       = primitive | nd::map(std::mem_fn(&HydroSystem::primitive_t::gas_pressure)) | nd::to_shared();
    result.mass_density       = primitive | nd::map(std::mem_fn(&HydroSystem::primitive_t::mass_density)) | nd::to_shared();
    result.radial_gamma_beta  = primitive | nd::map([](auto p){return radial_velocity_or_gamma_beta(p);}) | nd::to_shared();
    result.radial_coordinates = state.vertices | nd::midpoint_on_axis(0) | nd::to_shared();
    result.shock_radius       = find_shock_radius(state);

    return result;
}




//=============================================================================
template<typename HydroSystem>
void SedovProblem<HydroSystem>::write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("vertices", state.vertices);
    group.write("conserved", state.conserved);
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::read_solution(h5::Group&& group)
{
    auto state = solution_state_t();
    state.time      = group.read<double>("time");
    state.iteration = group.read<mara::rational_number_t>("iteration");
    state.vertices  = group.read<nd::unique_array<double, 1>>("vertices").shared();
    state.conserved = group.read<nd::unique_array<typename HydroSystem::conserved_t, 1>>("conserved").shared();
    return state;
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::new_solution(const mara::config_t& cfg)
{
    using namespace std::placeholders;

    auto initial_p = [cfg] (auto r)
    {
        auto explosion_density  = cfg.get_double("explosion_density");
        auto explosion_pressure = cfg.get_double("explosion_pressure");
        auto density_index      = cfg.get_double("density_index");
        auto temperature        = 1e-6;

        return typename HydroSystem::primitive_t()
        .with_mass_density(r < 1.0 ? explosion_density  : std::pow(r, -density_index))
        .with_gas_pressure(r < 1.0 ? explosion_pressure : std::pow(r, -density_index) * temperature);
    };
    auto to_conserved = std::bind(&HydroSystem::primitive_t::to_conserved_density, _1, gamma_law_index);

    auto nr             = cfg.get_int("nr");
    auto outer_radius   = cfg.get_double("outer_radius");
    auto radial_decades = std::log10(outer_radius);

    auto vertices = nd::linspace(-0.5, radial_decades, int(radial_decades * nr) + 1)
    | nd::map([] (auto y) { return std::pow(10.0, y); });

    auto dv = cell_volumes(vertices);
    auto xc = vertices | nd::midpoint_on_axis(0);
    auto state = solution_state_t();

    state.time = 0.0;
    state.iteration = 0;
    state.vertices = vertices.shared();
    state.conserved = xc | nd::map(initial_p) | nd::map(to_conserved) | multiply(dv) | nd::to_shared();

    return state;
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::create_solution(const mara::config_t& run_config)
{
    auto restart = run_config.get_string("restart");
    return restart.empty()
    ? new_solution(run_config)
    : read_solution(h5::File(restart, "r").open_group("solution"));
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::next_solution(const solution_state_t& state)
{
    using namespace std::placeholders;

    auto source_terms = std::bind(&HydroSystem::primitive_t::spherical_geometry_source_terms_radial, _1, _2, gamma_law_index);
    auto cons_to_prim = std::bind(HydroSystem::recover_primitive, std::placeholders::_1, gamma_law_index);
    auto extend_bc = mara::compose(extend_reflecting_inner(), extend_zero_gradient_outer());

    auto dr_min = state.vertices | nd::difference_on_axis(0) | nd::read_index(0);
    auto dt = mara::make_time(cfl_number * dr_min);
    auto dv = cell_volumes(state.vertices) | nd::to_shared();
    auto da = face_areas(state.vertices);
    auto rc = state.vertices | nd::midpoint_on_axis(0);

    auto u0 = state.conserved;
    auto p0 = u0 / dv | nd::map(cons_to_prim) | nd::to_shared();
    auto s0 = nd::zip_arrays(p0, rc) | nd::apply(source_terms) | multiply(dv);
    auto l0 = p0 | extend_bc | intercell_flux(0) | multiply(-da) | nd::difference_on_axis(0);
    auto u1 = u0 + (l0 + s0) * dt;

    return solution_state_t {
        state.time + dt.value,
        state.iteration + 1,
        state.vertices,
        u1 | nd::to_shared() };
}




//=============================================================================
template<typename HydroSystem>
auto SedovProblem<HydroSystem>::new_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_checkpoint");
    schedule.create_and_mark_as_due("write_diagnostics");
    schedule.create_and_mark_as_due("write_time_series");
    return schedule;
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::create_schedule(const mara::config_t& run_config)
{
    auto restart = run_config.get_string("restart");
    return restart.empty()
    ? new_schedule(run_config)
    : mara::read_schedule(h5::File(restart, "r").open_group("schedule"));
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time)
{
    auto next_schedule = schedule;
    auto cpi = run_config.get_double("cpi");
    auto dfi = run_config.get_double("dfi");
    auto tsi = run_config.get_double("tsi");

    if (time - schedule.last_performed("write_checkpoint")  >= cpi) next_schedule.mark_as_due("write_checkpoint",  cpi);
    if (time - schedule.last_performed("write_diagnostics") >= dfi) next_schedule.mark_as_due("write_diagnostics", dfi);
    if (time - schedule.last_performed("write_time_series") >= tsi) next_schedule.mark_as_due("write_time_series", tsi);

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
template<typename HydroSystem>
void SedovProblem<HydroSystem>::write_checkpoint(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_checkpoint");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", count, "h5")), "w");
    write_solution(file.require_group("solution"), state.solution_state);
    mara::write_schedule(file.require_group("schedule"), state.schedule);
    mara::write_config(file.require_group("run_config"), state.run_config);

    std::printf("write checkpoint: %s\n", file.filename().data());
}

template<typename HydroSystem>
void SedovProblem<HydroSystem>::write_diagnostics(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_diagnostics");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5")), "w");
    auto diagnostics = make_diagnostic_fields(state.solution_state);

    file.write("time",               diagnostics.time);
    file.write("gas_pressure",       diagnostics.gas_pressure);
    file.write("mass_density",       diagnostics.mass_density);
    file.write("specific_entropy",   diagnostics.specific_entropy);
    file.write("radial_gamma_beta",  diagnostics.radial_gamma_beta);
    file.write("radial_coordinates", diagnostics.radial_coordinates);
    file.write("shock_radius",       diagnostics.shock_radius);

    std::printf("write diagnostics: %s\n", file.filename().data());
}

template<typename HydroSystem>
void SedovProblem<HydroSystem>::write_time_series(const app_state_t& state, std::string outdir)
{
    auto file = h5::File(mara::filesystem::join({outdir, "time_series.h5"}), "r+");
    auto time         = file.open_dataset("time");
    auto shock_radius = file.open_dataset("shock_radius");
    auto current_size = state.schedule.num_times_performed("write_time_series");
    auto target_space = h5::hyperslab_t{{std::size_t(current_size)}, {1}, {1}, {1}};

    time.set_extent(current_size + 1);
    shock_radius.set_extent(current_size + 1);

    time.write(state.solution_state.time, time.get_space().select(target_space));
    shock_radius.write(find_shock_radius(state.solution_state), shock_radius.get_space().select(target_space));
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::create_app_state(mara::config_t run_config)
{
    auto state = app_state_t();
    state.run_config     = run_config;
    state.solution_state = create_solution(run_config);
    state.schedule       = create_schedule(run_config);
    return state;
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::next(const app_state_t& state)
{
    auto next_state = state;
    next_state.solution_state = next_solution(state.solution_state);
    next_state.schedule       = next_schedule(state.schedule, state.run_config, state.solution_state.time);
    return next_state;
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::simulation_should_continue(const app_state_t& state)
{
    auto time = state.solution_state.time;
    auto tfinal = state.run_config.get_double("tfinal");
    return time < tfinal;
}

template<typename HydroSystem>
auto SedovProblem<HydroSystem>::run_tasks(const app_state_t& state)
{
    auto next_state = state;
    auto outdir = state.run_config.get_string("outdir");

    if (state.schedule.is_due("write_checkpoint"))
    {
        write_checkpoint(state, outdir);
        next_state.schedule.mark_as_completed("write_checkpoint");
    }
    if (state.schedule.is_due("write_diagnostics"))
    {
        write_diagnostics(state, outdir);
        next_state.schedule.mark_as_completed("write_diagnostics");
    }
    if (state.schedule.is_due("write_time_series"))
    {
        write_time_series(state, outdir);
        next_state.schedule.mark_as_completed("write_time_series");
    }
    return next_state;
}




//=============================================================================
template<typename HydroSystem>
void SedovProblem<HydroSystem>::print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration.as_integral(), solution.time, kzps);
}

template<typename HydroSystem>
void SedovProblem<HydroSystem>::prepare_filesystem(const mara::config_t& cfg)
{
    if (cfg.get_string("restart").empty())
    {
        auto outdir = cfg.get_string("outdir");
        mara::filesystem::require_dir(outdir);

        auto file = h5::File(mara::filesystem::join(outdir, "time_series.h5"), "w");
        auto plist = h5::PropertyList::dataset_create().set_chunk(1000);
        auto space = h5::Dataspace::unlimited(0);

        file.require_dataset("time", h5::Datatype::native_double(), space, plist);
        file.require_dataset("shock_radius", h5::Datatype::native_double(), space, plist);
        mara::write_config(file.require_group("run_config"), cfg);
    }
    else
    {
        // should truncate trailing iterations here...
    }
}




//=============================================================================
class subprog_sedov : public mara::sub_program_t
{
public:

    template<typename HydroSystem>
    int run_main(const mara::config_t& run_config)
    {
        using prob             = SedovProblem<HydroSystem>;
        auto run_tasks_on_next = mara::compose(prob::run_tasks, prob::next);
        auto perf              = mara::perf_diagnostics_t();
        auto state             = prob::create_app_state(run_config);

        prob::prepare_filesystem(run_config);
        mara::pretty_print(std::cout, "config", run_config);

        state = prob::run_tasks(state);

        while (prob::simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(run_tasks_on_next, state);
            prob::print_run_loop_message(state.solution_state, perf);
        }

        run_tasks_on_next(state);
        return 0;
    }

    int main(int argc, const char* argv[]) override
    {
        auto run_config = create_run_config(argc, argv);

        if (run_config.get_int("newtonian") != 0)
        {
            return run_main<mara::euler>(run_config);
        }
        else
        {
            return run_main<mara::srhd>(run_config);
        }
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
