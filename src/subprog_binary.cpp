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
#define cfl_number 0.4
#define max_speed_assumed 10.0




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
#include "app_filesystem.hpp"
#include "physics_iso2d.hpp"




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart", std::string())
    .item("outdir",         "data")        // directory where data products are written to
    .item("cpi", 10.0)                     // checkpoint interval (chkpt.????.h5 - snapshot of app_state)
    .item("dfi", 1.0)                      // diagnostic field interval (diagnostics.????.h5 - for plotting 2d solution data)
    .item("tsi", 0.1)                      // time series interval
    .item("tfinal", 1.0)                   // simulation stop time
    .item("N", 256)                        // grid resolution (same in x and y)
    .item("SofteningRadius", 0.1)
    .item("MachNumber", 10.0)
    .item("ViscousAlpha", 0.1)
    .item("BinarySeparation", 1.0)
    .item("DomainRadius", 6.0)
    .item("BufferDampingRate", 10.0)
    .item("CounterRotate", 0);
}

namespace binary
{
    //=========================================================================
    struct solution_state_t
    {
        mara::unit_time<double> time = 0.0;
        mara::rational_number_t iteration = 0;
        nd::shared_array<mara::unit_length<double>, 1> x_vertices;
        nd::shared_array<mara::unit_length<double>, 1> y_vertices;
        nd::shared_array<mara::iso2d::conserved_t, 2> conserved;
    };

    //=========================================================================
    struct diagnostic_fields_t
    {
        mara::unit_time<double> time;
        nd::shared_array<mara::unit_length<double>, 1> x_vertices;
        nd::shared_array<mara::unit_length<double>, 1> y_vertices;
        nd::shared_array<double, 2> sigma;
        nd::shared_array<double, 2> phi_velocity;
        nd::shared_array<double, 2> radial_velocity;
    };

    //=========================================================================
    struct app_state_t
    {
        solution_state_t solution_state;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };

    //=========================================================================
    struct solver_data_t
    {
        nd::shared_array<mara::iso2d::conserved_t, 2> initial_conserved_field;
        nd::shared_array<mara::unit_rate<double>, 2> buffer_damping_rate_field;
    };


    //=========================================================================
    using location_2d_t     = mara::covariant_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t     = mara::covariant_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using acceleration_2d_t = mara::covariant_sequence_t<mara::dimensional_value_t<1, 0, -2, double>, 2>;

    struct point_mass_t
    {
        acceleration_2d_t gravitational_acceleration_at_point(location_2d_t field_point) const;
        location_2d_t mass_location = {{ 0.0, 0.0 }};
        mara::unit_mass<double> mass = 1.0;
        mara::unit_length<double> softening_radius = 0.1;
    };


    //=========================================================================
    template<typename VertexArrayType>auto cell_surface_area(const VertexArrayType& xv, const VertexArrayType& yv);
    template<typename VertexArrayType>auto cell_center_cartprod(const VertexArrayType& xv, const VertexArrayType& yv);
    auto make_diagnostic_fields(const solution_state_t& state);
    auto gravitational_acceleration_field(const solution_state_t& state);
    auto buffer_damping_rate_at_position(const mara::config_t& cfg);
    auto intercell_flux_on_axis(std::size_t axis);


    //=========================================================================
    void write_solution(h5::Group&& group, const solution_state_t& state);
    void write_diagnostic_fields(h5::Group&& group, const diagnostic_fields_t& diagnostics);


    //=========================================================================
    void write_checkpoint(const app_state_t& state, std::string outdir);
    void write_diagnostics(const app_state_t& state, std::string outdir);
    void write_time_series(const app_state_t& state, std::string outdir);


    //=========================================================================
    void print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf);
    void prepare_filesystem(const mara::config_t& cfg);
}

using namespace binary;




/**
 * @brief      Return the gravitational acceleration of a point mass at the
 *             given field point.
 *
 * @param[in]  field_point  The field point
 *
 * @return     The acceleration, whose type is ~
 *             covariant_sequence<acceleration, 2>
 */
acceleration_2d_t binary::point_mass_t::gravitational_acceleration_at_point(location_2d_t field_point) const
{
    auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
    auto dr  = field_point - mass_location;
    auto dr2 = (dr[0] * dr[0] + dr[1] * dr[1]).value;
    auto rs2 = (softening_radius * softening_radius).value;
    return -dr / mara::make_volume(std::pow(dr2 + rs2, 1.5)) * G * mass;
}




/**
 * @brief      Initial conditions from Tang+ (2017) MNRAS 469, 4258
 *
 * @param[in]  cfg   The run config
 *
 * @return     A function that maps (x, y) coordinates to primitive variable
 *             states
 *
 * @note       This should be a time-indepdent solution of flow in a thin disk,
 *             with alpha viscosity and a single point mass M located at the
 *             origin.
 */
static auto initial_disk_profile(const mara::config_t& cfg)
{
    return [cfg] (auto x_length, auto y_length)
    {
        auto SofteningRadius  = cfg.get_double("SofteningRadius");
        auto MachNumber       = cfg.get_double("MachNumber");
        auto ViscousAlpha     = cfg.get_double("ViscousAlpha");
        auto BinarySeparation = cfg.get_double("BinarySeparation");
        auto CounterRotate    = cfg.get_int("CounterRotate");

        auto GM = 1.0;
        auto x  = x_length.value;
        auto y  = y_length.value;

        auto rs             = SofteningRadius;
        auto r0             = BinarySeparation * 2.5;
        auto sigma0         = GM / BinarySeparation / BinarySeparation;
        auto r2             = x * x + y * y;
        auto r              = std::sqrt(r2);
        auto cavity_xsi     = 10.0;
        auto cavity_cutoff  = std::max(std::exp(-std::pow(r / r0, -cavity_xsi)), 1e-6);
        auto phi            = -GM * std::pow(r2 + rs * rs, -0.5);    
        auto ag             = -GM * std::pow(r2 + rs * rs, -1.5) * r;    
        auto cs2            = -phi / MachNumber / MachNumber;
        auto cs2_deriv      =   ag / MachNumber / MachNumber;
        auto sigma          = sigma0 * std::pow((r + rs) / r0, -0.5) * cavity_cutoff;
        auto sigma_deriv    = sigma0 * std::pow((r + rs) / r0, -1.5) * -0.5 / r0;
        auto dp_dr          = cs2 * sigma_deriv + cs2_deriv * sigma;
        auto omega2         = r < r0 ? GM / (4 * r0) : -ag / r + dp_dr / (sigma * r);        
        auto vq             = (CounterRotate ? -1 : 1) * r * std::sqrt(omega2);
        auto h0             = r / MachNumber;
        auto nu             = ViscousAlpha * std::sqrt(cs2) * h0; // ViscousAlpha * cs * h0
        auto vr             = -(3.0 / 2.0) * nu / (r + rs); // inward drift velocity (CHECK)
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
auto binary::cell_surface_area(const VertexArrayType& xv, const VertexArrayType& yv)
{
    auto dx = xv | nd::difference_on_axis(0);
    auto dy = yv | nd::difference_on_axis(0);
    return nd::cartesian_product(dx, dy) | nd::apply(std::multiplies<>());
}

template<typename VertexArrayType>
auto binary::cell_center_cartprod(const VertexArrayType& xv, const VertexArrayType& yv)
{
    auto xc = xv | nd::midpoint_on_axis(0);
    auto yc = yv | nd::midpoint_on_axis(0);
    return nd::cartesian_product(xc, yc);
}

auto binary::buffer_damping_rate_at_position(const mara::config_t& cfg)
{
    return [cfg] (mara::unit_length<double> x, mara::unit_length<double> y)
    {
        constexpr double tightness = 3.0;
        auto r = std::sqrt(std::pow(x.value, 2) + std::pow(y.value, 2));
        auto r1 = cfg.get_double("DomainRadius");
        return mara::make_rate(1.0 + std::tanh(tightness * (r - r1))) * cfg.get_double("BufferDampingRate");
    };
}

auto binary::gravitational_acceleration_field(const solution_state_t& state)
{
    auto star = point_mass_t();

    return nd::cartesian_product(
        state.x_vertices | nd::midpoint_on_axis(0),
        state.y_vertices | nd::midpoint_on_axis(0))
    | nd::apply([] (auto x, auto y) { return location_2d_t {{ x, y }}; })
    | nd::map([star] (auto field_point) { return star.gravitational_acceleration_at_point(field_point); });
}

auto binary::intercell_flux_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        using namespace std::placeholders;
        double sound_speed_squared = 0.1; // DEFINE PROPERLY

        auto L = array | nd::select_axis(axis).from(0).to(1).from_the_end();
        auto R = array | nd::select_axis(axis).from(1).to(0).from_the_end();
        auto nh = mara::unit_vector_t::on_axis(axis);
        auto riemann = std::bind(mara::iso2d::riemann_hlle, _1, _2, nh, sound_speed_squared);
        return nd::zip_arrays(L, R) | nd::apply(riemann);
    };
}

static auto next_solution(const solution_state_t& state, const solver_data_t& solver)
{
    auto force_to_source_terms = [] (auto v)
    {
        return mara::iso2d::flow_t {{0.0, v[0].value, v[1].value}};
    };

    auto dA = cell_surface_area(state.x_vertices, state.y_vertices);
    auto u0 = state.conserved;
    auto p0 = u0 / dA | nd::map(mara::iso2d::recover_primitive) | nd::to_shared();
    auto cell_mass = u0 | nd::map([] (auto u) { return u[0]; });
    auto sg = gravitational_acceleration_field(state) | nd::multiply(cell_mass) | nd::map(force_to_source_terms);
    auto bz = (solver.initial_conserved_field - u0) * solver.buffer_damping_rate_field;
    auto [dx, __1] = nd::meshgrid(state.x_vertices | nd::difference_on_axis(0), state.y_vertices);
    auto [__2, dy] = nd::meshgrid(state.x_vertices, state.y_vertices | nd::difference_on_axis(0));

    auto lx = p0 | nd::extend_periodic_on_axis(0) | intercell_flux_on_axis(0) | nd::multiply(-dy) | nd::difference_on_axis(0);
    auto ly = p0 | nd::extend_periodic_on_axis(1) | intercell_flux_on_axis(1) | nd::multiply(-dx) | nd::difference_on_axis(1);
    auto dt = mara::make_time(dx(0, 0).value / max_speed_assumed * cfl_number); // IMPLEMENT SEARCH FOR MAX SPEED
    auto u1 = u0 + (lx + ly + sg + bz) * dt;

    auto next_state = state;
    next_state.iteration += 1;
    next_state.time += dt;
    next_state.conserved = u1 | nd::to_shared();
    return next_state;
}




//=============================================================================
auto binary::make_diagnostic_fields(const solution_state_t& state)
{

    auto dA = cell_surface_area(state.x_vertices, state.y_vertices);
    auto [xc, yc] = nd::unzip_array(cell_center_cartprod(state.x_vertices, state.y_vertices));
    auto rc = xc * xc + yc * yc | nd::map([] (auto r2) { return mara::make_length(std::sqrt(r2.value)); });
    auto rhat_x =  xc / rc;
    auto rhat_y =  yc / rc;
    auto phat_x = -yc / rc;
    auto phat_y =  xc / rc;
    auto u = state.conserved;
    auto p = u / dA | nd::map(mara::iso2d::recover_primitive);
    auto sigma = p | nd::map(std::mem_fn(&mara::iso2d::primitive_t::sigma));
    auto vx = p | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_x));
    auto vy = p | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_y));

    auto result = diagnostic_fields_t();
    result.time            = state.time;
    result.x_vertices      = state.x_vertices;
    result.y_vertices      = state.y_vertices;
    result.sigma           = sigma                     | nd::to_shared();
    result.radial_velocity = vx * rhat_x + vy * rhat_y | nd::to_shared();
    result.phi_velocity    = vx * phat_x + vy * phat_y | nd::to_shared();
    return result;
}




//=============================================================================
void binary::write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("x_vertices", state.x_vertices);
    group.write("y_vertices", state.y_vertices);
    group.write("conserved", state.conserved);
}

void binary::write_diagnostic_fields(h5::Group&& group, const diagnostic_fields_t& diagnostics)
{
    group.write("time", diagnostics.time);
    group.write("x_vertices", diagnostics.x_vertices);
    group.write("y_vertices", diagnostics.y_vertices);
    group.write("sigma", diagnostics.sigma);
    group.write("phi_velocity", diagnostics.phi_velocity);
    group.write("radial_velocity", diagnostics.radial_velocity);
}

void binary::write_checkpoint(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_checkpoint");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", count, "h5")), "w");

    write_solution(file.require_group("solution"), state.solution_state);
    mara::write_schedule(file.require_group("schedule"), state.schedule);
    mara::write_config(file.require_group("run_config"), state.run_config);

    std::printf("write checkpoint: %s\n", file.filename().data());
}

void binary::write_diagnostics(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_diagnostics");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5")), "w");
    auto diagnostics = make_diagnostic_fields(state.solution_state);

    write_diagnostic_fields(file.open_group("/"), diagnostics);

    // for (auto item : diagnostics.time_series_data)
    // {
    //     file.write(item.first, item.second);
    // }

    std::printf("write diagnostics: %s\n", file.filename().data());
}

void binary::write_time_series(const app_state_t& state, std::string outdir)
{
    // auto file = h5::File(mara::filesystem::join({outdir, "time_series.h5"}), "r+");
    // auto current_size = state.schedule.num_times_performed("write_time_series");
    // auto target_space = h5::hyperslab_t{{std::size_t(current_size)}, {1}, {1}, {1}};

    // for (auto item : compute_time_series_data(state.solution_state))
    // {
    //     auto dataset = file.open_dataset(item.first);
    //     dataset.set_extent(current_size + 1);
    //     dataset.write(item.second, dataset.get_space().select(target_space));
    // }
}




//=============================================================================
static auto read_solution(h5::Group&& group)
{
    return solution_state_t(); // IMPLEMENT READING SOLUTION FROM CHECKPOINT
}

static auto new_solution(const mara::config_t& cfg)
{
    auto nx = cfg.get_int("N");
    auto ny = cfg.get_int("N");
    auto R0 = cfg.get_double("DomainRadius");

    auto xv = nd::linspace(-R0, R0, nx + 1) | nd::map(mara::make_length<double>);
    auto yv = nd::linspace(-R0, R0, ny + 1) | nd::map(mara::make_length<double>);

    auto u = cell_center_cartprod(xv, yv)
    | nd::apply(initial_disk_profile(cfg))
    | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area))
    | nd::multiply(cell_surface_area(xv, yv));

    auto state = solution_state_t();
    state.time = 0.0;
    state.iteration = 0;
    state.x_vertices = xv | nd::to_shared();
    state.y_vertices = yv | nd::to_shared();
    state.conserved = u | nd::to_shared();
    return state;
}

static auto create_solution(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");
    return restart.empty()
    ? new_solution(run_config)
    : read_solution(h5::File(restart, "r").open_group("solution"));
}

static auto create_solver_data(const mara::config_t& cfg)
{
    auto initial_state = new_solution(cfg);
    auto result = solver_data_t();

    result.initial_conserved_field = initial_state.conserved;
    result.buffer_damping_rate_field = cell_center_cartprod(initial_state.x_vertices, initial_state.y_vertices)
    | nd::apply(buffer_damping_rate_at_position(cfg))
    | nd::to_shared();

    return result;
}




//=============================================================================
static auto new_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_checkpoint");
    schedule.create_and_mark_as_due("write_diagnostics");
    schedule.create_and_mark_as_due("write_time_series");
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
static auto create_app_state(mara::config_t run_config)
{
    auto state = app_state_t();
    state.run_config     = run_config;
    state.solution_state = create_solution(run_config);
    state.schedule       = create_schedule(run_config);
    return state;
}

static auto create_app_state_next_function(const solver_data_t& solver_data)
{
    return [solver_data] (const app_state_t& state)
    {
        auto next_state = state;
        next_state.solution_state = next_solution(state.solution_state, solver_data);
        next_state.schedule       = next_schedule(state.schedule, state.run_config, state.solution_state.time.value);
        return next_state;
    };
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
void binary::print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.x_vertices.size() * solution.y_vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration.as_integral(), solution.time.value, kzps);
}

void binary::prepare_filesystem(const mara::config_t& cfg)
{
    if (cfg.get_string("restart").empty())
    {
        auto outdir = cfg.get_string("outdir");
        mara::filesystem::require_dir(outdir);

        auto file = h5::File(mara::filesystem::join(outdir, "time_series.h5"), "w");
        auto plist = h5::PropertyList::dataset_create().set_chunk(1000);
        auto space = h5::Dataspace::unlimited(0);

        // for (auto column_name : get_time_series_column_names())
        // {
        //     file.require_dataset(column_name, h5::Datatype::native_double(), space, plist);            
        // }
        mara::write_config(file.require_group("run_config"), cfg);
    }
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
        auto state       = create_app_state(run_config);
        auto solver_data = create_solver_data(run_config);
        auto next        = create_app_state_next_function(solver_data);

        prepare_filesystem(run_config);
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
