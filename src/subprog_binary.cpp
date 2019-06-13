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




#include <cmath>
#include <iostream>
#include "core_hdf5.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "app_config.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_subprogram.hpp"
#include "app_filesystem.hpp"
#include "physics_iso2d.hpp"
#include "model_two_body.hpp"
#define cfl_number                    0.4
#define density_floor                 0.0
#define sound_speed_squared          1e-4
// #define log10_sigma_diffusive_below  -6.0
// #define log10_sigma_aggressive_above -2.0




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart",             std::string())
    .item("outdir",              "data")        // directory where data products are written to
    .item("cpi",                 10.0)          // checkpoint interval (orbits; chkpt.????.h5 - snapshot of app_state)
    .item("dfi",                  1.0)          // diagnostic field interval (orbits; diagnostics.????.h5 - for plotting 2d solution data)
    .item("tsi",                  0.1)          // time series interval (orbits)
    .item("tfinal",               1.0)          // simulation stop time (orbits)
    .item("N",                    256)          // grid resolution (same in x and y)
    .item("rk_order",               2)          // time-stepping Runge-Kutta order: 1 or 2
    .item("reconstruct_method", "plm")          // zone extrapolation method: pcm or plm
    .item("plm_theta",            1.8)          // plm theta parameter: [1.0, 2.0]
    .item("riemann",           "hllc")          // riemann solver to use: hlle or hllc
    .item("softening_radius",     0.1)          // gravitational softening radius
    .item("sink_radius",          0.1)          // radius of mass (and momentum) subtraction region
    .item("sink_rate",            1e2)          // sink rate at the point masses (orbital angular frequency)
    // .item("mach_number",         10.0)       // not implemented yet
    // .item("viscous_alpha",        0.1)       // not implemented yet
    .item("domain_radius",        6.0)          // half-size of square domain
    .item("separation",           1.0)          // binary separation: 0.0 or 1.0 (zero emulates a single body)
    .item("mass_ratio",           1.0)          // binary mass ratio M2 / M1: (0.0, 1.0]
    .item("eccentricity",         0.0)          // orbital eccentricity: [0.0, 1.0)
    .item("buffer_damping_rate", 10.0)          // maximum rate of buffer zone, where solution is driven to initial state
    .item("counter_rotate",         0);         // retrograde disk option: 0 or 1
}




namespace binary
{


    //=========================================================================
    using location_2d_t     = mara::covariant_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t     = mara::covariant_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using acceleration_2d_t = mara::covariant_sequence_t<mara::dimensional_value_t<1, 0, -2, double>, 2>;
    using force_2d_t        = mara::covariant_sequence_t<mara::dimensional_value_t<1, 1, -2, double>, 2>;


    //=========================================================================
    enum class reconstruct_method_t
    {
        plm,
        pcm,
    };


    //=========================================================================
    enum class riemann_solver_t
    {
        hlle,
        hllc,
    };


    //=========================================================================
    struct solver_data_t
    {
        mara::unit_rate  <double>                      sink_rate;
        mara::unit_length<double>                      sink_radius;
        mara::unit_length<double>                      softening_radius;
        mara::unit_time  <double>                      recommended_time_step;

        double                                         plm_theta;
        int                                            rk_order;
        reconstruct_method_t                           reconstruct_method;
        riemann_solver_t                               riemann_solver;
        mara::two_body_parameters_t                    binary_params;

        nd::shared_array<mara::unit_length<double>, 1> x_vertices;
        nd::shared_array<mara::unit_length<double>, 1> y_vertices;
        nd::shared_array<mara::unit_rate  <double>, 2> buffer_damping_rate_field;
        nd::shared_array<mara::iso2d::conserved_t,  2> initial_conserved_field;
    };


    //=========================================================================
    struct solution_state_t
    {
        mara::unit_time<double>                       time = 0.0;
        mara::rational_number_t                       iteration = 0;
        nd::shared_array<mara::iso2d::conserved_t, 2> conserved;

        //=====================================================================
        solution_state_t operator+(const solution_state_t& other) const
        {
            return {
                time       + other.time,
                iteration  + other.iteration,
                conserved  + other.conserved | nd::to_shared(),
            };
        }
        solution_state_t operator*(mara::rational_number_t scale) const
        {
            return {
                time      * scale.as_double(),
                iteration * scale,
                conserved * scale.as_double() | nd::to_shared(),
            };
        }
    };


    //=========================================================================
    struct app_state_t
    {
        solution_state_t solution_state;
        solver_data_t solver_data;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };


    //=========================================================================
    struct diagnostic_fields_t
    {
        mara::unit_time<double> time;
        location_2d_t position_of_mass1;
        location_2d_t position_of_mass2;
        nd::shared_array<mara::unit_length<double>, 1> x_vertices;
        nd::shared_array<mara::unit_length<double>, 1> y_vertices;
        nd::shared_array<double, 2> sigma;
        nd::shared_array<double, 2> phi_velocity;
        nd::shared_array<double, 2> radial_velocity;
    };


    //=========================================================================
    template<typename VertexArrayType>auto cell_surface_area(const VertexArrayType& xv, const VertexArrayType& yv);
    template<typename VertexArrayType>auto cell_center_cartprod(const VertexArrayType& xv, const VertexArrayType& yv);
    auto initial_disk_profile_ring  (const mara::config_t& cfg);
    auto initial_disk_profile_tang17(const mara::config_t& cfg);

    auto diagnostic_fields(const solution_state_t& state, const solver_data_t& solver_data);
    auto gravitational_acceleration_field(mara::unit_time<double> time, const solver_data_t& solver_data);
    auto sink_rate_field(mara::unit_time<double> time, const solver_data_t& solver_data);
    auto buffer_damping_rate_at_position(const mara::config_t& cfg);
    auto estimate_gradient_plm(double plm_theta);
    auto recover_primitive(const mara::iso2d::conserved_per_area_t& conserved);
    auto advance(const solution_state_t& state, const solver_data_t& solver_data, mara::unit_time<double> dt);
    auto next_solution(const solution_state_t& state, const solver_data_t& solver_data);


    //=========================================================================
    void write_solution(h5::Group&& group, const solution_state_t& state);
    void write_diagnostic_fields(h5::Group&& group, const diagnostic_fields_t& diagnostics);
    void write_checkpoint(const app_state_t& state, std::string outdir);
    void write_diagnostics(const app_state_t& state, std::string outdir);
    void write_time_series(const app_state_t& state, std::string outdir);


    //=========================================================================
    void print_run_loop_message(const app_state_t& state, mara::perf_diagnostics_t perf);
    void prepare_filesystem(const mara::config_t& cfg);


    //=========================================================================
    auto read_solution(h5::Group&& group);
    auto new_solution(const mara::config_t& cfg);
    auto create_solution(const mara::config_t& run_config);
    auto create_solver_data(const mara::config_t& cfg);

    auto create_app_state(const mara::config_t& run_config);
    auto create_app_state_next_function(const solver_data_t& solver_data);
    auto simulation_should_continue(const app_state_t& state);
    auto run_tasks(const app_state_t& state);
}

using namespace binary;




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
auto binary::initial_disk_profile_tang17(const mara::config_t& cfg)
{
    auto softening_radius  = cfg.get_double("softening_radius");
    auto mach_number       = cfg.get_double("mach_number");
    auto viscous_alpha     = cfg.get_double("viscous_alpha");
    auto counter_rotate    = cfg.get_int("counter_rotate");

    return [=] (auto x_length, auto y_length)
    {
        auto GM = 1.0;
        auto x  = x_length.value;
        auto y  = y_length.value;

        auto rs             = softening_radius;
        auto r0             = 2.5;
        auto sigma0         = GM;
        auto r2             = x * x + y * y;
        auto r              = std::sqrt(r2);
        auto cavity_xsi     = 10.0;
        auto cavity_cutoff  = std::max(std::exp(-std::pow(r / r0, -cavity_xsi)), 1e-6);
        auto phi            = -GM * std::pow(r2 + rs * rs, -0.5);
        auto ag             = -GM * std::pow(r2 + rs * rs, -1.5) * r;
        auto cs2            = -phi / mach_number / mach_number;
        auto cs2_deriv      =   ag / mach_number / mach_number;
        auto sigma          = sigma0 * std::pow((r + rs) / r0, -0.5) * cavity_cutoff;
        auto sigma_deriv    = sigma0 * std::pow((r + rs) / r0, -1.5) * -0.5 / r0;
        auto dp_dr          = cs2 * sigma_deriv + cs2_deriv * sigma;
        auto omega2         = r < r0 ? GM / (4 * r0) : -ag / r + dp_dr / (sigma * r);
        auto vp             = (counter_rotate ? -1 : 1) * r * std::sqrt(omega2);
        auto h0             = r / mach_number;
        auto nu             = viscous_alpha * std::sqrt(cs2) * h0; // viscous_alpha * cs * h0
        auto vr             = -(3.0 / 2.0) * nu / (r + rs); // inward drift velocity (CHECK)
        auto vx             = vp * (-y / r) + vr * (x / r);
        auto vy             = vp * ( x / r) + vr * (y / r);

        return mara::iso2d::primitive_t()
            .with_sigma(sigma)
            .with_velocity_x(vx)
            .with_velocity_y(vy);
    };
}

auto binary::initial_disk_profile_ring(const mara::config_t& cfg)
{
    auto softening_radius  = cfg.get_double("softening_radius");
    auto counter_rotate    = cfg.get_int("counter_rotate");

    return [=] (auto x_length, auto y_length)
    {
        auto GM             = 1.0;
        auto x              = x_length.value;
        auto y              = y_length.value;
        auto rs             = softening_radius;
        auto rc             = 2.5;
        auto r2             = x * x + y * y;
        auto r              = std::sqrt(r2);
        auto sigma          = std::exp(-std::pow(r - rc, 2) / rc / 2);
        auto ag             = -GM * std::pow(r2 + rs * rs, -1.5) * r;
        auto omega2         = -ag / r;
        auto vp             = (counter_rotate ? -1 : 1) * r * std::sqrt(omega2);
        auto vx             = vp * (-y / r);
        auto vy             = vp * ( x / r);

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
        auto r1 = cfg.get_double("domain_radius");
        return mara::make_rate(1.0 + std::tanh(tightness * (r - r1))) * cfg.get_double("buffer_damping_rate");
    };
}

auto binary::gravitational_acceleration_field(mara::unit_time<double> time, const solver_data_t& solver_data)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, time.value);
    auto softening_radius = solver_data.softening_radius;
    auto accel = [softening_radius] (const mara::point_mass_t& body, auto x, auto y)
    {
        auto field_point   = location_2d_t {{ x, y }};
        auto mass_location = location_2d_t {{ body.position_x, body.position_y }};

        auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
        auto M   = mara::make_mass(body.mass);
        auto dr  = field_point - mass_location;
        auto dr2 = dr[0] * dr[0] + dr[1] * dr[1];
        auto rs2 = softening_radius * softening_radius;
        return -dr / (dr2 + rs2).pow<3, 2>() * G * M;
    };
    auto acceleration = [binary, accel] (auto x, auto y)
    {
        return accel(binary.body1, x, y) + accel(binary.body2, x, y);
    };
    return cell_center_cartprod(solver_data.x_vertices, solver_data.y_vertices) | nd::apply(acceleration);
}

auto binary::sink_rate_field(mara::unit_time<double> time, const solver_data_t& solver_data)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, time.value);
    auto sink = [binary, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (auto x, auto y)
    {
        auto dx1 = x - mara::make_length(binary.body1.position_x);
        auto dy1 = y - mara::make_length(binary.body1.position_y);
        auto dx2 = x - mara::make_length(binary.body2.position_x);
        auto dy2 = y - mara::make_length(binary.body2.position_y);

        auto s2 = sink_radius * sink_radius;
        auto a2 = (dx1 * dx1 + dy1 * dy1) / s2 / 2.0;
        auto b2 = (dx2 * dx2 + dy2 * dy2) / s2 / 2.0;

        return sink_rate * 0.5 * (std::exp(-a2) + std::exp(-b2));
    };
    return cell_center_cartprod(solver_data.x_vertices, solver_data.y_vertices) | nd::apply(sink);
}

auto binary::estimate_gradient_plm(double plm_theta)
{
    return [plm_theta] (
        const mara::iso2d::primitive_t& pl,
        const mara::iso2d::primitive_t& p0,
        const mara::iso2d::primitive_t& pr)
    {
        using std::min;
        using std::fabs;
        auto min3abs = [] (auto a, auto b, auto c) { return min(min(fabs(a), fabs(b)), fabs(c)); };
        auto sgn = [] (auto x) { return std::copysign(1, x); };

        // VARIABLE THETA IS DISABLED
        // --------------------------
        // auto clamp = [plm_theta] (double x) { return std::max(1.0, std::min(x, plm_theta)); };
        // double s0 = log10_sigma_diffusive_below;
        // double s1 = log10_sigma_aggressive_above;
        // double th = clamp(1.0 + (std::log10(p0.sigma()) - s0) / (s1 - s0));

        double th = plm_theta;

        auto result = mara::iso2d::primitive_t();

        for (std::size_t i = 0; i < 3; ++i)
        {
            auto a =  th * (p0[i] - pl[i]);
            auto b = 0.5 * (pr[i] - pl[i]);
            auto c =  th * (pr[i] - p0[i]);
            result[i] = 0.25 * std::fabs(sgn(a) + sgn(b)) * (sgn(a) + sgn(c)) * min3abs(a, b, c);
        }
        return result;
    };
}

auto binary::recover_primitive(const mara::iso2d::conserved_per_area_t& conserved)
{
    return mara::iso2d::recover_primitive(conserved, density_floor);
}

auto binary::advance(const solution_state_t& state, const solver_data_t& solver_data, mara::unit_time<double> dt)
{
    auto force_to_source_terms = [] (force_2d_t v)
    {
        return mara::iso2d::flow_t {{0.0, v[0].value, v[1].value}};
    };

    auto extrapolate_pcm = [] (std::size_t axis)
    {
        return [axis] (auto P)
        {
            auto L = nd::select_axis(axis).from(0).to(1).from_the_end();
            auto R = nd::select_axis(axis).from(1).to(0).from_the_end();
            return nd::zip_arrays(P | L, P | R);
        };
    };

    auto extrapolate_plm = [plm_theta=solver_data.plm_theta] (std::size_t axis)
    {
        return [plm_theta, axis] (auto P)
        {
            auto L = nd::select_axis(axis).from(0).to(1).from_the_end();
            auto R = nd::select_axis(axis).from(1).to(0).from_the_end();
            auto G = P
            | nd::zip_adjacent3_on_axis(axis)
            | nd::apply(estimate_gradient_plm(plm_theta))
            | nd::extend_zeros(axis)
            | nd::to_shared();

            return nd::zip_arrays(
                (P | L) + (G | L) * 0.5,
                (P | R) - (G | R) * 0.5);
        };
    };

    auto intercell_flux = [] (auto riemann_solver, std::size_t axis)
    {
        return [axis, riemann_solver] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(riemann_solver, _1, _2, sound_speed_squared, sound_speed_squared, nh);
            return left_and_right_states | nd::apply(riemann);
        };
    };

    auto advance_with = [&] (auto riemann_solver, auto extrapolate)
    {
        auto fhat_x = intercell_flux(riemann_solver, 0);
        auto fhat_y = intercell_flux(riemann_solver, 1);

        auto dA = cell_surface_area(solver_data.x_vertices, solver_data.y_vertices);
        auto u0 = state.conserved;
        auto p0 = u0 / dA | nd::map(recover_primitive) | nd::to_shared();
        auto dx = nd::get<0>(nd::cartesian_product(solver_data.x_vertices | nd::difference_on_axis(0), solver_data.y_vertices));
        auto dy = nd::get<1>(nd::cartesian_product(solver_data.x_vertices, solver_data.y_vertices | nd::difference_on_axis(0)));
        auto cell_mass = u0 | nd::map([] (auto u) { return u[0]; });

        auto lx = p0 | nd::extend_periodic_on_axis(0) | extrapolate(0) | fhat_x | nd::multiply(-dy) | nd::difference_on_axis(0);
        auto ly = p0 | nd::extend_periodic_on_axis(1) | extrapolate(1) | fhat_y | nd::multiply(-dx) | nd::difference_on_axis(1);
        auto sg = gravitational_acceleration_field(state.time, solver_data) | nd::multiply(cell_mass) | nd::map(force_to_source_terms);
        auto ss = -u0 * sink_rate_field(state.time, solver_data);
        auto bz = (solver_data.initial_conserved_field - u0) * solver_data.buffer_damping_rate_field;
        auto u1 = u0 + (lx + ly + sg + ss + bz) * dt;
        return u1 | nd::to_shared();
    };

    auto next_state = solution_state_t();
    next_state.iteration = state.iteration + 1;
    next_state.time      = state.time + dt;

    switch (solver_data.reconstruct_method)
    {
        case reconstruct_method_t::pcm:
            switch (solver_data.riemann_solver)
            {
                case riemann_solver_t::hlle: next_state.conserved = advance_with(mara::iso2d::riemann_hlle, extrapolate_pcm); break;
                case riemann_solver_t::hllc: next_state.conserved = advance_with(mara::iso2d::riemann_hllc, extrapolate_pcm); break;
            }
            break;
        case reconstruct_method_t::plm:
            switch (solver_data.riemann_solver)
            {
                case riemann_solver_t::hlle: next_state.conserved = advance_with(mara::iso2d::riemann_hlle, extrapolate_plm); break;
                case riemann_solver_t::hllc: next_state.conserved = advance_with(mara::iso2d::riemann_hllc, extrapolate_plm); break;
            }
            break;
    }
    return next_state;
}

auto binary::next_solution(const solution_state_t& state, const solver_data_t& solver_data)
{
    auto dt = solver_data.recommended_time_step;
    auto s0 = state;

    switch (solver_data.rk_order)
    {
        case 1:
        {
            return advance(s0, solver_data, dt);
        }
        case 2:
        {
            auto b0 = mara::make_rational(1, 2);
            auto s1 = advance(s0, solver_data, dt);
            auto s2 = advance(s1, solver_data, dt);
            return s0 * b0 + s2 * (1 - b0);
        }
    }
    throw std::invalid_argument("binary::next_solution");
}




//=============================================================================
auto binary::diagnostic_fields(const solution_state_t& state, const solver_data_t& solver_data)
{
    auto dA = cell_surface_area(solver_data.x_vertices, solver_data.y_vertices);
    auto [xc, yc] = nd::unzip_array(cell_center_cartprod(solver_data.x_vertices, solver_data.y_vertices));
    auto rc = xc * xc + yc * yc | nd::map([] (auto r2) { return mara::make_length(std::sqrt(r2.value)); });
    auto rhat_x =  xc / rc;
    auto rhat_y =  yc / rc;
    auto phat_x = -yc / rc;
    auto phat_y =  xc / rc;
    auto u = state.conserved;
    auto p = u / dA | nd::map(recover_primitive);
    auto sigma = p | nd::map(std::mem_fn(&mara::iso2d::primitive_t::sigma));
    auto vx = p | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_x));
    auto vy = p | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_y));
    auto binary = mara::compute_two_body_state(solver_data.binary_params, state.time.value);

    auto result = diagnostic_fields_t();
    result.time            = state.time;
    result.x_vertices      = solver_data.x_vertices;
    result.y_vertices      = solver_data.y_vertices;
    result.sigma           = sigma                     | nd::to_shared();
    result.radial_velocity = vx * rhat_x + vy * rhat_y | nd::to_shared();
    result.phi_velocity    = vx * phat_x + vy * phat_y | nd::to_shared();
    result.position_of_mass1 = location_2d_t {{ binary.body1.position_x, binary.body1.position_y }};
    result.position_of_mass2 = location_2d_t {{ binary.body2.position_x, binary.body2.position_y }};

    return result;
}




//=============================================================================
void binary::write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
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
    group.write("position_of_mass1", diagnostics.position_of_mass1);
    group.write("position_of_mass2", diagnostics.position_of_mass2);
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
    auto diagnostics = diagnostic_fields(state.solution_state, state.solver_data);

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
auto binary::create_solver_data(const mara::config_t& cfg)
{
    auto nx = cfg.get_int("N");
    auto ny = cfg.get_int("N");
    auto R0 = cfg.get_double("domain_radius");
    auto xv = nd::linspace(-R0, R0, nx + 1) | nd::map(mara::make_length<double>);
    auto yv = nd::linspace(-R0, R0, ny + 1) | nd::map(mara::make_length<double>);

    auto u = cell_center_cartprod(xv, yv)
    | nd::apply(initial_disk_profile_ring(cfg))
    | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area))
    | nd::multiply(cell_surface_area(xv, yv));

    auto b = cell_center_cartprod(xv, yv)
    | nd::apply(buffer_damping_rate_at_position(cfg));

    auto dA = cell_surface_area(xv, yv);
    auto p0 = u / dA | nd::map(recover_primitive);
    auto v0 = p0 | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_magnitude));
    auto dx = mara::make_length(2 * R0) / std::max(nx, ny);
    auto dt = dx / nd::max(v0) * 0.5 * cfl_number;

    auto result = solver_data_t();
    result.sink_rate             = cfg.get_double("sink_rate");
    result.sink_radius           = cfg.get_double("sink_radius");
    result.softening_radius      = cfg.get_double("softening_radius");
    result.plm_theta             = cfg.get_double("plm_theta");
    result.rk_order              = cfg.get_int("rk_order");
    result.recommended_time_step = dt;

    result.x_vertices = xv | nd::to_shared();
    result.y_vertices = yv | nd::to_shared();
    result.initial_conserved_field   = u | nd::to_shared();
    result.buffer_damping_rate_field = b | nd::to_shared();

    if      (cfg.get_string("riemann") == "hlle") result.riemann_solver = riemann_solver_t::hlle;
    else if (cfg.get_string("riemann") == "hllc") result.riemann_solver = riemann_solver_t::hllc;
    else throw std::invalid_argument("invalid riemann solver '" + cfg.get_string("riemann") + "', must be hlle or hllc");

    if      (cfg.get_string("reconstruct_method") == "pcm") result.reconstruct_method = reconstruct_method_t::pcm;
    else if (cfg.get_string("reconstruct_method") == "plm") result.reconstruct_method = reconstruct_method_t::plm;
    else throw std::invalid_argument("invalid reconstruct_method '" + cfg.get_string("reconstruct_method") + "', must be plm or pcm");

    result.binary_params.total_mass   = 1.0;
    result.binary_params.separation   = cfg.get_double("separation");
    result.binary_params.mass_ratio   = cfg.get_double("mass_ratio");
    result.binary_params.eccentricity = cfg.get_double("eccentricity");

    return result;
}

auto binary::read_solution(h5::Group&& group)
{
    return solution_state_t(); // IMPLEMENT READING SOLUTION FROM CHECKPOINT
}

auto binary::new_solution(const mara::config_t& cfg)
{
    auto state = solution_state_t();
    state.time = 0.0;
    state.iteration = 0;
    state.conserved = create_solver_data(cfg).initial_conserved_field;
    return state;
}

auto binary::create_solution(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");
    return restart.empty()
    ? new_solution(run_config)
    : read_solution(h5::File(restart, "r").open_group("solution"));
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
    auto cpi = run_config.get_double("cpi") * 2 * M_PI;
    auto dfi = run_config.get_double("dfi") * 2 * M_PI;
    auto tsi = run_config.get_double("tsi") * 2 * M_PI;

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
auto binary::create_app_state(const mara::config_t& run_config)
{
    auto state = app_state_t();
    state.run_config     = run_config;
    state.solution_state = create_solution(run_config);
    state.schedule       = create_schedule(run_config);
    state.solver_data    = create_solver_data(run_config);
    return state;
}

auto binary::create_app_state_next_function(const solver_data_t& solver_data)
{
    return [solver_data] (const app_state_t& state)
    {
        auto next_state = state;
        next_state.solution_state = next_solution(state.solution_state, solver_data);
        next_state.schedule       = next_schedule(state.schedule, state.run_config, state.solution_state.time.value);
        return next_state;
    };
}

auto binary::simulation_should_continue(const app_state_t& state)
{
    auto orbits = state.solution_state.time / (2 * M_PI);
    return orbits < state.run_config.get<double>("tfinal");
}

auto binary::run_tasks(const app_state_t& state)
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
void binary::print_run_loop_message(const app_state_t& state, mara::perf_diagnostics_t perf)
{
    auto kzps =
    state.solver_data.x_vertices.size() *
    state.solver_data.y_vertices.size() / perf.execution_time_ms;

    std::printf("[%04d] orbits=%3.7lf kzps=%3.2lf\n",
        state.solution_state.iteration.as_integral(),
        state.solution_state.time.value / (2 * M_PI), kzps);
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
        auto run_config  = create_run_config(argc, argv);
        auto state       = create_app_state(run_config);
        auto next        = create_app_state_next_function(state.solver_data);
        auto perf        = mara::perf_diagnostics_t();

        prepare_filesystem(run_config);
        mara::pretty_print(std::cout, "config", run_config);
        state = run_tasks(state);

        while (simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(mara::compose(run_tasks, next), state);
            print_run_loop_message(state, perf);
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
