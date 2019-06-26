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
#include "core_tree.hpp"
#include "mesh_prolong_restrict.hpp"
#include "mesh_tree_operators.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"
#include "app_performance.hpp"
#include "app_schedule.hpp"
#include "app_serialize.hpp"
#include "app_serialize_tree.hpp"
#include "app_subprogram.hpp"
#include "physics_iso2d.hpp"
#include "model_two_body.hpp"
#define cfl_number                    0.4
#define density_floor                 0.0
#define sound_speed_squared          1e-4




//=============================================================================
namespace binary
{


    //=========================================================================
    using location_2d_t  = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t  = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using accel_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -2, double>, 2>;
    using force_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 1, -2, double>, 2>;

    template<typename ArrayValueType>
    using quad_tree_t = mara::arithmetic_binary_tree_t<nd::shared_array<ArrayValueType, 2>, 2>;


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
        quad_tree_t<location_2d_t>                     vertices;
    };


    //=========================================================================
    struct solution_t
    {
        mara::unit_time<double>                        time = 0.0;
        mara::rational_number_t                        iteration = 0;
        quad_tree_t<mara::iso2d::conserved_per_area_t> conserved;
        solution_t operator+(const solution_t& other) const;
        solution_t operator*(mara::rational_number_t scale) const;
    };


    //=========================================================================
    struct state_t
    {
        solution_t solution;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };


    //=========================================================================
    struct diagnostic_fields_t
    {
        mara::config_t                                 run_config;
        mara::unit_time<double>                        time;
        quad_tree_t<location_2d_t>                     vertices;
        quad_tree_t<double>                            sigma;
        quad_tree_t<double>                            radial_velocity;
        quad_tree_t<double>                            phi_velocity;
        location_2d_t                                  position_of_mass1;
        location_2d_t                                  position_of_mass2;
    };


    //=========================================================================
    auto config_template();
    auto initial_disk_profile(const mara::config_t& run_config);
    auto gravitational_acceleration_field(mara::unit_time<double> time, const solver_data_t& solver_data);
    auto sink_rate_field(mara::unit_time<double> time, const solver_data_t& solver_data);


    //=========================================================================
    auto create_run_config(int argc, const char* argv[]);
    auto create_vertices     (const mara::config_t& run_config);
    auto create_binary_params(const mara::config_t& run_config);
    auto create_solver_data  (const mara::config_t& run_config);
    auto create_solution     (const mara::config_t& run_config);
    auto create_schedule     (const mara::config_t& run_config);
    auto create_state        (const mara::config_t& run_config);


    //=========================================================================
    auto next_solution(const solution_t& solution, const solver_data_t& solver_data);
    auto next_schedule(const state_t& state);
    auto next_state(const state_t& state, const solver_data_t& solver_data);


    //=========================================================================
    auto run_tasks(const state_t& state);
    auto simulation_should_continue(const state_t& state);
    auto diagnostic_fields(const solution_t& solution, const mara::config_t& run_config);
    void prepare_filesystem(const mara::config_t& run_config);
    void print_run_loop_message(const state_t& state, mara::perf_diagnostics_t perf);
}




//=============================================================================
namespace mara
{
    template<> void write<binary::solution_t>         (h5::Group&, std::string, const binary::solution_t&);
    template<> void write<binary::state_t>            (h5::Group&, std::string, const binary::state_t&);
    template<> void write<binary::diagnostic_fields_t>(h5::Group&, std::string, const binary::diagnostic_fields_t&);
    template<> void read<binary::solution_t>          (h5::Group&, std::string, binary::solution_t&);
    template<> void read<binary::state_t>             (h5::Group&, std::string, binary::state_t&);
    template<> void read<binary::diagnostic_fields_t> (h5::Group&, std::string, binary::diagnostic_fields_t&);
}




//=============================================================================
auto binary::config_template()
{
    return mara::make_config_template()
    .item("restart",             std::string())
    .item("outdir",              "data")        // directory where data products are written to
    .item("cpi",                 10.0)          // checkpoint interval (orbits; chkpt.????.h5 - snapshot of app_state)
    .item("dfi",                  1.0)          // diagnostic field interval (orbits; diagnostics.????.h5)
    .item("tsi",                  0.1)          // time series interval (orbits)
    .item("tfinal",               1.0)          // simulation stop time (orbits)
    .item("depth",                  4)
    .item("block_size",            32)
    .item("focus_factor",        0.75)
    .item("rk_order",               2)          // time-stepping Runge-Kutta order: 1 or 2
    .item("reconstruct_method", "plm")          // zone extrapolation method: pcm or plm
    .item("plm_theta",            1.8)          // plm theta parameter: [1.0, 2.0]
    .item("riemann",           "hllc")          // riemann solver to use: hlle or hllc
    .item("softening_radius",     0.1)          // gravitational softening radius
    .item("sink_radius",          0.1)          // radius of mass (and momentum) subtraction region
    .item("sink_rate",            1e2)          // sink rate at the point masses (orbital angular frequency)
    .item("domain_radius",       48.0)          // half-size of square domain
    .item("separation",           1.0)          // binary separation: 0.0 or 1.0 (zero emulates a single body)
    .item("mass_ratio",           1.0)          // binary mass ratio M2 / M1: (0.0, 1.0]
    .item("eccentricity",         0.0)          // orbital eccentricity: [0.0, 1.0)
    .item("buffer_damping_rate", 10.0)          // maximum rate of buffer zone, where solution is driven to initial state
    .item("counter_rotate",         0);         // retrograde disk option: 0 or 1
}




//=============================================================================
auto binary::initial_disk_profile(const mara::config_t& run_config)
{
    auto softening_radius  = run_config.get_double("softening_radius");
    auto counter_rotate    = run_config.get_int("counter_rotate");

    return [=] (location_2d_t point)
    {
        auto GM             = 1.0;
        auto x              = point[0].value;
        auto y              = point[1].value;
        auto rs             = softening_radius;
        auto rc             = 2.5;
        auto r2             = x * x + y * y;
        auto r              = std::sqrt(r2);
        auto sigma          = std::exp(-std::min(5.0, std::pow(r - rc, 2) / rc / 2));
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

auto binary::gravitational_acceleration_field(mara::unit_time<double> time, const solver_data_t& solver_data)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, time.value);
    auto accel = [softening_radius=solver_data.softening_radius] (const mara::point_mass_t& body, location_2d_t field_point)
    {
        auto mass_location = location_2d_t { body.position_x, body.position_y };

        auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
        auto M   = mara::make_mass(body.mass);
        auto dr  = field_point - mass_location;
        auto dr2 = dr[0] * dr[0] + dr[1] * dr[1];
        auto rs2 = softening_radius * softening_radius;
        return -dr / (dr2 + rs2).pow<3, 2>() * G * M;
    };

    auto acceleration = [binary, accel] (location_2d_t p)
    {
        return accel(binary.body1, p) + accel(binary.body2, p);
    };

    return solver_data.vertices.map([acceleration] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(acceleration);
    });
}

auto binary::sink_rate_field(mara::unit_time<double> time, const solver_data_t& solver_data)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, time.value);
    auto sink = [binary, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (location_2d_t p)
    {
        auto dx1 = p[0] - mara::make_length(binary.body1.position_x);
        auto dy1 = p[1] - mara::make_length(binary.body1.position_y);
        auto dx2 = p[0] - mara::make_length(binary.body2.position_x);
        auto dy2 = p[1] - mara::make_length(binary.body2.position_y);

        auto s2 = sink_radius * sink_radius;
        auto a2 = (dx1 * dx1 + dy1 * dy1) / s2 / 2.0;
        auto b2 = (dx2 * dx2 + dy2 * dy2) / s2 / 2.0;

        return sink_rate * 0.5 * (std::exp(-a2) + std::exp(-b2));
    };

    return solver_data.vertices.map([sink] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(sink);
    });
}




//=========================================================================
template<>
void mara::write<binary::solution_t>(h5::Group& group, std::string name, const binary::solution_t& solution)
{
    auto location = group.require_group(name);
    mara::write(location, "time",       solution.time);
    mara::write(location, "iteration",  solution.iteration);
    mara::write(location, "conserved",  solution.conserved);
}

template<>
void mara::write<binary::state_t>(h5::Group& group, std::string name, const binary::state_t& state)
{
    auto location = group.require_group(name);
    mara::write(location, "solution",   state.solution);
    mara::write(location, "schedule",   state.schedule);
    mara::write(location, "run_config", state.run_config);
}

template<>
void mara::write<binary::diagnostic_fields_t>(h5::Group& group, std::string name, const binary::diagnostic_fields_t& diagnostics)
{
    auto location = group.require_group(name);
    mara::write(location, "run_config",        diagnostics.run_config);
    mara::write(location, "time",              diagnostics.time);
    mara::write(location, "vertices",          diagnostics.vertices);
    mara::write(location, "sigma",             diagnostics.sigma);
    mara::write(location, "radial_velocity",   diagnostics.radial_velocity);
    mara::write(location, "phi_velocity",      diagnostics.phi_velocity);
    mara::write(location, "position_of_mass1", diagnostics.position_of_mass1);
    mara::write(location, "position_of_mass2", diagnostics.position_of_mass2);
}

template<>
void mara::read<binary::solution_t>(h5::Group& group, std::string name, binary::solution_t& solution)
{
    auto location = group.open_group(name);
    mara::read(location, "time",       solution.time);
    mara::read(location, "iteration",  solution.iteration);
    mara::read(location, "conserved",  solution.conserved);
}

template<>
void mara::read<binary::state_t>(h5::Group& group, std::string name, binary::state_t& state)
{
    auto location = group.open_group(name);
    mara::read(location, "solution",   state.solution);
    mara::read(location, "schedule",   state.schedule);
}

template<>
void mara::read<binary::diagnostic_fields_t>(h5::Group& group, std::string name, binary::diagnostic_fields_t& diagnostics)
{
    auto location = group.open_group(name);
    // mara::read(location, "run_config",        diagnostics.run_config);
    mara::read(location, "time",              diagnostics.time);
    mara::read(location, "vertices",          diagnostics.vertices);
    mara::read(location, "sigma",             diagnostics.sigma);
    mara::read(location, "radial_velocity",   diagnostics.radial_velocity);
    mara::read(location, "phi_velocity",      diagnostics.phi_velocity);
    mara::read(location, "position_of_mass1", diagnostics.position_of_mass1);
    mara::read(location, "position_of_mass2", diagnostics.position_of_mass2);
}




//=============================================================================
binary::solution_t binary::solution_t::operator+(const solution_t& other) const
{
    return {
        time       + other.time,
        iteration  + other.iteration,
        (conserved + other.conserved).map(nd::to_shared()),
    };
}

binary::solution_t binary::solution_t::operator*(mara::rational_number_t scale) const
{
    return {
        time       * scale.as_double(),
        iteration  * scale,
        (conserved * scale.as_double()).map(nd::to_shared()),
    };
}




//=============================================================================
auto binary::create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);
    return args.count("restart")
    ? config_template()
            .create()
            .update(mara::read_config(h5::File(args.at("restart"), "r").open_group("run_config")))
            .update(args)
    : config_template().create().update(args);
}

auto binary::create_vertices(const mara::config_t& run_config)
{
    auto domain_radius = run_config.get_double("domain_radius");
    auto focus_factor  = run_config.get_double("focus_factor");
    auto block_size    = run_config.get_int("block_size");
    auto depth         = run_config.get_int("depth");

    auto refinement_radius = [focus_factor] (std::size_t level, double centroid_radius)
    {
        return centroid_radius < focus_factor / level;
    };
    auto verts = mara::create_vertex_quadtree(refinement_radius, block_size, depth);

    return verts.map([domain_radius] (auto block)
    {
        return (block * domain_radius).shared();
    });
}

auto binary::create_binary_params(const mara::config_t& run_config)
{
    auto binary = mara::two_body_parameters_t();
    binary.total_mass   = 1.0;
    binary.separation   = run_config.get_double("separation");
    binary.mass_ratio   = run_config.get_double("mass_ratio");
    binary.eccentricity = run_config.get_double("eccentricity");
    return binary;
}

auto binary::create_solver_data(const mara::config_t& run_config)
{
    //=========================================================================
    auto vertices = create_vertices(run_config);

    auto primitive = vertices.map([&run_config] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(initial_disk_profile(run_config));
    });

    auto min_dx = vertices.map([] (auto block)
    {
        return block
        | nd::map([] (auto p) { return p[0]; })
        | nd::difference_on_axis(0)
        | nd::min();
    }).min();

    auto min_dy = vertices.map([] (auto block)
    {
        return block
        | nd::map([] (auto p) { return p[1]; })
        | nd::difference_on_axis(1)
        | nd::min();
    }).min();

    auto max_velocity = std::max(mara::make_velocity(1.0), primitive.map([] (auto block)
    {
        return block
        | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_magnitude))
        | nd::max();
    }).max());


    //=========================================================================
    auto result = solver_data_t();
    result.sink_rate             = run_config.get_double("sink_rate");
    result.sink_radius           = run_config.get_double("sink_radius");
    result.softening_radius      = run_config.get_double("softening_radius");
    result.plm_theta             = run_config.get_double("plm_theta");
    result.rk_order              = run_config.get_int("rk_order");
    result.recommended_time_step = std::min(min_dx, min_dy) / max_velocity * cfl_number;
    result.binary_params         = create_binary_params(run_config);
    result.vertices              = vertices;

    if      (run_config.get_string("riemann") == "hlle") result.riemann_solver = riemann_solver_t::hlle;
    else if (run_config.get_string("riemann") == "hllc") result.riemann_solver = riemann_solver_t::hllc;
    else throw std::invalid_argument("invalid riemann solver '" + run_config.get_string("riemann") + "', must be hlle or hllc");

    if      (run_config.get_string("reconstruct_method") == "pcm") result.reconstruct_method = reconstruct_method_t::pcm;
    else if (run_config.get_string("reconstruct_method") == "plm") result.reconstruct_method = reconstruct_method_t::plm;
    else throw std::invalid_argument("invalid reconstruct_method '" + run_config.get_string("reconstruct_method") + "', must be plm or pcm");

    return result;
}

auto binary::create_solution(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");

    if (restart.empty())
    {
        auto conserved = create_vertices(run_config).map([&run_config] (auto block)
        {
            return block
            | nd::midpoint_on_axis(0)
            | nd::midpoint_on_axis(1)
            | nd::map(initial_disk_profile(run_config))
            | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area))
            | nd::to_shared();
        });
        return solution_t{0, 0.0, conserved};
    }
    return mara::read<solution_t>(h5::File(restart, "r").open_group("/"), "solution");
}

auto binary::create_schedule(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");

    if (restart.empty())
    {
        auto schedule = mara::schedule_t();
        schedule.create_and_mark_as_due("write_checkpoint");
        schedule.create_and_mark_as_due("write_diagnostics");
        schedule.create_and_mark_as_due("write_time_series");
        return schedule;
    }
    return mara::read_schedule(h5::File(restart, "r").open_group("schedule"));
}

auto binary::create_state(const mara::config_t& run_config)
{
    return state_t{
        create_solution(run_config),
        create_schedule(run_config),
        run_config,
    };
}




//=============================================================================
auto binary::next_solution(const solution_t& solution, const solver_data_t& solver_data)
{
    auto component = [] (std::size_t component)
    {
        return nd::map([component] (auto p) { return p[component]; });
    };

    auto extend = [] (auto tree, std::size_t axis)
    {
        return tree.indexes().map([tree, axis] (auto index)
        {
            auto C = mara::get_cell_block(tree, index);
            auto L = mara::get_cell_block(tree, index.prev_on(axis)) | nd::select_final(1, axis);
            auto R = mara::get_cell_block(tree, index.next_on(axis)) | nd::select_first(1, axis);
            return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis);
        });
    };

    auto area_from_vertices = [component] (auto vertices)
    {
        auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };

    auto extrapolate_pcm = [] (std::size_t axis)
    {
        return [axis] (auto P)
        {
            auto L = nd::select_axis(axis).from(0).to(1).from_the_end();
            auto R = nd::select_axis(axis).from(1).to(0).from_the_end();
            return nd::zip(P | L, P | R);
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

    auto force_to_source_terms = [] (force_2d_t v)
    {
        return mara::iso2d::flow_t{0.0, v[0].value, v[1].value};
    };

    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, density_floor);
    auto dt = solver_data.recommended_time_step;
    auto v0 = solver_data.vertices;
    auto u0 = solution.conserved;
    auto p0 = u0.map(nd::map(recover_primitive)).map(nd::to_shared());
    auto dA = v0.map(area_from_vertices).map(nd::to_shared());
    auto dx = v0.map(component(0)).map(nd::difference_on_axis(0)).map(nd::to_shared());
    auto dy = v0.map(component(1)).map(nd::difference_on_axis(1)).map(nd::to_shared());
    auto fx = extend(p0, 0).map(extrapolate_pcm(0)).map(intercell_flux(mara::iso2d::riemann_hlle, 0)) * dy;
    auto fy = extend(p0, 1).map(extrapolate_pcm(1)).map(intercell_flux(mara::iso2d::riemann_hlle, 1)) * dx;
    auto lx = -fx.map(nd::difference_on_axis(0));
    auto ly = -fy.map(nd::difference_on_axis(1));
    auto m0 = u0.map(component(0)) * dA; // cell masses
    auto sg = (gravitational_acceleration_field(solution.time, solver_data) * m0).map(nd::map(force_to_source_terms));
    auto ss = -u0 * sink_rate_field(solution.time, solver_data) * dA;
    auto u1 = u0 + (lx + ly + ss + sg) * dt / dA;

    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        u1.map(nd::to_shared()),
    };

    // These are the same ops as the ones above, but create fewer intermediate trees:
    // auto dx = v0.map([component] (auto v) { return v | component(0) | nd::difference_on_axis(0) | nd::to_shared(); });
    // auto dy = v0.map([component] (auto v) { return v | component(1) | nd::difference_on_axis(1) | nd::to_shared(); });
}

auto binary::next_schedule(const state_t& state)
{
    return mara::mark_tasks_in(state, state.solution.time.value,
        {{"write_checkpoint",  state.run_config.get_double("cpi") * 2 * M_PI},
         {"write_diagnostics", state.run_config.get_double("dfi") * 2 * M_PI},
         {"write_time_series", state.run_config.get_double("tsi") * 2 * M_PI}});
}

auto binary::next_state(const state_t& state, const solver_data_t& solver_data)
{
    return state_t{
        next_solution(state.solution, solver_data),
        next_schedule(state),
        state.run_config,
    };
}




//=============================================================================
auto binary::simulation_should_continue(const state_t& state)
{
    return state.solution.time / (2 * M_PI) < state.run_config.get<double>("tfinal");
}

auto binary::diagnostic_fields(const solution_t& solution, const mara::config_t& run_config)
{
    auto solver_data = create_solver_data(run_config);
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);

    auto component = [] (std::size_t component)
    {
        return nd::map([component] (auto p) { return p[component]; });
    };

    auto area_from_vertices = [component] (auto vertices)
    {
        auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };

    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, density_floor);
    auto v0 = solver_data.vertices;
    auto c0 = v0.map(nd::midpoint_on_axis(0)).map(nd::midpoint_on_axis(1));
    auto xc = c0.map(component(0));
    auto yc = c0.map(component(1));
    auto u0 = solution.conserved;
    auto p0 = u0.map(nd::map(recover_primitive)).map(nd::to_shared());
    auto dA = v0.map(area_from_vertices).map(nd::to_shared());

    auto rc = (xc * xc + yc * yc).map(nd::map([] (mara::unit_area<double> r2) { return r2.pow<1, 2>(); }));
    auto rhat_x =  xc / rc;
    auto rhat_y =  yc / rc;
    auto phat_x = -yc / rc;
    auto phat_y =  xc / rc;
    auto sigma = p0.map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::sigma)));
    auto vx    = p0.map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_x)));
    auto vy    = p0.map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_y)));
    auto vr    = vx * rhat_x + vy * rhat_y;
    auto vp    = vx * phat_x + vy * phat_y;

    return diagnostic_fields_t{
        run_config,
        solution.time,
        v0,
        sigma.map(nd::to_shared()),
        vr   .map(nd::to_shared()),
        vp   .map(nd::to_shared()),
        {binary.body1.position_x, binary.body1.position_y},
        {binary.body2.position_x, binary.body2.position_y},
    };
}




//=============================================================================
auto binary::run_tasks(const state_t& state)
{


    //=========================================================================
    auto write_checkpoint  = [] (const state_t& state)
    {
        auto outdir = state.run_config.get_string("outdir");
        auto count  = state.schedule.num_times_performed("write_checkpoint");
        auto fname  = mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", count, "h5"));
        auto group  = h5::File(fname, "w").open_group("/");
        mara::write(group, "/", state);
        std::printf("write checkpoint: %s\n", fname.data());
        return mara::complete_task_in(state, "write_checkpoint");
    };


    //=========================================================================
    auto write_diagnostics = [] (const state_t& state)
    {
        auto outdir = state.run_config.get_string("outdir");
        auto count  = state.schedule.num_times_performed("write_diagnostics");
        auto fname  = mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5"));
        auto group  = h5::File(fname, "w").open_group("/");
        mara::write(group, "/", diagnostic_fields(state.solution, state.run_config));
        std::printf("write diagnostics: %s\n", fname.data());
        return mara::complete_task_in(state, "write_diagnostics");
    };


    //=========================================================================
    auto write_time_series = [] (const state_t& state)
    {
        // TODO
        return mara::complete_task_in(state, "write_time_series");
    };


    return mara::run_scheduled_tasks(state, {
        {"write_checkpoint",  write_checkpoint},
        {"write_diagnostics", write_diagnostics},
        {"write_time_series", write_time_series}});
}

void binary::prepare_filesystem(const mara::config_t& run_config)
{
    auto outdir = run_config.get_string("outdir");
    mara::filesystem::require_dir(outdir);
}

void binary::print_run_loop_message(const state_t& state, mara::perf_diagnostics_t perf)
{
    auto kzps = state
    .solution
    .conserved
    .map([] (auto&& block) { return block.size(); })
    .sum() / perf.execution_time_ms;

    std::printf("[%04d] orbits=%3.7lf kzps=%3.2lf\n",
        state.solution.iteration.as_integral(),
        state.solution.time.value / (2 * M_PI), kzps);
}




//=============================================================================
class subprog_binary : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto run_config  = binary::create_run_config(argc, argv);
        auto solver_data = binary::create_solver_data(run_config);
        auto state       = binary::create_state(run_config);
        auto next        = std::bind(binary::next_state, std::placeholders::_1, solver_data);
        auto perf        = mara::perf_diagnostics_t();

        binary::prepare_filesystem(run_config);
        mara::pretty_print(std::cout, "config", run_config);
        state = binary::run_tasks(state);

        while (binary::simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(mara::compose(binary::run_tasks, next), state);
            binary::print_run_loop_message(state, perf);
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
