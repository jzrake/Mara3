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
#if MARA_COMPILE_SUBPROGRAM_BINARY_FMR




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
namespace binary_fmr
{


    //=========================================================================
    using location_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using acceleration_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -2, double>, 2>;
    using force_2d_t        = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 1, -2, double>, 2>;

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

        nd::shared_array<mara::unit_length<double>, 1> x_vertices;
        nd::shared_array<mara::unit_length<double>, 1> y_vertices;
        nd::shared_array<mara::unit_rate  <double>, 2> buffer_damping_rate_field;
        nd::shared_array<mara::iso2d::conserved_t,  2> initial_conserved_field;
    };


    //=========================================================================
    struct solution_t
    {
        mara::unit_time<double>                        time = 0.0;
        mara::rational_number_t                        iteration = 0;
        quad_tree_t<mara::iso2d::conserved_per_area_t> conserved;

        //=====================================================================
        solution_t operator+(const solution_t& other) const
        {
            return {
                time       + other.time,
                iteration  + other.iteration,
                (conserved + other.conserved).map(nd::to_shared()),
            };
        }
        solution_t operator*(mara::rational_number_t scale) const
        {
            return {
                time       * scale.as_double(),
                iteration  * scale,
                (conserved * scale.as_double()).map(nd::to_shared()),
            };
        }
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
        quad_tree_t<mara::iso2d::conserved_per_area_t> conserved;
        location_2d_t                                  position_of_mass1;
        location_2d_t                                  position_of_mass2;
        // nd::shared_array<double, 2>                    sigma;
        // nd::shared_array<double, 2>                    phi_velocity;
        // nd::shared_array<double, 2>                    radial_velocity;
    };


    //=========================================================================
    auto config_template();
    auto initial_disk_profile(const mara::config_t& cfg);


    //=========================================================================
    auto create_run_config(int argc, const char* argv[]);
    auto create_vertices(const mara::config_t& run_config);
    auto create_solution(const mara::config_t& run_config);
    auto create_solver_data(const mara::config_t& cfg);
    auto create_schedule(const mara::config_t& cfg);
    auto create_state(const mara::config_t& run_config);
    auto create_next_function(const solver_data_t& solver_data);


    //=========================================================================
    bool simulation_should_continue(const state_t& state);
    auto diagnostic_fields(const solution_t& solution, const mara::config_t& run_config);
    void prepare_filesystem(const mara::config_t& cfg);
}




//=============================================================================
namespace mara
{
    void write(h5::Group& group, std::string name, const binary_fmr::solution_t& solution);
    void write(h5::Group& group, std::string name, const binary_fmr::state_t& state);
    void write(h5::Group& group, std::string name, const binary_fmr::diagnostic_fields_t& diagnostics);
}




//=============================================================================
auto binary_fmr::config_template()
{
    return mara::make_config_template()
    .item("restart",             std::string())
    .item("outdir",              "data")        // directory where data products are written to
    .item("cpi",                 10.0)          // checkpoint interval (orbits; chkpt.????.h5 - snapshot of app_state)
    .item("dfi",                  1.0)          // diagnostic field interval (orbits; diagnostics.????.h5)
    .item("tsi",                  0.1)          // time series interval (orbits)
    .item("tfinal",               1.0)          // simulation stop time (orbits)
    .item("depth",                  4)
    .item("block_size",            64)
    .item("focus_factor",         1.0)
    .item("rk_order",               2)          // time-stepping Runge-Kutta order: 1 or 2
    .item("reconstruct_method", "plm")          // zone extrapolation method: pcm or plm
    .item("plm_theta",            1.8)          // plm theta parameter: [1.0, 2.0]
    .item("riemann",           "hllc")          // riemann solver to use: hlle or hllc
    .item("softening_radius",     0.1)          // gravitational softening radius
    .item("sink_radius",          0.1)          // radius of mass (and momentum) subtraction region
    .item("sink_rate",            1e2)          // sink rate at the point masses (orbital angular frequency)
    .item("domain_radius",        6.0)          // half-size of square domain
    .item("separation",           1.0)          // binary separation: 0.0 or 1.0 (zero emulates a single body)
    .item("mass_ratio",           1.0)          // binary mass ratio M2 / M1: (0.0, 1.0]
    .item("eccentricity",         0.0)          // orbital eccentricity: [0.0, 1.0)
    .item("buffer_damping_rate", 10.0)          // maximum rate of buffer zone, where solution is driven to initial state
    .item("counter_rotate",         0);         // retrograde disk option: 0 or 1
}




//=============================================================================
auto binary_fmr::initial_disk_profile(const mara::config_t& cfg)
{
    auto softening_radius  = cfg.get_double("softening_radius");
    auto counter_rotate    = cfg.get_int("counter_rotate");

    return [=] (location_2d_t point)
    {
        auto GM             = 1.0;
        auto x              = point[0].value;
        auto y              = point[1].value;
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




//=========================================================================
void mara::write(h5::Group& group, std::string name, const binary_fmr::solution_t& solution)
{
    auto location = group.require_group(name);
    mara::write(location, "time",       solution.time);
    mara::write(location, "iteration",  solution.iteration);
    mara::write(location, "conserved",  solution.conserved);
}

void mara::write(h5::Group& group, std::string name, const binary_fmr::state_t& state)
{
    auto location = group.require_group(name);
    mara::write(location, "solution",   state.solution);
    mara::write(location, "schedule",   state.schedule);
    mara::write(location, "run_config", state.run_config);
}

void mara::write(h5::Group& group, std::string name, const binary_fmr::diagnostic_fields_t& diagnostics)
{
    auto location = group.require_group(name);
    mara::write(location, "run_config",        diagnostics.run_config);
    mara::write(location, "time",              diagnostics.time);
    mara::write(location, "vertices",          diagnostics.vertices);
    mara::write(location, "conserved",         diagnostics.conserved);
    mara::write(location, "position_of_mass1", diagnostics.position_of_mass1);
    mara::write(location, "position_of_mass2", diagnostics.position_of_mass2);
}




//=============================================================================
auto binary_fmr::create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);
    return args.count("restart")
    ? config_template()
            .create()
            .update(mara::read_config(h5::File(args.at("restart"), "r").open_group("run_config")))
            .update(args)
    : config_template().create().update(args);
}

auto binary_fmr::create_solver_data(const mara::config_t& cfg)
{
    return solver_data_t{};
}

auto binary_fmr::create_vertices(const mara::config_t& run_config)
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

auto binary_fmr::create_solution(const mara::config_t& run_config)
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

auto binary_fmr::create_schedule(const mara::config_t& cfg)
{
    return mara::schedule_t{};
}

auto binary_fmr::create_state(const mara::config_t& run_config)
{
    return state_t{
        create_solution(run_config),
        create_schedule(run_config),
        run_config,
    };
}

auto binary_fmr::create_next_function(const solver_data_t& solver_data)
{
    return [] (const state_t& state) { return state; };
}




//=============================================================================
bool binary_fmr::simulation_should_continue(const state_t& state)
{
    return state.solution.time / (2 * M_PI) < state.run_config.get<double>("tfinal");
}

auto binary_fmr::diagnostic_fields(const solution_t& solution, const mara::config_t& run_config)
{
    return diagnostic_fields_t{
        run_config,
        solution.time,
        create_vertices(run_config),
        solution.conserved,
        location_2d_t{0, 0}, // TODO
        location_2d_t{0, 0}, // TODO
    };
}

void binary_fmr::prepare_filesystem(const mara::config_t& cfg)
{
    auto outdir = cfg.get_string("outdir");
    mara::filesystem::require_dir(outdir);
}




//=============================================================================
class subprog_binary_fmr : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto run_config  = binary_fmr::create_run_config(argc, argv);
        auto solver_data = binary_fmr::create_solver_data(run_config);
        auto state       = binary_fmr::create_state(run_config);
        // auto next        = binary_fmr::create_next_function(solver_data);
        auto perf        = mara::perf_diagnostics_t();

        binary_fmr::prepare_filesystem(run_config);
        mara::pretty_print(std::cout, "config", run_config);
        // state = run_tasks(state);

        // while (binary_fmr::simulation_should_continue(state))
        {
            // std::tie(state, perf) = mara::time_execution(mara::compose(run_tasks, next), state);
            // print_run_loop_message(state, perf);
        }

        auto file = h5::File("diagnostics.0000.h5", "w");
        auto group = file.require_group("/");
        mara::write(group, "/", diagnostic_fields(state.solution, run_config));

        // run_tasks(next(state));
        return 0;
    }

    std::string name() const override
    {
        return "binary_fmr";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_binary_fmr()
{
    return std::make_unique<subprog_binary_fmr>();
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY_FMR
