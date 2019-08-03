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




#include <iostream>
#include "subprog_binary.hpp"
#include "mesh_tree_operators.hpp"
#include "core_ndarray_ops.hpp"
#include "app_serialize.hpp"
#include "app_serialize_tree.hpp"
#include "app_subprogram.hpp"
#include "app_filesystem.hpp"




//=============================================================================
namespace binary
{
    auto next_solution(const solution_t& solution, const solver_data_t& solver_data);
    auto next_schedule(const state_t& state);
    auto next_state(const state_t& state, const solver_data_t& solver_data);
    auto run_tasks(const state_t& state, const solver_data_t& solver_data);
    auto simulation_should_continue(const state_t& state);
}




//=============================================================================
mara::config_template_t binary::create_config_template()
{
    return mara::make_config_template()
    .item("restart",             std::string())
    .item("outdir",              "data")        // directory where data products are written to
    .item("cpi",                 10.0)          // checkpoint interval (orbits; chkpt.????.h5 - snapshot of app_state)
    .item("dfi",                  1.0)          // diagnostic field interval (orbits; diagnostics.????.h5)
    .item("tsi",                 2e-3)          // time series interval (orbits)
    .item("tfinal",               1.0)          // simulation stop time (orbits)
    .item("cfl_number",           0.4)          // the Courant number to use
    .item("depth",                  4)
    .item("block_size",            24)
    .item("focus_factor",        2.00)
    .item("focus_index",         2.00)
    .item("threaded",               1)          // set to 0 to disable multi-threaded tree updates
    .item("rk_order",               2)          // time-stepping Runge-Kutta order: 1 or 2
    .item("reconstruct_method", "plm")          // zone extrapolation method: pcm or plm
    .item("plm_theta",            1.8)          // plm theta parameter: [1.0, 2.0]
    .item("riemann",           "hlle")          // riemann solver to use: hlle only (hllc disabled until further testing)
    .item("softening_radius",    0.05)          // gravitational softening radius
    .item("source_term_softening", 1.)          // number of cells within which the Sr source term is suppressed
    .item("sink_radius",         0.05)          // radius of mass (and momentum) subtraction region
    .item("sink_rate",           50.0)          // sink rate at the point masses (orbital angular frequency)
    .item("buffer_damping_rate", 10.0)          // maximum rate of buffer zone, where solution is driven to initial state
    .item("domain_radius",       24.0)          // half-size of square domain
    .item("disk_radius",          2.0)          // characteristic disk radius (in units of binary separation)
    .item("disk_mass",           1e-3)          // total disk mass (in units of the binary mass)
    .item("ambient_density",     1e-4)          // surface density beyond torus
    .item("separation",           1.0)          // binary separation: 0.0 or 1.0 (zero emulates a single body)
    .item("mass_ratio",           1.0)          // binary mass ratio M2 / M1: (0.0, 1.0]
    .item("eccentricity",         0.0)          // orbital eccentricity: [0.0, 1.0)
    .item("counter_rotate",         0)          // retrograde disk option: 0 or 1
    .item("mach_number",         40.0)          // disk mach number; for locally isothermal EOS
    .item("alpha",                0.0);         // viscous alpha coefficient
}




//=============================================================================
binary::primitive_field_t binary::create_disk_profile(const mara::config_t& run_config)
{
    auto softening_radius  = run_config.get_double("softening_radius");
    auto disk_radius       = run_config.get_double("disk_radius");
    auto mach_number       = run_config.get_double("mach_number");
    auto disk_mass         = run_config.get_double("disk_mass");
    auto ambient_density   = run_config.get_double("ambient_density");
    auto counter_rotate    = run_config.get_int("counter_rotate");
    auto rc = disk_radius;
    auto s1 = ambient_density;

    auto sigma = [=] (double r)
    {
        auto sigma0 = disk_mass / (17.0618 * rc * rc); // see mathematica notebook
        auto x = r / rc;
        return sigma0 * (std::exp(-0.5 * (x - 1) * (x - 1)) + s1);
    };

    auto dp_dr = [=] (double r)
    {
        auto GM = 1.0;
        auto Ma = mach_number;
        auto rs = softening_radius;
        auto x = r / rc;
        return (GM / Ma / Ma / (r + rs)) * (x * (1 - x) * (1 - s1 / sigma(r)) - 1.0);
    };

    return [=] (location_2d_t point)
    {
        auto x              = point[0].value;
        auto y              = point[1].value;
        auto r2             = x * x + y * y;
        auto r              = std::sqrt(r2);
        auto rs             = softening_radius;
        auto GM             = 1.0;
        auto vp             = std::sqrt(GM / (r + rs) + dp_dr(r)) * (counter_rotate ? -1 : 1);
        auto vx             = vp * (-y / r);
        auto vy             = vp * ( x / r);

        return mara::iso2d::primitive_t()
            .with_sigma(sigma(r))
            .with_velocity_x(vx)
            .with_velocity_y(vy);
    };
}

mara::config_t binary::create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);
    return args.count("restart")
    ? create_config_template()
            .create()
            .update(mara::read_config(h5::File(args.at("restart"), "r").open_group("run_config")))
            .update(args)
    : create_config_template().create().update(args);
}

binary::quad_tree_t<binary::location_2d_t> binary::create_vertices(const mara::config_t& run_config)
{
    auto domain_radius = run_config.get_double("domain_radius");
    auto focus_factor  = run_config.get_double("focus_factor");
    auto focus_index   = run_config.get_double("focus_index");
    auto block_size    = run_config.get_int("block_size");
    auto depth         = run_config.get_int("depth");

    auto refinement_radius = [focus_factor, focus_index] (std::size_t level, double centroid_radius)
    {
        return centroid_radius < focus_factor / std::pow(level, focus_index);
    };

    return mara::create_vertex_quadtree(refinement_radius, block_size, depth)
    .map([domain_radius] (auto block)
    {
        return (block * domain_radius).shared();
    });
}

mara::orbital_elements_t binary::create_binary_params(const mara::config_t& run_config)
{
    auto binary = mara::orbital_elements_t();
    binary.total_mass   = 1.0;
    binary.separation   = run_config.get_double("separation");
    binary.mass_ratio   = run_config.get_double("mass_ratio");
    binary.eccentricity = run_config.get_double("eccentricity");
    return binary;
}

binary::solution_t binary::create_solution(const mara::config_t& run_config)
{
    auto conserved = create_vertices(run_config).map([&run_config] (auto block)
    {
        auto cell_centers = block | nd::midpoint_on_axis(0) | nd::midpoint_on_axis(1);
        auto primitive = cell_centers | nd::map(create_disk_profile(run_config));
        return nd::zip(cell_centers, primitive)
        | nd::apply([] (auto x, auto p) { return p.to_conserved_angmom_per_area(x); })
        | nd::to_shared();
    });

    return solution_t{
        0, 0.0, conserved, {}, {}, {}, {}, {}, {},
        mara::make_full_orbital_elements_with_zeros(),
        mara::make_full_orbital_elements_with_zeros(),
    };
}

mara::schedule_t binary::create_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_checkpoint");
    schedule.create_and_mark_as_due("write_diagnostics");
    schedule.create_and_mark_as_due("record_time_series");
    return schedule;
}

binary::state_t binary::create_state(const mara::config_t& run_config)
{
    auto restart = run_config.get<std::string>("restart");

    if (restart.empty())
    {
        return {
            create_solution(run_config),
            create_schedule(run_config),
            {},
            run_config,
        };
    }
    return mara::read<state_t>(h5::File(restart, "r").open_group("/"), "/").with(run_config);
}




//=============================================================================
auto binary::next_solution(const solution_t& solution, const solver_data_t& solver_data)
{
    auto can_fail = [] (const solution_t& solution, const solver_data_t& solver_data, auto dt, bool safe_mode)
    {
        auto s0 = solution;

        switch (solver_data.rk_order)
        {
            case 1:
            {
                return advance(s0, solver_data, dt, safe_mode);
            }
            case 2:
            {
                auto b0 = mara::make_rational(1, 2);
                auto s1 = advance(s0, solver_data, dt, safe_mode);
                auto s2 = advance(s1, solver_data, dt, safe_mode);
                return s0 * b0 + s2 * (1 - b0);
            }
        }
        throw std::invalid_argument("binary::next_solution");
    };

    try {
        return can_fail(solution, solver_data, solver_data.recommended_time_step, false);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return can_fail(solution, solver_data, solver_data.recommended_time_step * 0.5, true);
    }
}

auto binary::next_schedule(const state_t& state)
{
    return mara::mark_tasks_in(state, state.solution.time.value,
        {{"write_checkpoint",   state.run_config.get_double("cpi") * 2 * M_PI},
         {"write_diagnostics",  state.run_config.get_double("dfi") * 2 * M_PI},
         {"record_time_series", state.run_config.get_double("tsi") * 2 * M_PI}});
}

auto binary::next_state(const state_t& state, const solver_data_t& solver_data)
{
    return state_t{
        next_solution(state.solution, solver_data),
        next_schedule(state),
        state.time_series,
        state.run_config,
    };
}




//=============================================================================
auto binary::simulation_should_continue(const state_t& state)
{
    return state.solution.time / (2 * M_PI) < state.run_config.get_double("tfinal");
}




//=============================================================================
auto binary::run_tasks(const state_t& state, const solver_data_t& solver_data)
{


    //=========================================================================
    auto write_checkpoint  = [] (const state_t& state)
    {
        auto outdir = state.run_config.get_string("outdir");
        auto count  = state.schedule.num_times_performed("write_checkpoint");
        auto fname  = mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", count, "h5"));
        auto group  = h5::File(fname, "w").open_group("/");
        auto next_state = mara::complete_task_in(state, "write_checkpoint");
        mara::write(group, "/", next_state);
        std::printf("write checkpoint: %s\n", fname.data());
        return next_state;
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
    auto record_time_series = [&solver_data] (state_t state)
    {
        auto sample = time_series_sample_t();
        sample.time                         = state.solution.time;
        sample.mass_accreted_on             = state.solution.mass_accreted_on;
        sample.angular_momentum_accreted_on = state.solution.angular_momentum_accreted_on;
        sample.integrated_torque_on         = state.solution.integrated_torque_on;
        sample.work_done_on                 = state.solution.work_done_on;
        sample.mass_ejected                 = state.solution.mass_ejected;
        sample.angular_momentum_ejected     = state.solution.angular_momentum_ejected;
        sample.disk_mass                    = disk_mass            (state.solution, solver_data);
        sample.disk_angular_momentum        = disk_angular_momentum(state.solution, solver_data);
        sample.orbital_elements_acc         = state.solution.orbital_elements_acc;
        sample.orbital_elements_grav        = state.solution.orbital_elements_grav;

        state.time_series = state.time_series.prepend(sample);
        return mara::complete_task_in(state, "record_time_series");
    };


    return mara::run_scheduled_tasks(state, {
        {"write_diagnostics", write_diagnostics},
        {"record_time_series", record_time_series},
        {"write_checkpoint",  write_checkpoint}});
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
        auto tasks       = std::bind(binary::run_tasks, std::placeholders::_1, solver_data);
        auto perf        = mara::perf_diagnostics_t();

        binary::prepare_filesystem(run_config);
        binary::set_scheme_globals(run_config);
        mara::pretty_print(std::cout, "config", run_config);
        state = tasks(state);

        while (binary::simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(mara::compose(tasks, next), state);
            binary::print_run_loop_message(state, perf);
        }

        tasks(next(state));
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
