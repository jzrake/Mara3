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




#include "subprog_binary.hpp"
#include "app_serialize.hpp"
#include "app_serialize_tree.hpp"
#include "app_subprogram.hpp"
#include "app_filesystem.hpp"
#define cfl_number 0.4




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
    .item("focus_index",         1.00)
    .item("rk_order",               2)          // time-stepping Runge-Kutta order: 1 or 2
    .item("reconstruct_method", "plm")          // zone extrapolation method: pcm or plm
    .item("plm_theta",            1.8)          // plm theta parameter: [1.0, 2.0]
    .item("riemann",           "hlle")          // riemann solver to use: hlle only (hllc disabled until further testing)
    .item("softening_radius",     0.1)          // gravitational softening radius
    .item("sink_radius",          0.1)          // radius of mass (and momentum) subtraction region
    .item("sink_rate",            1e2)          // sink rate at the point masses (orbital angular frequency)
    .item("domain_radius",       48.0)          // half-size of square domain
    .item("separation",           1.0)          // binary separation: 0.0 or 1.0 (zero emulates a single body)
    .item("mass_ratio",           1.0)          // binary mass ratio M2 / M1: (0.0, 1.0]
    .item("eccentricity",         0.0)          // orbital eccentricity: [0.0, 1.0)
    .item("buffer_damping_rate", 10.0)          // maximum rate of buffer zone, where solution is driven to initial state
    .item("counter_rotate",         0)          // retrograde disk option: 0 or 1
    .item("mach_number",         10.0);
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
    auto focus_index   = run_config.get_double("focus_index");
    auto block_size    = run_config.get_int("block_size");
    auto depth         = run_config.get_int("depth");

    auto refinement_radius = [focus_factor, focus_index] (std::size_t level, double centroid_radius)
    {
        return centroid_radius < focus_factor / std::pow(level, focus_index);
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

binary::solver_data_t binary::create_solver_data(const mara::config_t& run_config)
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

    auto buffer_rate_field = vertices.map([&run_config] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map([&run_config] (location_2d_t p)
        {
            auto tightness   = mara::dimensional_value_t<-1, 0, 0, double>(3.0);
            auto buffer_rate = mara::dimensional_value_t< 0, 0,-1, double>(run_config.get_double("buffer_damping_rate"));
            auto r1 = mara::make_length(run_config.get_double("domain_radius"));
            auto rc = (p[0] * p[0] + p[1] * p[1]).pow<1, 2>();
            auto y = (tightness * (rc - r1)).scalar();
            return buffer_rate * (1.0 + std::tanh(y));
        })
        | nd::to_shared();
    });


    //=========================================================================
    auto result = solver_data_t();
    result.mach_number           = run_config.get_double("mach_number");
    result.sink_rate             = run_config.get_double("sink_rate");
    result.sink_radius           = run_config.get_double("sink_radius");
    result.softening_radius      = run_config.get_double("softening_radius");
    result.plm_theta             = run_config.get_double("plm_theta");
    result.rk_order              = run_config.get_int("rk_order");
    result.recommended_time_step = std::min(min_dx, min_dy) / max_velocity * cfl_number;
    result.binary_params         = create_binary_params(run_config);
    result.buffer_rate_field     = buffer_rate_field;
    result.vertices              = vertices;
    result.initial_conserved     = primitive
    .map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area)))
    .map(nd::to_shared());

    if      (run_config.get_string("riemann") == "hlle") result.riemann_solver = riemann_solver_t::hlle;
    // else if (run_config.get_string("riemann") == "hllc") result.riemann_solver = riemann_solver_t::hllc;
    else throw std::invalid_argument("invalid riemann solver '" + run_config.get_string("riemann") + "', must be hlle");

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
auto binary::next_solution(const solution_t& state, const solver_data_t& solver_data)
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
