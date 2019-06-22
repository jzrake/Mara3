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
#if MARA_COMPILE_SUBPROGRAM_AMRSAND




#include <iostream>
#include "app_subprogram.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"
#include "app_performance.hpp"
#include "app_serialize.hpp"
#include "app_serialize_tree.hpp"
#include "core_dimensional.hpp"
#include "core_rational.hpp"
#include "core_ndarray_ops.hpp"
#include "core_tree.hpp"
#include "mesh_prolong_restrict.hpp"
#include "mesh_tree_operators.hpp"




//=============================================================================
struct AmrSandbox
{
    using location_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, 0, double>, 2>;
    using conserved_t   = mara::dimensional_value_t<-2, 1, 0, double>;
    template<typename T> using quadtree_t = mara::arithmetic_binary_tree_t<T, 2>;




    //=========================================================================
    struct solution_state_t
    {
        mara::rational_number_t iteration;
        mara::unit_time<double> time;
        quadtree_t<nd::shared_array<location_2d_t, 2>> vertices;
        quadtree_t<nd::shared_array<conserved_t, 2>> conserved;
    };




    //=========================================================================
    struct app_state_t
    {
        solution_state_t solution;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };




    //=========================================================================
    struct diagnostics_t
    {
        mara::unit_time<double> time;
        quadtree_t<nd::shared_array<location_2d_t, 2>> vertices;
        quadtree_t<nd::shared_array<conserved_t, 2>> conserved;
        void write(h5::Group&& group) const;
    };




    //=========================================================================
    static auto config_template();
    static auto create_solution_state(const mara::config_t& run_config);
    static auto create_schedule(const mara::config_t& run_config);
    static auto create_app_state(const mara::config_t& run_config);
    static auto create_diagnostics(const app_state_t& state);
    static auto next_solution(const solution_state_t& solution);
    static auto next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time);
    static auto next(const app_state_t& state);
    static auto run_tasks(const app_state_t& state);




    //=========================================================================
    static void print_run_loop_message(const app_state_t& state, mara::perf_diagnostics_t perf);
    static void prepare_filesystem(const mara::config_t& cfg);
    static void write_diagnostics(const app_state_t& state);
    static bool simulation_should_continue(const app_state_t& state);
};




//=============================================================================
void AmrSandbox::diagnostics_t::write(h5::Group&& group) const
{
    group.write("time", time);
    mara::write_tree(group.require_group("vertices"), vertices);
    mara::write_tree(group.require_group("conserved"), conserved);
}




//=============================================================================
auto AmrSandbox::config_template()
{
    return mara::make_config_template()
    .item("restart",             std::string())
    .item("outdir",                     "data")          // directory where data products are written to
    .item("dfi",                           1.0)          // diagnostic field interval (orbits; diagnostics.????.h5 - for plotting 2d solution data)
    .item("tfinal",                        1.0)          // simulation stop time (orbits)
    .item("block_size",                     16)
    .item("depth",                           4);
}    




//=============================================================================
auto AmrSandbox::create_solution_state(const mara::config_t& run_config)
{
    auto vertices = mara::create_vertex_quadtree(
        [] (std::size_t level, double centroid_radius) { return centroid_radius < 1.0 / level; },
        run_config.get_int("block_size"),
        run_config.get_int("depth"));

    return solution_state_t
    {
        0,
        0.0,
        vertices,
        vertices.map([] (auto block)
        { return block
            | nd::midpoint_on_axis(0)
            | nd::midpoint_on_axis(1)
            | nd::map([] (location_2d_t p) { return std::exp(-(p[0] * p[0] + p[1] * p[1]).value / 0.025); })
            | nd::map([] (auto u) { return conserved_t{u}; })
            | nd::to_shared(); }),
    };
}

auto AmrSandbox::create_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_diagnostics");
    return schedule;
}

auto AmrSandbox::create_app_state(const mara::config_t& run_config)
{
    return app_state_t
    {
        create_solution_state(run_config),
        create_schedule(run_config),
        run_config,
    };
}

auto AmrSandbox::create_diagnostics(const app_state_t& state)
{
    return diagnostics_t
    {
        state.solution.time,
        state.solution.vertices,
        state.solution.conserved,
    };
}




//=============================================================================
void AmrSandbox::print_run_loop_message(const app_state_t& state, mara::perf_diagnostics_t perf)
{
    auto num_zones = state.solution.vertices
    .map([] (auto&& block) { return block.size(); })
    .sum();

    auto kzps = num_zones / perf.execution_time_ms;

    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n",
        state.solution.iteration.as_integral(),
        state.solution.time.value, kzps);
}

void AmrSandbox::prepare_filesystem(const mara::config_t& cfg)
{
    mara::filesystem::require_dir(cfg.get_string("outdir"));
}

void AmrSandbox::write_diagnostics(const app_state_t& state)
{
    auto outdir = state.run_config.get_string("outdir");
    auto count = state.schedule.num_times_performed("write_diagnostics");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5")), "w");
    create_diagnostics(state).write(file.open_group("."));
    std::printf("write diagnostics: %s\n", file.filename().data());
}

bool AmrSandbox::simulation_should_continue(const app_state_t& state)
{
    return state.solution.time.value < state.run_config.get_double("tfinal");
}




//=============================================================================
auto AmrSandbox::next_schedule(const mara::schedule_t& schedule, const mara::config_t& run_config, double time)
{
    auto next_schedule = schedule;
    auto dfi = run_config.get_double("dfi");

    if (time - schedule.last_performed("write_diagnostics") >= dfi) next_schedule.mark_as_due("write_diagnostics", dfi);

    return next_schedule;
}

auto AmrSandbox::next_solution(const solution_state_t& solution)
{
    auto n = std::max(solution.vertices.front().shape(0), solution.vertices.front().shape(1));
    auto dt = mara::make_time(2.0 / n / (1 << solution.vertices.depth()));

    auto take = [] (std::size_t component)
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
    auto area_from_vertices = [take] (auto vertices)
    {
        auto dx = vertices | take(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | take(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };
    auto spacing_on_axis = [take] (std::size_t axis)
    {
        return [take, axis] (auto vertices)
        {
            return vertices | take(axis) | nd::difference_on_axis(axis);
        };
    };
    auto flux_from_conserved_density = [] (std::size_t axis)
    {
        return [axis] (auto u)
        {
            return u * mara::make_velocity(0.5) | nd::select_axis(axis).from(0).to(1).from_the_end();
        };
    };

    auto v0 = solution.vertices;
    auto dA = solution.vertices.map(area_from_vertices).map(nd::to_shared());
    auto dx = solution.vertices.map(spacing_on_axis(0)).map(nd::to_shared());
    auto dy = solution.vertices.map(spacing_on_axis(1)).map(nd::to_shared());
    auto u0 = solution.conserved;
    auto fx = extend(u0, 0).map(flux_from_conserved_density(0)) * dy;
    auto fy = extend(u0, 1).map(flux_from_conserved_density(1)) * dx;
    auto lx = -fx.map(nd::difference_on_axis(0));
    auto ly = -fy.map(nd::difference_on_axis(1));
    auto u1 = u0 + (lx + ly) * dt / dA;

    return solution_state_t{
        solution.iteration + 1,
        solution.time + dt,
        solution.vertices,
        u1.map(nd::to_shared()),
    };
}

auto AmrSandbox::next(const app_state_t& state)
{
    return app_state_t{
        next_solution(state.solution),
        next_schedule(state.schedule, state.run_config, state.solution.time.value),
        state.run_config,
    };
}

auto AmrSandbox::run_tasks(const app_state_t& state)
{
    auto next_state = state;

    if (state.schedule.is_due("write_diagnostics"))
    {
        write_diagnostics(state);
        next_state.schedule.mark_as_completed("write_diagnostics");
    }
    return next_state;
}




//=============================================================================
class subprog_amrsand : public mara::sub_program_t
{
public:

    int run_main(const mara::config_t& cfg)
    {
        using prob             = AmrSandbox;
        auto run_tasks_on_next = mara::compose(prob::run_tasks, prob::next);
        auto perf              = mara::perf_diagnostics_t();
        auto state             = prob::create_app_state(cfg);

        mara::pretty_print(std::cout, "config", cfg);
        prob::prepare_filesystem(cfg);

        state = prob::run_tasks(state);

        while (prob::simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(run_tasks_on_next, state);
            prob::print_run_loop_message(state, perf);
        }

        run_tasks_on_next(state);
        return 0;
    }

    int main(int argc, const char* argv[]) override
    {
        return run_main(AmrSandbox::config_template().create().update(mara::argv_to_string_map(argc, argv)));
    }

    std::string name() const override
    {
        return "amrsand";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_amrsand()
{
    return std::make_unique<subprog_amrsand>();
}

#endif // MARA_COMPILE_SUBPROGRAM_AMRSAND
