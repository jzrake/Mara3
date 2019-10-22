/**
 ==============================================================================
 Copyright 2019, Christopher Tiede

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




#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <cmath>
#include <vector>
#include "app_config.hpp"
#include "app_parallel.hpp"
#include "app_serialize.hpp"
#include "app_serialize_tree.hpp"
#include "app_loads_dumps.hpp"
#include "app_performance.hpp"
#include "core_mpi.hpp"
#include "core_hdf5.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "core_linked_list.hpp"
#include "mesh_tree_operators.hpp"
#include "physics_iso2d.hpp"
#define sound_speed_squared    0.1
#define guard_zone_count       1




//=============================================================================
template<>
struct h5::hdf5_type_info<mara::iso2d::conserved_per_area_t>
{
    using native_type = mara::iso2d::conserved_per_area_t;
    static auto make_datatype_for(const native_type& value) { return h5::Datatype::native_double().as_array(3); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto convert_to_writable(const native_type& value) { return value; }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(const native_type& value) { return &value; }
    static auto get_address(native_type& value) { return &value; }
};




namespace iso_mpi
{


    template<typename ArrayValueType>
    using quad_tree_t       = mara::arithmetic_binary_tree_t<nd::shared_array<ArrayValueType, 2>, 2>;
    using index_tree_t      = mara::arithmetic_binary_tree_t<mara::tree_index_t<2>, 2>;
    using neighbor_tree_t   = mara::arithmetic_binary_tree_t<mara::linked_list_t<std::size_t>, 2>;
    using location_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using primitive_field_t = std::function<mara::iso2d::primitive_t(location_2d_t)>;




    // ========================================================================
    struct solution_t
    {
        mara::unit_time<double>                         time = 0.0;
        mara::rational_number_t                         iteration = 0;
        quad_tree_t<location_2d_t>                      vertices;
        quad_tree_t<mara::iso2d::conserved_per_area_t>  conserved;


        // Overload operators to manipulate solution_t types
        //=====================================================================
        solution_t operator+(const solution_t& other) const
        {
            return {
                time       + other.time,
                iteration  + other.iteration,
                vertices,
                (conserved + other.conserved).map(nd::to_shared())
            };
        }
        solution_t operator*(mara::rational_number_t scale) const
        {
            return {
                time       * scale.as_double(),
                iteration  * scale,
                vertices,
                (conserved * scale.as_double()).map(nd::to_shared())
            };
        }
    };




    //=========================================================================
    struct mpi_setup_t
    {
        mpi::Communicator                               comm;
        mara::arithmetic_binary_tree_t<std::size_t, 2>  decomposition;
    };




    //=========================================================================
    struct state_t
    {
        solution_t          solution;
        mara::config_t      run_config;
    };




    //=========================================================================
    mara::config_template_t             create_config_template();
    mara::config_t                      create_run_config     (int argc, const char* argv[]);
    mpi_setup_t                         create_mpi_setup      (const mara::config_t& run_config);
    index_tree_t                        create_domain_topology(const mara::config_t& run_config);
    state_t                             create_state   (const mara::config_t& run_config, const mpi_setup_t& mpi_setup);
    solution_t                          create_solution(const mara::config_t& run_config, const mpi_setup_t& mpi_setup);
    quad_tree_t<location_2d_t>          create_vertex_blocks(const mara::config_t& run_config, const mpi_setup_t& mpi_setup);

    solution_t  advance      (const solution_t& solution, const mpi_setup_t& mpi_setup, mara::unit_time<double> dt);
    solution_t  next_solution(const state_t& state, const mpi_setup_t& mpi_setup);
    state_t     next_state   (const state_t& state, const mpi_setup_t& mpi_setup);

    template<typename ValueType>
    auto mpi_fill_tree(quad_tree_t<ValueType> block_tree, const mpi_setup_t& mpi_setup);
    auto simulation_should_continue(const state_t& state);
}




//=============================================================================
auto component(std::size_t cmpnt)
{
    return nd::map([cmpnt] (auto p) { return p[cmpnt]; });
};

template<std::size_t I>
static auto component()
{
    return nd::map([] (auto p) { return mara::get<I>(p); });
};

auto recover_primitive(const mara::iso2d::conserved_per_area_t& conserved)
{
    return mara::iso2d::recover_primitive(conserved);
}




//=============================================================================
mara::config_template_t iso_mpi::create_config_template()
{
    return mara::make_config_template()
     .item("outdir", "hydro_run")       // directory where data products are written
     .item("cpi",           10.0)       // checkpoint interval
     .item("rk_order",         1)		// timestepping order
     .item("tfinal",         1.0)
     .item("cfl",            0.4)       // courant number
     .item("domain_radius",  1.0)       // half-size of square domain
     .item("block_size",      24);      // number of cells in each direction
}

mara::config_t iso_mpi::create_run_config( int argc, const char* argv[] )
{
    auto args = mara::argv_to_string_map( argc, argv );
    return create_config_template().create().update(args);
}




//=============================================================================
iso_mpi::index_tree_t iso_mpi::create_domain_topology(const mara::config_t& run_config)
{
    // auto centroid_radius = [] (mara::tree_index_t<2> index)
    // {
    //     double x = 2.0 * ((index.coordinates[0] + 0.5) / (1 << index.level) - 0.5);
    //     double y = 2.0 * ((index.coordinates[1] + 0.5) / (1 << index.level) - 0.5);
    //     return std::sqrt(x * x + y * y);
    // };

    // return mara::tree_with_topology<2>([centroid_radius] (auto index)
    // {
    //     return centroid_radius(index) < 0.25 ? (index.level < 4) : (index.level < 3);
    // });
    
    return mara::tree_with_topology<2>([] (auto index)
    {
        return index.level < 3;
    });
}




/**
 * @brief             Create 2D array of location_2d_t points representing the vertices
 *
 * @param  run_config Config object
 *
 * @return            2D array of vertices
 */
iso_mpi::quad_tree_t<iso_mpi::location_2d_t> iso_mpi::create_vertex_blocks(
    const mara::config_t& run_config,
    const iso_mpi::mpi_setup_t& mpi_setup)
{
    auto my_rank       = mpi_setup.comm.rank();
    auto rank_tree     = mpi_setup.decomposition;
    auto block_size    = run_config.get_int("block_size");
    auto domain_radius = run_config.get_double("domain_radius");

    auto build_my_vertex_blocks = [my_rank, domain_radius, block_size] (auto iv)
    {
        if (iv.second == my_rank)
        {
            auto index = iv.first;
            auto block_length = domain_radius / (1 << (index.level - 1));

            auto x0 = index.coordinates[0] * block_length;
            auto y0 = index.coordinates[1] * block_length;
            auto x_points = nd::linspace(0, 1, block_size + 1) * block_length + x0;
            auto y_points = nd::linspace(0, 1, block_size + 1) * block_length + y0;

            return nd::cartesian_product( x_points, y_points )
                | nd::apply([] (double x, double y) { return iso_mpi::location_2d_t{x, y}; })
                | nd::to_shared();
        }
        return nd::shared_array<iso_mpi::location_2d_t, 2>{};
    };
    return rank_tree.pair_indexes().map(build_my_vertex_blocks);
}




/**
 * @brief     Apply initial condition to a tuple of position coordinates
 *
 */
auto initial_condition_shocktube(iso_mpi::location_2d_t position)
{
    auto density = position[0] > 1.0 ? 0.1 : 1.0;
    auto vx      = 0.0;
    auto vy      = 0.0;

    return mara::iso2d::primitive_t()
     .with_sigma(density)
     .with_velocity_x(vx)
     .with_velocity_y(vy);
}

auto initial_condition_cylinder(iso_mpi::location_2d_t position)
{
	auto x = position[0];
	auto y = position[1];

	auto r  = (x - 1) * (x - 1) + (y - 1) * (y - 1);
	auto vx = 0.0;
	auto vy = 0.0;

	// auto density = r < 0.2 ? 1.0 : 0.1;
    auto density = std::exp(-r.value * r.value / 0.01) + 0.1;

	return mara::iso2d::primitive_t()
	 .with_sigma(density)
	 .with_velocity_x(vx)
	 .with_velocity_y(vy);
}




/**
 * @brief     Create an initial solution object according to initial_condition()
 *
 */
iso_mpi::solution_t iso_mpi::create_solution(const mara::config_t& run_config, const mpi_setup_t& mpi_setup)
{
    auto vertices = create_vertex_blocks(run_config, mpi_setup);
    auto cell_centers = vertices.map([] (auto block)
    {
        if (block.size() == 0)
            return block | nd::to_shared();

        return block | nd::midpoint_on_axis(0) | nd::midpoint_on_axis(1) | nd::to_shared();
    });

    auto conserved = cell_centers.map([] (auto block)
    {
        return block
        | nd::map(initial_condition_cylinder)
        | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area))
        | nd::to_shared();
    });
    return solution_t{0.0, 0, vertices, conserved};
}




/**
 * @brief     Setup the mpi environment, create tree of ranks for the
 *            domain decomposition, and create tree of neighbor maps
 */
iso_mpi::mpi_setup_t iso_mpi::create_mpi_setup(const mara::config_t& run_config)
{
    // For now a uniformly refined tree of depth 3
    auto topology = create_domain_topology(run_config);
    auto comm     = mpi::comm_world();

    auto decomposition = mara::build_rank_tree<2>(topology, comm.size());
    return {comm, decomposition};
}




/**
 * @brief               Creates state object
 *
 * @param   run_config  configuration object
 *
 * @return              a state object
 */
iso_mpi::state_t iso_mpi::create_state(const mara::config_t& run_config, const mpi_setup_t& mpi_setup)
{
    return state_t{
        create_solution(run_config, mpi_setup),
        run_config
    };
}




/**
 * @brief      Takes an iterable object and a predicate (object) -> bool.
 *             Returns a vector of values for which predicate was satisfied.
 */
template<typename Iterable, typename Predicate>
auto filter(const Iterable& container, Predicate predicate)
{
    using value_type = std::decay_t<decltype(*container.begin())>;
    auto result = std::vector<value_type>();

    for (auto i : container)
        if (predicate(i))
            result.push_back(i);

    return result;
}

template<typename Iterable, typename Predicate>
auto remove_if(const Iterable& container, Predicate predicate)
{
    return filter(container, [predicate] (auto v) { return ! predicate(v); });
}

template<typename Iterable>
auto linked_list_from(const Iterable& container)
{
    using value_type = std::decay_t<decltype(*container.begin())>;
    return mara::linked_list_t<value_type>(container.begin(), container.end());
}




/**
 * @return   Returns a boolean function (index) -> bool that gives
 *           true if my process owns the index
 */
auto block_is_owned_by(std::size_t rank, const iso_mpi::mpi_setup_t& mpi_setup)
{
    return [rank, rank_tree = mpi_setup.decomposition] (auto index)
    {
        return rank_tree.at(index) == rank;
    };
}




/**
 * @brief   Return a unique vector of all the indexes my process needs in order
 *          to update all of the blocks that I own
 */
auto indexes_of_nonlocal_blocks(const iso_mpi::mpi_setup_t& mpi_setup)
{
    /**
     * @brief   Gives the index at target, it's parent, or all 4 of its children
     */
    auto indexes_at = [] (auto tree, auto target)
    {
        if (tree.contains(target))
            return mara::linked_list_t<mara::tree_index_t<2>>().prepend(target);

        if (tree.contains(target.parent_index()))
            return mara::linked_list_t<mara::tree_index_t<2>>().prepend(target.parent_index());

        return linked_list_from(tree.indexes().node_at(target));
    };


    /**
     * @brief   Get all neighbor indexes to a given index
     */
    auto get_neighbors_at = [indexes_at] (auto tree, auto idx)
    {
        return  indexes_at(tree, idx.next_on(0))
        .concat(indexes_at(tree, idx.prev_on(0)))
        .concat(indexes_at(tree, idx.next_on(1)))
        .concat(indexes_at(tree, idx.prev_on(1)));
    };


    // can fix this to make it more logical/efficient but I think it works for now
    //=========================================================================
    auto rank_tree = mpi_setup.decomposition;
    auto my_rank   = mpi_setup.comm.rank();
    auto indexes   = mara::linked_list_t<mara::tree_index_t<2>>();

    for (auto ir : rank_tree.pair_indexes())
    {
        auto idx  = ir.first;
        auto rank = ir.second;

        if (rank != my_rank)
            continue;

        indexes = indexes.concat(get_neighbors_at(rank_tree, idx));
    }
    return remove_if(indexes.unique(), block_is_owned_by(my_rank, mpi_setup));
}




template<typename ValueType>
auto iso_mpi::mpi_fill_tree(quad_tree_t<ValueType> block_tree, const mpi_setup_t& mpi_setup)
{
    using message_type_t = std::pair<mara::tree_index_t<2>, nd::shared_array<mara::iso2d::primitive_t, 2>>;

    auto comm             = mpi_setup.comm;
    auto rank_tree        = mpi_setup.decomposition;
    auto indexes_to_recv  = indexes_of_nonlocal_blocks(mpi_setup);   
    auto indexes_to_send  = comm.all_gather(indexes_to_recv);
    auto requests         = std::vector<mpi::Request>();

    for (auto rank : filter(nd::arange(comm.size()), [rank = comm.rank()] (auto r) { return r != rank; }))
        for (auto i : filter(indexes_to_send[rank], block_is_owned_by(comm.rank(), mpi_setup)))
            requests.push_back(comm.isend(mara::dumps(std::pair(i, block_tree.at(i))), rank, 0));

    for (auto index : indexes_to_recv)
    {
        auto [idx, block] = mara::loads<message_type_t>(comm.recv(rank_tree.at(index), 0));
        block_tree = block_tree.insert(idx, block);
    }

    comm.barrier(); // This barrier ensures all processes have completed their
                    // non-blocking sends before the requests go out of scope.

    for (const auto& request : requests)
        if (! request.is_ready())
            throw std::logic_error("A send request was not completed");

    return block_tree;
}



/**
 * @note   Tree should be the tree of prims after it has been filled with info
 *         from neighboring processes
 */
template<typename TreeType>
static auto extend(TreeType tree, std::size_t axis, std::size_t guard_count)
{
    return [tree, axis, guard_count] (auto ib)
    {
        auto [index, block] = ib;

        if (block.size() == 0)
            return block | nd::to_shared();

        auto C = tree.at(index);
        auto L = mara::get_cell_block(tree, index.prev_on(axis), mara::compose(nd::to_shared(), nd::select_final(guard_count, axis)));
        auto R = mara::get_cell_block(tree, index.next_on(axis), mara::compose(nd::to_shared(), nd::select_first(guard_count, axis)));
        return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis) | nd::to_shared();
    };
};




//=============================================================================
iso_mpi::solution_t iso_mpi::advance(const solution_t& solution, const mpi_setup_t& mpi_setup, mara::unit_time<double> dt)
{


    /*
     * @brief      Return an array of intercell fluxes by calling the specified
     *             riemann solver
     *
     * @param[in]  riemann_solver  The riemann solver to use
     * @param[in]  axis            The axis to get the fluxes on
     *
     * @return     An array operator that returns arrays of fluxes
     */
    auto intercell_flux = [] (std::size_t axis)
    {
        return [axis, riemann_solver=mara::iso2d::riemann_hllc] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(riemann_solver, _1, _2, sound_speed_squared, sound_speed_squared, nh);
            return left_and_right_states | nd::apply(riemann);
        };
    };


    /**
     * @brief      Return a function to give the fluxes for each block in the tree
     *
     * @param[in]  solution  The solution object
     * @param[in]  p0_ex     The primitives extended in x
     * @param[in]  p0_ey     The primitives extended in y
     */
    auto block_fluxes = [intercell_flux] (auto solution, auto p0_ex, auto p0_ey)
    {
        return [=] (auto tree_index)
        {
            auto xv = solution.vertices.at(tree_index);

            if (xv.size() == 0)
            {
                auto flux = nd::shared_array<mara::iso2d::flux_t, 2>() * mara::make_length(1.0) | nd::to_shared();
                return std::make_pair(flux, flux);
            }

            auto dx = xv | component<0>() | nd::difference_on_axis(0);
            auto dy = xv | component<1>() | nd::difference_on_axis(1);

            auto fx = p0_ex.at(tree_index) | nd::zip_adjacent2_on_axis(0) | intercell_flux(0);
            auto fy = p0_ey.at(tree_index) | nd::zip_adjacent2_on_axis(1) | intercell_flux(1);

            auto fhat_x = fx | nd::multiply(dy) | nd::to_shared();
            auto fhat_y = fy | nd::multiply(dx) | nd::to_shared();

            return std::make_pair(fhat_x, fhat_y);
        };
    };


    /**
     * @brief      Return a function to update the block at each tree index
     *
     * @param[in]  solution  The solution
     * @param[in]  fx        The fluxes in x
     * @param[in]  fy        The fluxes in y
     * @param[in]  dt        The timestep
     */
    auto block_update = [] (auto solution, auto fx, auto fy, auto cell_areas, auto dt)
    {
        return [=] (auto tree_index)
        {
            auto u0 = solution.conserved.at(tree_index);

            if (u0.size() == 0)
                return u0 | nd::to_shared();

            auto lx = fx.at(tree_index) | nd::difference_on_axis(0);
            auto ly = fy.at(tree_index) | nd::difference_on_axis(1);
            auto dA = cell_areas.at(tree_index);

            auto result = u0 - (lx + ly) * dt / dA;
            return result | nd::to_shared();
        };
    };


    // ========================================================================
    auto u0  =  solution.conserved;
    auto w0  =  u0.map([] (auto Q) { return Q | nd::map(recover_primitive); });

    auto cell_areas = solution.vertices.map([] (auto block)
    {
        if (block.size() == 0)
            return nd::array_t<nd::shared_provider_t<mara::dimensional_value_t<2, 0, 0, double>, 2>>();

        auto dx = block | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = block | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx | nd::multiply(dy) | nd::to_shared();
    });


    // Extend for ghost-cells and get fluxes with specified riemann solver
    // ========================================================================
    auto w0_full = iso_mpi::mpi_fill_tree(w0.map([] (auto b) { return b.shared(); }), mpi_setup);  
    auto w0_ex   = w0.pair_indexes().map(extend(w0_full, 0, guard_zone_count));
    auto w0_ey   = w0.pair_indexes().map(extend(w0_full, 1, guard_zone_count));
    
    auto fhat   = w0.indexes().map(block_fluxes(solution, w0_ex, w0_ey));
    auto fhat_x = fhat.map([] (auto t) { return std::get<0>(t); });
    auto fhat_y = fhat.map([] (auto t) { return std::get<1>(t); });


    // Updated conserved densities
    //=========================================================================
    auto u1 = u0.indexes().map(block_update(solution, fhat_x, fhat_y, cell_areas, dt));


    // Updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        solution.vertices,
        u1,
    };
}




mara::unit_time<double> get_timestep(const iso_mpi::solution_t& s, double cfl)
{

    auto get_min_spacing  = [] (auto verts)
    {
        if (verts.size() == 0)
        {
            return mara::make_length(1e3);
        }

        auto min_dx = verts | component<0>() | nd::difference_on_axis(0) | nd::min();
        auto min_dy = verts | component<1>() | nd::difference_on_axis(1) | nd::min();
        return std::min(min_dx, min_dy);
    };

    auto get_max_velocity = [] (auto block)
    {
        if (block.size() == 0)
        {
            return mara::make_velocity(1.0);
        }

        auto vmax =  block | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_magnitude)) | nd::max();
        return std::max(vmax, mara::make_velocity(1.0));
    };


    auto v = s.vertices;
    auto w = s.conserved.map([] (auto U) { return U | nd::map(recover_primitive); });

    auto lmin = v.map(get_min_spacing).min();
    auto vmax = w.map(get_max_velocity).max();

    double my_max_dt = (lmin / vmax * cfl).value;


    double global_dt = mpi::comm_world().all_reduce(my_max_dt, mpi::operation::min);

    return mara::make_time(global_dt);
}




// ============================================================================
auto iso_mpi::simulation_should_continue(const state_t& state)
{
    return state.solution.time < state.run_config.get_double("tfinal");
}

iso_mpi::solution_t iso_mpi::next_solution(const state_t& state, const mpi_setup_t& mpi)
{
    auto s0  = state.solution;
    auto dt = get_timestep(s0, state.run_config.get_double("cfl"));

    switch (state.run_config.get_int("rk_order"))
    {
        case 1:
        {
            return advance(s0, mpi, dt);
        }
        case 2:
        {
            auto b0 = mara::make_rational(1, 2);
            auto s1 = advance(s0, mpi, dt);
            auto s2 = advance(s1, mpi, dt);
            return s1 * b0 + s2 * (1 - b0);
        }
    }
    throw std::invalid_argument("binary::next_solution");
}

iso_mpi::state_t iso_mpi::next_state(const iso_mpi::state_t& state, const iso_mpi::mpi_setup_t& mpi_setup)
{
    return iso_mpi::state_t{
        iso_mpi::next_solution(state, mpi_setup),
        state.run_config
    };
}

void output_solution_h5(const iso_mpi::solution_t& s, std::string fname)
{
	std::cout << "   Outputting: " << fname << std::endl;
	auto group = h5::File(fname, "w" ).open_group("/");

	mara::write(group, "time"      , s.time     );
	mara::write(group, "vertices"  , s.vertices );
	mara::write(group, "conserved" , s.conserved);
}

void output_solution_parallel_h5(const iso_mpi::solution_t& s, const iso_mpi::mpi_setup_t mpi, std::string fname)
{

    std::function<bool(mara::tree_index_t<2>)> is_my_block = [mpi] (auto idx)
    {
        return mpi.decomposition.at(idx) == mpi.comm.rank();
    };
    
    //=========================================================================    
    if (mpi::is_master())
    {
        std::cout << "   Outputting: " << fname << std::endl;
        auto group = h5::File(fname, "w" ).open_group("/");
        mara::write(group, "time", s.time);
        group.close();
    }


    for(auto rank : nd::arange(mpi.comm.size()))
    {     
        if (rank == mpi.comm.rank())
        {
            auto group = h5::File(fname, "r+").open_group("/");
            mara::write(group, "vertices"  , s.vertices , is_my_block);
            mara::write(group, "conserved" , s.conserved, is_my_block);
            group.close();
        }
        mpi.comm.barrier();
    }
}


// ============================================================================
int main(int argc, const char* argv[])
{

    auto session     = mpi::Session();
    auto run_config  = iso_mpi::create_run_config(argc, argv);
    auto mpi_setup   = iso_mpi::create_mpi_setup(run_config);
    auto state       = iso_mpi::create_state(run_config, mpi_setup);
    auto comm        = mpi::comm_world();


    if (mpi::is_master())
    {
        mara::pretty_print(std::cout, "config", run_config);
    }


    // write initial state to a file
    //=========================================================================
    // for (int rank = 0; rank < comm.size(); ++rank)
    // {
    //     if (rank == comm.rank())
    //     {
    //         auto fname = std::string("initial.") + std::to_string(rank) + ".h5";
    //         output_solution_h5(state.solution, fname);
    //     }
    //     comm.barrier();
    // }

    output_solution_parallel_h5(state.solution, mpi_setup, "initial_par.h5");


    while (iso_mpi::simulation_should_continue(state))
    {
        state = iso_mpi::next_state(state, mpi_setup);
        mpi::printf_master(" %d : t = %0.2f \n", state.solution.iteration.as_integral(), state.solution.time.value);
    }
  

    output_solution_parallel_h5(state.solution, mpi_setup, "final_par.h5");


    // write final state to a file
    //=========================================================================
    // for (int rank = 0; rank < comm.size(); ++rank)
    // {
    //     if (rank == comm.rank())
    //     {
    //         auto fname = std::string("final.") + std::to_string(rank) + ".h5";
    //         output_solution_h5(state.solution, fname);
    //     }
    //     comm.barrier();
    // }

    return 0;
}
