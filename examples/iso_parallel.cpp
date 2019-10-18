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
#define cs2    1e-1
#define ng     1




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




namespace euler
{

    // Type definitions for simplicity later
    // ========================================================================
    using location_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using primitive_field_t = std::function<mara::iso2d::primitive_t(location_2d_t)>;

    using index_tree_t    = mara::arithmetic_binary_tree_t<mara::tree_index_t<2>, 2>;
    using neighbor_tree_t = mara::arithmetic_binary_tree_t<std::map<std::string, mara::linked_list_t<std::size_t>>, 2>;

    template<typename ArrayValueType>
    using quad_tree_t = mara::arithmetic_binary_tree_t<nd::shared_array<ArrayValueType, 2>, 2>;



    // ========================================================================
    struct solution_t
    {
        mara::unit_time<double>                         time=0.0;
        mara::rational_number_t                         iteration=0;
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
        neighbor_tree_t                                 neighbors;
    };




    //=========================================================================
    struct state_t
    {
        solution_t          solution;
        mara::config_t      run_config;
    };




    // Declaration of necessary functions
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

    auto simulation_should_continue(const state_t& state);

    template<typename ValueType>
    auto mpi_fill_tree(const quad_tree_t<ValueType>& block_tree, const mpi_setup_t& mpi_setup);
    // auto mpi_fill_tree(quad_tree_t<mara::iso2d::primitive_t> block_tree, euler::mpi_setup_t& mpi_setup);

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
mara::config_template_t euler::create_config_template()
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

mara::config_t euler::create_run_config( int argc, const char* argv[] )
{
    auto args = mara::argv_to_string_map( argc, argv );
    return create_config_template().create().update(args);
}




//=============================================================================
euler::index_tree_t euler::create_domain_topology(const mara::config_t& run_config)
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
euler::quad_tree_t<euler::location_2d_t> euler::create_vertex_blocks(
    const mara::config_t& run_config,
    const euler::mpi_setup_t& mpi_setup)
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
                | nd::apply([] (double x, double y) { return euler::location_2d_t{x, y}; })
                | nd::to_shared();
        }
        return nd::shared_array<euler::location_2d_t, 2>{};
    };
    return rank_tree.pair_indexes().map(build_my_vertex_blocks);
}




/**
 * @brief     Apply initial condition to a tuple of position coordinates
 *
 */
auto initial_condition_shocktube(euler::location_2d_t position)
{
    auto density = position[0] > 1.0 ? 0.1 : 1.0;
    auto vx      = 0.0;
    auto vy      = 0.0;

    return mara::iso2d::primitive_t()
     .with_sigma(density)
     .with_velocity_x(vx)
     .with_velocity_y(vy);
}

auto initial_condition_cylinder(euler::location_2d_t position)
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
euler::solution_t euler::create_solution(const mara::config_t& run_config, const mpi_setup_t& mpi_setup)
{
    auto vertices = create_vertex_blocks(run_config, mpi_setup);
    auto cell_centers = vertices.map([] (auto block)
    {
        if (block.size() == 0)
        {
            return block | nd::to_shared();
        }
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
euler::mpi_setup_t euler::create_mpi_setup(const mara::config_t& run_config)
{
    // For now a uniformly refined tree of depth 3
    auto topology = create_domain_topology(run_config);
    auto comm     = mpi::comm_world();

    auto decomposition = mara::build_rank_tree<2>(topology, comm.size());
    auto neighbors     = decomposition
                         .indexes()
                         .map([decomposition] (auto idx) { return mara::get_quad_map(decomposition, idx); });

    return {comm, decomposition, neighbors};
}




/**
 * @brief               Creates state object
 *
 * @param   run_config  configuration object
 *
 * @return              a state object
 */
euler::state_t euler::create_state(const mara::config_t& run_config, const mpi_setup_t& mpi_setup)
{
    return state_t{
        create_solution(run_config, mpi_setup),
        run_config
    };
}


//=============================================================================
template<typename ValueType>
auto euler::mpi_fill_tree(const quad_tree_t<ValueType>& block_tree, const mpi_setup_t& mpi_setup)
{
    using message_type_t = std::pair<mara::tree_index_t<2>, nd::shared_array<mara::iso2d::primitive_t, 2>>;


    auto comm       = mpi_setup.comm;
    auto rank_tree  = mpi_setup.decomposition;
    auto neigh_tree = mpi_setup.neighbors;
    auto my_rank    = comm.rank();


    // 1. Accumulate all the indexes I need to fill my block_tree
    std::vector<mara::tree_index_t<2>> index_vector;

    for (auto [idx, rank] : rank_tree.pair_indexes())
    {
        if (rank == my_rank)
        {
            auto map   = neigh_tree.at(idx);
            auto north = map["north"].head();
            auto south = map["south"].head();
            auto east  = map[ "east"].head();
            auto west  = map[ "west"].head();

            if (north != my_rank)
                index_vector.push_back(idx.prev_on(1));

            if (south != my_rank)
                index_vector.push_back(idx.next_on(1));

            if (east != my_rank)
                index_vector.push_back(idx.next_on(0));

            if (west != my_rank)
                index_vector.push_back(idx.prev_on(0));
        }
    }

    // 1a. Keep only unique indexes
    std::vector<mara::tree_index_t<2>>::iterator it;
    it = std::unique(index_vector.begin(), index_vector.end());
    index_vector.resize(distance(index_vector.begin(), it));


    // 2. All_gather this vector of indexes
    auto rank_needs = comm.all_gather(index_vector);

    if (rank_needs.size() != comm.size())
    {
        throw std::logic_error("vector all_gather made wrong size vector");
    }


    // 3. Look through ragged_vector of indeces for requests from me
    //        -> do these isends and keep the request around
    std::vector<mpi::Request> requests;
    for (std::size_t rank=0; rank < comm.size(); ++rank)
    {
        if (rank != my_rank)
        {
            for (auto index : rank_needs[rank])
            {
                if (rank_tree.at(index) == my_rank)
                {
                    //auto block_s = mara::dumps(block_tree.at(index));
                    auto block_pair_s = mara::dumps(std::pair(index, block_tree.at(index)));
                    requests.push_back(comm.isend(block_pair_s, rank, 0));
                }
            }
        }
    }

    // 3a. Make sure all sends  have been issued
    comm.barrier();


    // 4. Look through my vector, get rank that owns that index, post
    //    recv's, and put them in full_tree
    auto full_tree = block_tree;
    for(auto index : index_vector)
    {
        // auto block = mara::loads(comm.recv(rank_tree.at(index), 0));
        // full_tree  = full_tree.insert(index, block);
        auto [idx, block] = mara::loads<message_type_t>(comm.recv(rank_tree.at(index), 0));
        full_tree = full_tree.insert(idx, block);
    }

    // 4a. Make sure all send requests were completed
    // for(std::size_t r = 0; r < requests.size(); ++r)
    // {
    //     if (! requests[r].is_ready())
    //     {
    //         throw std::logic_error("A recieve request was not matched with its expected send...");
    //     }
    // }

    // 5. Return full_tree
    return full_tree.map([] (auto b) { return b.shared(); });
}




/**
 * @note      tree should be the tree of prims after it has been filled with info
 *            from neighboring processes
 */
template<typename TreeType>
static auto extend(TreeType tree, std::size_t axis, std::size_t guard_count)
{
    return [tree, axis, guard_count] (auto ib)
    {
        auto [index, block] = ib;
        if (block.size() == 0)
        {
            return block | nd::to_shared();
        }

        auto C = tree.at(index);
        auto L = mara::get_cell_block(tree, index.prev_on(axis), mara::compose(nd::to_shared(), nd::select_final(guard_count, axis)));
        auto R = mara::get_cell_block(tree, index.next_on(axis), mara::compose(nd::to_shared(), nd::select_first(guard_count, axis)));
        return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis) | nd::to_shared();
    };
};




//=============================================================================
euler::solution_t euler::advance(const solution_t& solution, const mpi_setup_t& mpi_setup, mara::unit_time<double> dt)
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
            auto riemann = std::bind(riemann_solver, _1, _2, cs2, cs2, nh);
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
            {
                return u0 | nd::to_shared();
            }

            auto lx = fx.at(tree_index) | nd::difference_on_axis(0);
            auto ly = fy.at(tree_index) | nd::difference_on_axis(1);
            auto dA = cell_areas.at(tree_index);

            auto result =  u0 - (lx + ly) * dt / dA;
            return result | nd::to_shared();
        };
    };

    // ========================================================================
    auto u0  =  solution.conserved;
    auto w0  =  u0.map([] (auto Q) { return Q | nd::map(recover_primitive); });

    auto cell_areas = solution.vertices.map([] (auto block)
    {
        if(block.size() == 0)
        {
            //return type fo cell_areas...
            return nd::array_t<nd::shared_provider_t<mara::dimensional_value_t<2, 0, 0, double>, 2>>();
        }

        auto dx = block | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = block | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx | nd::multiply(dy) | nd::to_shared();
    });

    // Extend for ghost-cells and get fluxes with specified riemann solver
    // ========================================================================
    // auto ng = 1;  // number of ghost cells
    auto w0_full = euler::mpi_fill_tree(w0.map([] (auto b) { return b.shared(); }), mpi_setup);  
    auto w0_ex   = w0.pair_indexes().map(extend(w0_full, 0, ng));
    auto w0_ey   = w0.pair_indexes().map(extend(w0_full, 1, ng));
    
    // auto extend_local = [] (auto axis)
    // {
    //     return [axis] (auto block)
    //     {
    //         if(block.size() == 0) return block | nd::to_shared();
    //         return block | nd::extend_periodic_on_axis(axis) | nd::to_shared();
    //     };

    // };
    // auto w0_ex = w0.map(extend_local(0));
    // auto w0_ey = w0.map(extend_local(1));

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
        // u1.map([] (auto i) { return i.shared(); })
        u1
    };
}




mara::unit_time<double> get_timestep(const euler::solution_t& s, double cfl)
{
    // 1. get max allowed timestep on each process
    // 
    // 2. do communication to get minimum of all processes timesteps
    // 
    // 3. return this minimum
    

    // auto get_min_spacing  = [] (auto verts)
    // {
    //     if (verts.size() == 0)
    //     {
    //         return mara::make_length(1e3);
    //     }

    //     auto min_dx = verts | component<0>() | nd::difference_on_axis(0) | nd::min();
    //     auto min_dy = verts | component<1>() | nd::difference_on_axis(1) | nd::min();
    //     return std::min(min_dx, min_dy);
    // };

    // auto get_max_velocity = [] (auto block)
    // {
    //     if (block.size() == 0)
    //     {
    //         return mara::make_velocity(1.0);
    //     }

    //     auto vmax =  block | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_magnitude)) | nd::max();
    //     return std::max(vmax, mara::make_velocity(1.0));
    // };


    // auto v = s.vertices;
    // auto w = s.conserved.map([] (auto U) { return U | nd::map(recover_primitive); });

    // auto lmin = v.map(get_min_spacing).min();
    // auto vmax = w.map(get_max_velocity).max();

    // auto my_max_dt = lmin / vmax * cfl;


    // auto v      = s.vertices;
    // auto min_dx = v.map([] (auto v) { return v | component<0>() | nd::difference_on_axis(0) | nd::min(); }).min();
    // auto min_dy = v.map([] (auto v) { return v | component<1>() | nd::difference_on_axis(1) | nd::min(); }).min();

    // auto get_velocity_mag = std::mem_fn(&mara::iso2d::primitive_t::velocity_magnitude);

    // auto primitive = s.conserved.map([] (auto Q) { return Q | nd::map(recover_primitive); });
    // auto velocity  = primitive.map([get_velocity_mag] (auto W) { return W | nd::map(get_velocity_mag); });
    // auto v_max     = velocity.map([] (auto v) { return std::max(v | nd::max(), mara::make_velocity(1.0)); }).max();

    // return std::min(min_dx, min_dy) / v_max * cfl;

    double local_dt = 0.01;
    double global_dt = local_dt; //mpi::comm_world().all_reduce(local_dt, mpi::operation::min);

    return mara::make_time(global_dt);
}




// ============================================================================
auto euler::simulation_should_continue(const state_t& state)
{
    return state.solution.time < state.run_config.get_double("tfinal");
}

euler::solution_t euler::next_solution(const state_t& state, const mpi_setup_t& mpi)
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

euler::state_t euler::next_state(const euler::state_t& state, const euler::mpi_setup_t& mpi_setup)
{
    return euler::state_t{
        euler::next_solution(state, mpi_setup),
        state.run_config
    };
}

void output_solution_h5(const euler::solution_t& s, std::string fname)
{
    // TODO: write parallel I/O
	std::cout << "   Outputting: " << fname << std::endl;
	auto group = h5::File(fname, "w" ).open_group("/");

	mara::write(group, "time"      , s.time     );
	mara::write(group, "vertices"  , s.vertices );
	mara::write(group, "conserved" , s.conserved);
}




// ============================================================================
int main(int argc, const char* argv[])
{

    auto session     = mpi::Session();
    auto run_config  = euler::create_run_config(argc, argv);
    auto mpi_setup   = euler::create_mpi_setup(run_config);
    auto state       = euler::create_state(run_config, mpi_setup);
    auto comm        = mpi::comm_world();


    if (mpi::is_master())
    {
        mara::pretty_print(std::cout, "config", run_config);
    }


    // write initial state to a file
    //=========================================================================
    for (int rank = 0; rank < comm.size(); ++rank)
    {
        if (rank == comm.rank())
        {
            auto fname = std::string("initial.") + std::to_string(rank) + ".h5";
            output_solution_h5(state.solution, fname);
        }
        comm.barrier();
    }


    while (euler::simulation_should_continue(state))
    {
        state = euler::next_state(state, mpi_setup);
        mpi::printf_master(" %d : t = %0.2f \n", state.solution.iteration.as_integral(), state.solution.time.value);
    }
  


    // write final state to a file
    //=========================================================================
    for (int rank = 0; rank < comm.size(); ++rank)
    {
        if (rank == comm.rank())
        {
            auto fname = std::string("final.") + std::to_string(rank) + ".h5";
            output_solution_h5(state.solution, fname);
        }
        comm.barrier();
    }

    

    // auto prim_tree =  state.solution.conserved.map([] (auto Q) { return Q | nd::map(recover_primitive) | nd::to_shared(); });
    // auto full_tree =  euler::mpi_fill_tree(prim_tree, mpi_setup);


    // // Output the old and new number of blocks on each process
    // //=========================================================================
    // auto is_a_block = [] (auto block) { return block.size() == 0 ? 0 : 1; };
    // auto orig_num   = prim_tree.map(is_a_block).sum();
    // auto finl_num   = full_tree.map(is_a_block).sum();
    // for(auto rank=0; rank < comm.size(); ++rank)
    // {
    //     if(rank == comm.rank())
    //         std::printf("rank %d: (%d, %d)\n", comm.rank(), orig_num, finl_num);
    //     comm.barrier();
    // }
    // //=========================================================================



    // // Make sure that my blocks now have all their neighbors
    // //=========================================================================
    // auto has_neighbors = [] (auto tree)
    // {
    //     return [tree] (auto ib)
    //     {
    //         auto [index, block] = ib;
    //         int  count = 0;
    //         if(block.size() != 0)
    //         {
    //             auto north = index.prev_on(1);
    //             auto south = index.next_on(1);
    //             auto east  = index.next_on(0);
    //             auto west  = index.prev_on(0);
    //             if (tree.at(north).size() == 0) ++count;
    //             if (tree.at(south).size() == 0) ++count;
    //             if (tree.at(east ).size() == 0) ++count;
    //             if (tree.at(west ).size() == 0) ++count;
    //         }
    //         return count;
    //     };
    // };
    // for(auto rank=0; rank < comm.size(); ++rank)
    // {
    //     if(rank == comm.rank())
    //         std::printf("rank %d has_neighbors == false: %d\n", 
    //                     comm.rank(),
    //                     prim_tree.pair_indexes().map(has_neighbors(full_tree)).sum());
    //     comm.barrier();
    // }
    // //=========================================================================


    // // Do an extend on my blocks
    // //=========================================================================
    // auto w0_ex   = prim_tree.pair_indexes().map(extend(full_tree, 0, 1));
    // auto w0_ey   = prim_tree.pair_indexes().map(extend(full_tree, 1, 1));



    return 0;
}