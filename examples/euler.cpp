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

#include "../src/app_config.hpp"
#include "../src/app_serialize.hpp"
#include "app_performance.hpp"
#include "../src/core_ndarray.hpp"
#include "../src/core_ndarray_ops.hpp"
#include "../src/core_hdf5.hpp"
#include "../src/physics_iso2d.hpp"

#include "euler.hpp"
#define cs2    1e-4


// ============================================================================
namespace euler
{
	euler::solution_t next_solution( const state_t& state );
	euler::state_t    next_state   ( const state_t& state );
	auto simulation_should_continue( const state_t& state );
}

// ============================================================================

/**
 * @brief 			Create the config template
 * 
 */
mara::config_template_t euler::create_config_template()
{
	return mara::make_config_template()
	 .item("outdir", "hydro_run")		//directory where data products are written
	 .item("cpi",           10.0)		//checkpoint interval
	 .item("tfinal",         1.0)		
	 .item("clf",            0.4)		//courant number 
	 .item("domain_radius",  1.0)		//half-size of square domain
	 .item("N",              100);		//Number of cells in each direction
}

mara::config_t euler::create_run_config( int argc, const char* argv[] )
{
	auto   args = mara::argv_to_string_map( argc, argv );
	return create_config_template().create().update(args);
}

// ============================================================================


/**
 * @brief             Create 2D array of location_2d_t points representing the vertices
 * 
 * @param  run_config Config object
 *
 * @return            2D array of vertices
 */
auto create_vertices( mara::config_t& run_config )
{
	auto N 	 	= run_config.get_int("N");
	auto radius = run_config.get_double("domain_radius");

	auto x_points = nd::linspace(-radius, radius, N+1);
	auto y_points = nd::linspace(-radius, radius, N+1);

	return nd::cartesian_product( x_points, y_points );
}


// euler::solver_data_t  euler::create_solver_data( const mara::config_t& run_config )
// {
// 	auto vertices = create_vertices(run_config); 
// 	auto min_dx   = vertices | nd::difference_on_axis(0) | nd::min();
// 	auto min_dy   = vertices | nd::difference_on_axis(1) | nd::min();

// 	//=========================================================================
// 	auto primitive =  vertices | nd::difference_on_axis(0) | nd::difference_on_axis(1) | nd::map(initial_condition(run_config));
// 	auto velocity  = primitive | nd::map(std::mem_fn( &mara::iso2d::primitive_t::velocity_magnitude ));
// 	auto v_max     = std::max( mara::make_velocity(1.0), velocity|nd::max() );


// 	//=========================================================================
//     auto result = solver_data_t();
//     result.vertices 			 = vertices;
//     result.rk_order 			 = run_config.get_int("rk_order");
//     result.recommended_time_step = std::min(min_dx, min_dy) / max_velocity * run_config.get_double("cfl_number");

// 	return result;
// }


/**
 * @brief			  	Apply initial condition
 * 
 * @param  run_config 	Config object
 * 
 * @return            	A function giving primitive variables at some point of location_2d_t
 *
 * (euler::primitive_field_t)
 */

// euler::primitive_field_t  initial_condition( const mara::config_t& run_config)
// {
// 	return [] (double x, double y)
// 	{
// 		auto x 	  	 = point[0].value;
// 		//auto y 	 	 = point[1].value;

// 		// Shocktube centered on 0
// 		//============================
// 		//auto r2 	  = x*x + y*y;
// 		//auto r  	  = std::sqrt(r);
// 		auto density = x>0 ? 0.0 : 1.0;
// 		auto vx      = 0.0;
// 		auto vy 	 = 0.0;

// 		return mara::iso2d::primitive_t()
// 		 .with_sigma(density)
// 		 .with_velocity_x(vx)
// 		 .with_velocity_y(vy);
// 	};

// }

auto initial_condition( double x, double y )
{
	auto density = x > 0 ? 0.0 : 1.0;
	auto vx 	 = 0.0;
	auto vy 	 = 0.0;

	return mara::iso2d::primitive_t()
	 .with_sigma(density)
     .with_velocity_x(vx)
	 .with_velocity_y(vy);
}


/**
 * @brief             Create an initial solution object according to initial_condition()
 *                    	
 * @param  run_config configuration object
 
 * @return            solution object
 */
euler::solution_t euler::create_solution( const mara::config_t& run_config )
{
	
	//auto conserved = nd::cartesian_product(x_points, y_points) | nd::apply(/*init_cond*/);
	auto vertices  = create_vertices(run_config);
	auto conserved = vertices
		 | nd::midpoint_on_axis(0)
		 | nd::midpoint_on_axis(1)
		 | nd::apply(initial_condition)  
		 | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area)) //prim2cons
         | nd::to_shared();

	return solution_t{ 0.0, 0, vertices, conserved };
}


/**
 * @brief				Creates state object
 * 
 * @param	run_config 	configuration object
 * 
 * @return          	a state object
 */
euler::state_t euler::create_state( const mara::config_t& run_config )
{
	return state_t{
		create_solution(run_config),
		run_config
	};
}

// ============================================================================

euler::solution_t euler::advance( const solution_t& solution, mara::unit_time<double> dt)
{
	//Some helper functions....
	// ========================================================================
   
    /**
     * @brief      An operator on arrays of sequences: takes a single component
     *             of a sequence and returns an array.
     *
     * @param[in]  component  The component to take
     *
     * @return     An array whose value type is the sequence value type
     */
    auto component = [] (std::size_t cmpnt)
    {
    	//TODO: Fix this???
        return nd::map([cmpnt] (auto p) { return std::get<cmpnt>(p); });
    };

    
    /**
     * @brief      Return an array of areas dx * dy from the given vertex
     *             locations.
     *
     * @param[in]  vertices  An array of vertices
     *
     * @return     A new array
     */
    auto area_from_vertices = [] (auto vertices)
    {
    	auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
    	auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
    	return dx * dy;
    };

    /*
     * @brief      Return an array of intercell fluxes by calling the specified
     *             riemann solver
     *
     * @param[in]  riemann_solver  The riemann solver to use
     * @param[in]  axis            The axis to get the fluxes on
     *
     * @return     An array operator that returns arrays of fluxes
     */
    auto intercell_flux = [] (auto riemann_solver, std::size_t axis)
    {
        return [axis, riemann_solver] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(riemann_solver, _1, _2, cs2, cs2, nh);
            return left_and_right_states | nd::apply(riemann);
        };
    };

    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);

/*
    auto f = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);
    auto F = nd::map(f);
    auto w = F(solution.conserved);
*/

	auto v0  =  solution.vertices;
    auto u0  =  solution.conserved;
    auto w0  =  u0 | nd::map(recover_primitive);
    auto dx  =  v0 | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
    auto dy  =  v0 | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
    auto dA  =  v0 | area_from_vertices; //Need a map?


    // Extend for ghost-cells and get fluxes with specified riemann solver
    // ========================================================================
	auto fx  =  w0 | nd::extend_zero_gradient(0) | intercell_flux( mara::iso2d::riemann_hllc, 0) * dy;
	auto fy  =  w0 | nd::extend_zero_gradient(1) | intercell_flux( mara::iso2d::riemann_hllc, 1) * dx;
	auto lx  =  fx | nd::difference_on_axis(0);
	auto ly  =  fy | nd::difference_on_axis(1);

	// Updated conserved densities
    //=========================================================================
    auto u1  = u0 - (lx + ly ) * dt / dA;

    // Updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        v0,
        u1
    };
	
}

// ============================================================================

auto euler::simulation_should_continue( const state_t& state )
{
	return state.solution.time < state.run_config.get_double("tfinal");
}

double get_timestep( const euler::solution_t& s, double cfl )
{
    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);

	auto vertices = s.vertices; 
 	auto min_dx   = vertices | nd::difference_on_axis(0) | nd::min();
 	auto min_dy   = vertices | nd::difference_on_axis(1) | nd::min();

 	//=========================================================================
 	auto primitive =  s.conserved | nd::map(recover_primitive);
 	auto velocity  = primitive | nd::map(std::mem_fn( &mara::iso2d::primitive_t::velocity_magnitude ));
 	auto v_max     = std::max( mara::make_velocity(1.0), velocity|nd::max() );

    return std::min(min_dx, min_dy) / v_max * cfl;
}

euler::solution_t euler::next_solution( const state_t& state )
{
	auto s0 = state.solution;
	auto dt = get_timestep( s0, state.run_config.get_double("cfl_number") );

	switch( state.run_config.get_int("rk_order") )
    {
        case 1:
        {
            return advance(s0, dt);
        }
        case 2:
        {
            auto b0 = mara::make_rational(1, 2);
            auto s1 = advance(s0, dt);
            auto s2 = advance(s1, dt);
            return s0 * b0 + s2 * (1 - b0);
        }
    }
    throw std::invalid_argument("binary::next_solution");
}

euler::state_t euler::next_state( const euler::state_t& state )
{
	return euler::state_t{
		euler::next_solution( state ),
		state.run_config
	};
}



// ============================================================================
// ============================================================================

int main(int argc, const char* argv[])
{
	auto run_config  = euler::create_run_config(argc, argv);
	auto state 		 = euler::create_state(run_config);

	mara::pretty_print( std::cout, "config", run_config );

	while( euler::simulation_should_continue(state) )
	{
		state = euler::next_state(state);
	}

	return 0;
}

