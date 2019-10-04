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
#include "app_config.hpp"
#include "app_serialize.hpp"
#include "app_performance.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "core_hdf5.hpp"
#include "physics_iso2d.hpp"

#define cs2    1e-1




// ============================================================================
//                                  Header 
// ============================================================================


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

//=============================================================================




namespace euler
{


    // Type definitions for simplicity later
    // ========================================================================
    using location_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using primitive_field_t = std::function<mara::iso2d::primitive_t(location_2d_t)>;




    // Solver structs
    // ========================================================================
    struct solution_t
    {
        mara::unit_time<double>                                time=0.0;
        mara::rational_number_t                                iteration=0;
        nd::shared_array<location_2d_t, 2>                     vertices;
        nd::shared_array<mara::iso2d::conserved_per_area_t, 2> conserved;




        // Overload operators to manipulate solution_t types
        //=====================================================================
        solution_t operator+(const solution_t& other) const
        {
            return {
                time       + other.time,
                iteration  + other.iteration,
                vertices,
                conserved  + other.conserved | nd::to_shared(),
            };
        }
        solution_t operator*(mara::rational_number_t scale) const
        {
            return {
                time      * scale.as_double(),
                iteration * scale,
                vertices,
                conserved * scale.as_double() | nd::to_shared(),
            };
        }
    };




    struct state_t
    {
        solution_t          solution;
        mara::config_t      run_config;
    };




    // Declaration of necessary functions
    //=========================================================================
    mara::config_template_t             create_config_template();
    mara::config_t                      create_run_config( int argc, const char* argv[] );
    nd::shared_array<location_2d_t, 2>  create_vertices( const mara::config_t& run_config );
    solution_t                          create_solution( const mara::config_t& run_config );
    state_t                             create_state   ( const mara::config_t& run_config );
    solution_t                          advance( const solution_t&, mara::unit_time<double> dt );
    euler::solution_t next_solution( const state_t& state );
    euler::state_t    next_state   ( const state_t& state );
    auto simulation_should_continue( const state_t& state );
}




// ============================================================================
//                               Body 
// ============================================================================




/**
 * @brief      An operator on arrays of sequences: takes a single component
 *             of a sequence and returns an array.
 *
 * @param[in]  component  The component to take
 *
 * @return     An array whose value type is the sequence value type
 */
auto component(std::size_t cmpnt)
{
	//WHEN PULL NEW VERSION:
	//   going to need to become a template that instead of std::size_t will need something else...
    return nd::map([cmpnt] (auto p) { return p[cmpnt]; });
};




auto recover_primitive(const mara::iso2d::conserved_per_area_t& conserved)
{
    return mara::iso2d::recover_primitive(conserved);
}




/**
 * @brief      Create the config template
 *
 */
mara::config_template_t euler::create_config_template()
{
    return mara::make_config_template()
     .item("outdir", "hydro_run")       // directory where data products are written
     .item("cpi",           10.0)       // checkpoint interval
     .item("rk_order",         1)		// timestepping order
     .item("tfinal",         1.0)       
     .item("cfl",            0.4)       // courant number 
     .item("domain_radius",  1.0)       // half-size of square domain
     .item("N",              100);      // number of cells in each direction
}




mara::config_t euler::create_run_config( int argc, const char* argv[] )
{
    auto args = mara::argv_to_string_map( argc, argv );
    return create_config_template().create().update(args);
}




/**
 * @brief             Create 2D array of location_2d_t points representing the vertices
 * 
 * @param  run_config Config object
 *
 * @return            2D array of vertices
 */
nd::shared_array<euler::location_2d_t, 2> euler::create_vertices( const mara::config_t& run_config )
{
    auto N      = run_config.get_int("N");
    auto radius = run_config.get_double("domain_radius");

    auto x_points = nd::linspace(-radius, radius, N+1);
    auto y_points = nd::linspace(-radius, radius, N+1);

    return nd::cartesian_product( x_points, y_points )
    | nd::apply([] (double x, double y) { return euler::location_2d_t{x, y}; })
    | nd::to_shared();
}




/**
 * @brief               Apply initial condition
 * 
 * @param  run_config   Config object
 * 
 * @return              A function giving primitive variables at some point of location_2d_t
 *
 */
auto initial_condition_shocktube(euler::location_2d_t position)
{
    auto density = position[0] > 0.0 ? 0.1 : 1.0; 
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

	auto r  = x*x + y*y;
	auto vx = 0.0;
	auto vy = 0.0;

	auto density = r<0.2 ? 1.0 : 0.1;

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
    
    auto vertices  = create_vertices(run_config);
    auto conserved = vertices
         | nd::midpoint_on_axis(0)
         | nd::midpoint_on_axis(1)
         | nd::map(initial_condition_cylinder)
         | nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area)) // prim2cons
         | nd::to_shared();
    return solution_t{ 0.0, 0, vertices, conserved };
}




/**
 * @brief               Creates state object
 * 
 * @param   run_config  configuration object
 * 
 * @return              a state object
 */
euler::state_t euler::create_state( const mara::config_t& run_config )
{
    return state_t{
        create_solution(run_config),
        run_config
    };
}




euler::solution_t euler::advance( const solution_t& solution, mara::unit_time<double> dt)
{


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


    //auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);
    auto v0  =  solution.vertices;
    auto u0  =  solution.conserved;
    auto w0  =  u0 | nd::map(recover_primitive);
    auto dx  =  v0 | component(0) | nd::difference_on_axis(0);
    auto dy  =  v0 | component(1) | nd::difference_on_axis(1);
    auto dA  =  v0 | area_from_vertices;


    // Extend for ghost-cells and get fluxes with specified riemann solver
    // ========================================================================
    auto fx  =  w0 | nd::extend_zero_gradient(0) | nd::zip_adjacent2_on_axis(0) | intercell_flux(0);
    auto fy  =  w0 | nd::extend_zero_gradient(1) | nd::zip_adjacent2_on_axis(1) | intercell_flux(1);
    auto lx  =  fx | nd::multiply(dy) | nd::difference_on_axis(0);
    auto ly  =  fy | nd::multiply(dx) | nd::difference_on_axis(1);


    // Updated conserved densities
    //=========================================================================
    auto u1  = u0 - (lx + ly ) * dt / dA;


    // Updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        v0,
        u1.shared(),
    };
}




// ============================================================================
auto euler::simulation_should_continue( const state_t& state )
{
    return state.solution.time < state.run_config.get_double("tfinal");
}




mara::unit_time<double> get_timestep( const euler::solution_t& s, double cfl )
{
    //return 0.01;

    //auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);

    auto min_dx   = s.vertices | component(0) | nd::difference_on_axis(0) | nd::min();
    auto min_dy   = s.vertices | component(1) | nd::difference_on_axis(1) | nd::min();

    //=========================================================================
    auto primitive =  s.conserved | nd::map(recover_primitive);
    auto velocity  = primitive | nd::map(std::mem_fn( &mara::iso2d::primitive_t::velocity_magnitude ));
    auto v_max     = std::max( mara::make_velocity(1.0), velocity|nd::max() );

    return std::min(min_dx, min_dy) / v_max * cfl;
}




euler::solution_t euler::next_solution( const state_t& state )
{
    auto s0 = state.solution;
    auto dt = get_timestep( s0, state.run_config.get_double("cfl") );

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




void output_solution_h5( const euler::solution_t& s, std::string fname )
{	
	std::cout << "   Outputting: " << fname << std::endl;
	auto group = h5::File( fname, "w" ).open_group("/");
    //auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);

	mara::write(group, "time"      , s.time      );
	mara::write(group, "vertices"  , s.vertices  );
	mara::write(group, "conserved" , s.conserved );
	//h5f.write( "primitive" , s.conserved | nd::map(recover_primitive) | nd::to_shared() );
}


// ============================================================================



 
int main(int argc, const char* argv[])
{
    auto run_config  = euler::create_run_config(argc, argv);
    auto state       = euler::create_state(run_config);

    mara::pretty_print( std::cout, "config", run_config );
    output_solution_h5( state.solution, "initial.h5" );

    while( euler::simulation_should_continue(state) )
    {
        state = euler::next_state(state);

        //if( (state.solution.iteration)%1  == 0)
		printf( " %d : t = %0.2f \n", state.solution.iteration.as_integral(), state.solution.time.value );

    }

    output_solution_h5( state.solution, "output.h5" );

    return 0;
}
