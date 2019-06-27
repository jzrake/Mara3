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
#include <cmath>

#include "../src/core_ndarray.hpp"
#include "../src/core_ndarray_ops.hpp"

#include "../src/core_hdf5.hpp"
#include "../src/app_serialize.hpp"


/**
 *
 *	General structure for all Mara projects
 *
 */

// ============================================================================

struct config_g
{
	int    ncell;
	double tfinal;

	int output_n;
	int output_N;
};

// Every file will have a struct called state_t
struct state_t
{
	int iteration;
	double time;
	double vx;
	double vy;
	nd::shared_array<double,1> x_vertices;      
	nd::shared_array<double,1> y_vertices;
	nd::shared_array<double,2> concentration; 
};


// ============================================================================

/**
 * @brief generate 2D gaussian peak 
 * 
 */
double initial_condition( double x, double y )
{
	auto r2 = x*x + y*y;
	auto s2 = 0.01;
	return std::exp(-r2 / s2 / 2.0);
}

/**
 *  @brief		Initialize a state struct with a gaussian peak in 2D
 *  
 */

state_t init( const config_g& config )
{
	double vx = 1.0;
	double vy = -1.0;

	auto ncell = config.ncell;
	auto x_vertices = nd::linspace( -1, 1, ncell + 1 );
	auto y_vertices = nd::linspace( -1, 1, ncell + 1 );

	auto x_centers = x_vertices | nd::midpoint_on_axis(0);
	auto y_centers = y_vertices | nd::midpoint_on_axis(0);

	auto concentration = nd::cartesian_product(x_centers, y_centers) | nd::apply(initial_condition);
	
	return { 
	 0,
	 0.0,
	 vx,
	 vy,
	 x_vertices.shared(),
	 y_vertices.shared(),
	 concentration.shared() 
	};
}



/**
 * @brief		Update concentration for the 1D advection equation
 * 				with propogation speed v=1
 *
 * @param		current state object
 *
 * @return 		nd array of new concentrations for initializing next state
 * 				
 */
auto advance_concentration( const state_t& state, double dt )
{
	// Helper functions to extract the 0 or 1 component of an array of tuples
 	auto tuple_index_0 = nd::map([] (auto t) { return std::get<0>(t); });
 	auto tuple_index_1 = nd::map([] (auto t) { return std::get<1>(t); });


 	//Get 'area' of faces
	auto dx = nd::cartesian_product( state.x_vertices, state.y_vertices ) | tuple_index_0 | nd::difference_on_axis(0);
	auto dy = nd::cartesian_product( state.x_vertices, state.y_vertices ) | tuple_index_1 | nd::difference_on_axis(1);
	auto dA = (dx | nd::midpoint_on_axis(1)) * (dy | nd::midpoint_on_axis(0));

	
	//Get cell centers
	auto x_centers = state.x_vertices | nd::midpoint_on_axis(0);
	auto y_centers = state.y_vertices | nd::midpoint_on_axis(0);


	//Define function to get upwind flux
	auto total_upwind_flux = [] ( auto flux_arr, double wave_speed, int axis )
	{
		auto FL = flux_arr | nd::select_axis(axis).from(0).to(1).from_the_end();
		auto FR = flux_arr | nd::select_axis(axis).from(1).to(0).from_the_end();
		auto Fh = wave_speed>0 ? FL:FR;  //if true use FR, and if false use FL

		return Fh | nd::difference_on_axis(axis);
	};


	//Get cell-centered fluxes in x and y direction
	auto x_flux = state.concentration | nd::extend_periodic_on_axis(0) | nd::multiply(state.vx); 
	auto y_flux = state.concentration | nd::extend_periodic_on_axis(1) | nd::multiply(state.vy); 



	//Get upwind fluxes and multiply by 'face' length
	auto x_net_flux = total_upwind_flux( x_flux, state.vx, 0 ) | nd::multiply( dy|nd::midpoint_on_axis(0) );
	auto y_net_flux = total_upwind_flux( y_flux, state.vy, 1 ) | nd::multiply( dx|nd::midpoint_on_axis(1) );


	return state.concentration - (x_net_flux + y_net_flux) * dt / dA | nd::to_shared();
}

/**
 * @brief		Calculates the maximum permitted timestep
 * 
 * @param		current state object
 * 
 * @return 		timestep	
 *
 * 				THIS DOESNT WORK RIGHT NOW
 */
double get_dt( const state_t& state )
{
	double cfl = 0.5;

	// Helper functions to extract the 0 or 1 component of an array of tuples
 	auto tuple_index_0 = nd::map([] (auto t) { return std::get<0>(t); });
 	auto tuple_index_1 = nd::map([] (auto t) { return std::get<1>(t); });

	auto dx  = nd::cartesian_product( state.x_vertices, state.y_vertices ) | tuple_index_0 | nd::difference_on_axis(0);
	auto dy  = nd::cartesian_product( state.x_vertices, state.y_vertices ) | tuple_index_1 | nd::difference_on_axis(1);

	return cfl * std::min( dx|nd::min(), dy|nd::min() ) / std::max( state.vx, state.vy );
}


/**
 * @brief		Update the state one step in time
 *
 * @param		current state
 *
 * @return 		new state object
 * 
 */
state_t next( const state_t& state ) //ensure don't modify state even though passing by address
{
	// Returns a state_t struct that has been updated one step in time
	double dt = get_dt( state );
	//double dt = 0.01;

	return { 
		state.iteration + 1, 
		state.time + dt,
		state.vx,
		state.vy,
		state.x_vertices,
		state.y_vertices,
		advance_concentration( state, dt )
	};
}

// ============================================================================

void output_state_h5( const state_t& state, std::string fname )
{
	std::cout << "   Outputting: " << fname << std::endl;
	auto h5f = h5::File( fname, "w" );

	h5f.write( "t"             , state.time          );
	h5f.write( "x_vertices"    , state.x_vertices    );
	h5f.write( "y_vertices"    , state.y_vertices    );
	h5f.write( "concentration" , state.concentration );

	h5f.close();
}


//=============================================================================

int main()
{
	int    ncell  = 200;
	int    nout   = 50;
	double tfinal = 10.0;
	config_g config{ncell, tfinal, 0, nout};

	auto state = init(config); //Instantiate instance of state struct
	output_state_h5( state, "checkpoint_000.h5" );

	double delta = config.tfinal / config.output_N;
	while( state.time < config.tfinal )
	{
		state = next( state );
		if( state.iteration%10  == 0)
			printf( " %d : t = %0.2f \n", state.iteration, state.time );

		if( state.time / delta - config.output_n > 1.0  )
		{
			config.output_n++;

			//std::stringstream ss;
			//ss << std::setw(3) << std::setfill("0") << config.output_n;
			char buffer[256]; sprintf( buffer, "%03d", config.output_n );
			std::string num(buffer);
			std::string fname = "checkpoint_" + num + ".h5";

			output_state_h5( state, fname );
		}
	}
	return 0;
}

