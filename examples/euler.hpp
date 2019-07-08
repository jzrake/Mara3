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
#include "../src/app_performance.hpp"
#include "../src/app_schedule.hpp"
#include "../src/app_subprogram.hpp"
#include "../src/app_filesystem.hpp"
#include "../src/app_parallel.hpp"
#include "../src/core_ndarray.hpp"
#include "../src/core_ndarray_ops.hpp"
#include "../src/core_hdf5.hpp"
#include "../src/physics_iso2d.hpp"

// MIGHT NEED SOME OTHER FILES INCLUDED


namespace euler{

    // Type definitions for simplicity later
    // ============================================================================
    using location_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using primitive_field_t = std::function<mara::iso2d::primitive_t(location_2d_t)>;


    // Solver structs
    // ============================================================================


    // struct solver_data_t
    // {
    //     int                                            rk_order;
    //     mara::unit_time  <double>                      recommended_time_step;
    //     nd::shared_array<location_2d_t>                vertices;
    // };

    struct solution_t
    {
    	mara::unit_time<double> 					           time=0.0;
    	mara::rational_number_t						           iteration=0;
        //location_2d_t                                          vertices;
        nd::shared_array<double, 2>                            vertices;
    	nd::shared_array<mara::iso2d::conserved_per_area_t, 2> conserved;
    	
        // Overload operators to manipulate solution_t types
    	//=====================================================================
            solution_t operator+(const solution_t& other) const
            {
                //auto evaluate = mara::evaluate_on<MARA_PREFERRED_THREAD_COUNT>();

                return {
                    time       + other.time,
                    iteration  + other.iteration,
                    vertices,                                   //Added vertices to solution_t
                    conserved  + other.conserved | evaluate,
                    //conserved  + other.conserved
                };
            }
            solution_t operator*(mara::rational_number_t scale) const
            {
                //auto evaluate = mara::evaluate_on<MARA_PREFERRED_THREAD_COUNT>();

                return {
                    time      * scale.as_double(),
                    iteration * scale,
                    vertices,                                   //Added vertices to solution_t
                    conserved * scale.as_double() | evaluate,
                };
            }
    };

    struct state_t
    {
    	solution_t			solution;
    	mara::config_t 		run_config;
    };

    // Declaration of necessary functions
    //=========================================================================
    mara::config_template_t     create_config_template();
    mara::config_t              create_run_config( int argc, const char* argv[] );
    //mara::schedule_t          create_schedule(...);
    nd::shared_array<double,2>  create_vertices( const mara::config_t& run_config );
    solution_t                  create_solution( const mara::config_t& run_config );
    state_t                     create_state   ( const mara::config_t& run_config );

    solution_t                  advance( const solution_t&, mara::unit_time<double> dt );
}