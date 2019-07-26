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
#include "physics_mhd.hpp"

#define gamma_law_index 1.4

// ============================================================================
//                                  Header 
// ============================================================================


namespace mhd_2dCT
{


    // Type definitions for simplicity later
    // ========================================================================
    using location_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1 , 0,  0, double>, 2>;
    using velocity_2d_t     = mara::arithmetic_sequence_t<mara::dimensional_value_t<1 , 0, -1, double>, 2>;
    using primitive_field_t = std::function<mara::mhd::primitive_t(location_2d_t)>;

    // Solver structs
    // ========================================================================
    struct solution_t
    {
        mara::unit_time<double>                                     time=0.0;
        mara::rational_number_t                                     iteration=0;
        nd::shared_array<location_2d_t, 2>                          vertices;
        nd::shared_array<mara::mhd::conserved_density_euler_t, 2>   conserved;
        nd::shared_array<mara::mhd::unit_field,                2>   bfield_x;
        nd::shared_array<mara::mhd::unit_field,                2>   bfield_y;
        nd::shared_array<mara::mhd::unit_field,                2>   bfield_z;

        // Overload operators to manipulate solution_t types
        //=====================================================================
        solution_t operator+(const solution_t& other) const
        {
            return {
                time       + other.time,
                iteration  + other.iteration,
                vertices,
                conserved  + other.conserved | nd::to_shared(),
                bfield_x   + other.bfield_x  | nd::to_shared(),
                bfield_y   + other.bfield_y  | nd::to_shared(),
                bfield_z   + other.bfield_z  | nd::to_shared(),
            };
        }
        solution_t operator*(mara::rational_number_t scale) const
        {
            return {
                time      * scale.as_double(),
                iteration * scale,
                vertices,
                conserved * scale.as_double() | nd::to_shared(),
                bfield_x  * scale.as_double() | nd::to_shared(),
                bfield_y  * scale.as_double() | nd::to_shared(),
                bfield_z  * scale.as_double() | nd::to_shared(),
            };
        }
    };

    struct diagnostic_fields_t
    {
        mara::config_t                                run_config;
        mara::unit_time<double>                       time;
        nd::shared_array<location_2d_t,         2>    vertices;
        nd::shared_array<mara::mhd::unit_field, 2>    div_b; 
    };

    struct state_t
    {
        solution_t          solution;
        mara::config_t      run_config;
    };


    // Declaration of necessary functions
    //=========================================================================
    mara::config_template_t             create_config_template();
    mara::config_t                      create_run_config     ( int argc, const char* argv[] );
    nd::shared_array<location_2d_t, 2>  create_vertices       ( const mara::config_t& run_config );
    solution_t                          create_solution       ( const mara::config_t& run_config );
    state_t                             create_state          ( const mara::config_t& run_config );
    
    state_t                             next_state            ( const state_t& state );
    solution_t                          next_solution         ( const state_t& state );
    solution_t                          advance               ( const solution_t& solution, mara::unit_time<double> dt );
    diagnostic_fields_t                 diagnostic_fields     ( const solution_t& solution, mara::config_t& run_config);

    auto simulation_should_continue( const state_t& state );
    void print_run_loop_message    ( const state_t& state);
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
}

/**
 * @brief      An operator on arrays of magnetic fields. Takes three components of a field
 *             and returns a sequence defining a field at a point
 *             
 * @param  b1  Components 1, 2, 3
 * 
 * @return     The magnetic field vector
 */
auto to_magnetic_vector(mara::mhd::unit_field b1, mara::mhd::unit_field b2, mara::mhd::unit_field b3)
{
    return mara::mhd::magnetic_field_t{b1, b2, b3};
};



/**
 * @brief             Build primitives array
 * 
 * @param  conserved  Array of conserved hydro quantities
 * 
 * @param  B          Array of magnetic field vectors
 * 
 * @return            Array of 8 primitive, cell-centered quantities
 */
auto recover_primitive(const mara::mhd::conserved_density_euler_t& conserved,
                       const mara::mhd::magnetic_field_t& B)
{
    double temp_floor = 1e-4;
    return mara::mhd::recover_primitive(conserved, B, gamma_law_index, temp_floor);
}   

/**
 * @brief      Create the config template
 *
 */
mara::config_template_t mhd_2dCT::create_config_template()
{
    return mara::make_config_template()
     .item("outdir", "hydro_run")       // directory where data products are written
     .item("cpi",             10)       // checkpoint interval
     .item("doi",             10)       // diagnostic output interval
     .item("rk_order",         1)		// timestepping order
     .item("tfinal",         1.0)       
     .item("cfl",            0.4)       // courant number 
     .item("domain_radius",  1.0)       // half-size of square domain
     .item("N",              100);      // number of cells in each direction
}

mara::config_t mhd_2dCT::create_run_config( int argc, const char* argv[] )
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
nd::shared_array<mhd_2dCT::location_2d_t, 2> mhd_2dCT::create_vertices( const mara::config_t& run_config )
{
    auto N      = run_config.get_int("N");
    auto radius = run_config.get_double("domain_radius");

    auto x_points = nd::linspace(-radius, radius, N+1);
    auto y_points = nd::linspace(-radius, radius, N+1);

    return nd::cartesian_product( x_points, y_points )
    | nd::apply([] (double x, double y) { return mhd_2dCT::location_2d_t{x, y}; })
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
auto initial_condition(mhd_2dCT::location_2d_t position)
{
    auto x = position[0];
    auto nexus = x < 0.0;

    auto density  = nexus ? 1.0  :  0.125; 
    auto pressure = nexus ? 1.0  :  0.1;
    auto vx       = nexus ? 0.0  :  0.0;
    auto vy       = nexus ? 0.0  :  0.0;
    auto vz       = nexus ? 0.0  :  0.0;
    auto bx       = nexus ? 0.75 :  0.75;
    auto by       = nexus ? 1.00 : -1.0;
    auto bz       = nexus ? 0.00 :  0.0;

    return mara::mhd::primitive_t()
     .with_mass_density(density)
     .with_gas_pressure(pressure)
     .with_velocity_1(vx)
     .with_velocity_2(vy)
     .with_velocity_3(vz)
     .with_bfield_1(bx)
     .with_bfield_2(by)
     .with_bfield_3(bz);
}

auto kelvin_helmholtz(mhd_2dCT::location_2d_t position)
{
    if( gamma_law_index!= 1.4 )
        throw std::invalid_argument("wrong gamma: for this problem gamma=1.4");

    auto x     = position[0].value;
    auto y     = position[1].value;
    auto nexus = std::abs(y) < 0.3;

    auto density  = nexus ? 2.0  :  1.0; 
    auto vx       = nexus ? 0.5  : -0.5;

    auto amp    = 0.01;
    auto pert_y = amp * std::sin(x);

    auto pressure = 2.5;
    auto vy       = pert_y;
    auto vz       = 0.0;
    auto bx       = 0.5;   
    auto by       = 0.0;
    auto bz       = 0.0;

    return mara::mhd::primitive_t()
     .with_mass_density(density)
     .with_gas_pressure(pressure)
     .with_velocity_1(vx)
     .with_velocity_2(vy)
     .with_velocity_3(vz)
     .with_bfield_1(bx)
     .with_bfield_2(by)
     .with_bfield_3(bz);
}

auto orzsag_tang_vortex(mhd_2dCT::location_2d_t position)
{
    auto x = position[0];
    auto y = position[1];

    auto density = 1.0;
    auto v0      = 1.0;
    auto p0      = 1. / gamma_law_index;
    auto b0      = 1. / gamma_law_index;

    auto vx = -v0 * std::sin(2*M_PI*y.value);
    auto vy =  v0 * std::sin(2*M_PI*x.value);
    auto vz =  0.0;

    auto bx = -b0 * std::sin(2*M_PI*y.value);
    auto by = -b0 * std::sin(4*M_PI*x.value);
    auto bz =  0.0;


    return mara::mhd::primitive_t()
     .with_mass_density(density)
     .with_gas_pressure(p0)
     .with_velocity_1(vx)
     .with_velocity_2(vy)
     .with_velocity_3(vz)
     .with_bfield_1(bx)
     .with_bfield_2(by)
     .with_bfield_3(bz);
}

auto blast_wave(mhd_2dCT::location_2d_t position)
{
    if( gamma_law_index!= 1.4 )
        throw std::invalid_argument("wrong gamma: for this problem gamma=1.4");

    auto x = position[0];
    auto y = position[1];
    auto r = std::sqrt(x.value * x.value + y.value * y.value);

    auto density = 1.0;

    auto blast    = r < 0.1;
    auto pressure = blast ? 1000. : 0.1;

    auto bx = 28.2;  // = 100/sqrt(4*pi)
    auto by = 0.0;
    auto bz = 0.0;
    
    return mara::mhd::primitive_t()
     .with_mass_density(density)
     .with_gas_pressure(pressure)
     .with_velocity_1(0.0)
     .with_velocity_2(0.0)
     .with_velocity_3(0.0)
     .with_bfield_1(bx)
     .with_bfield_2(by)
     .with_bfield_3(bz);
}


/**
 * @brief             Create an initial solution object according to initial_condition()
 *                      
 * @param  run_config configuration object
 
 * @return            solution object
 */
mhd_2dCT::solution_t mhd_2dCT::create_solution( const mara::config_t& run_config )
{
    // helper function to do cons2prim
    // ========================================================================
    auto to_conserved = [] (auto p) { return p.to_conserved_density_euler(gamma_law_index); };
    
    // helper fxn to get cell-centered magnetic fields
    // ========================================================================
    auto get_field    = [] (auto p) { return p.get_magnetic_field_vector(); };

    auto vertices  = create_vertices(run_config);
    auto primitive = vertices 
            | nd::midpoint_on_axis(0) 
            | nd::midpoint_on_axis(1) 
            | nd::map(orzsag_tang_vortex);
    
    auto conserved = primitive 
            | nd::map(to_conserved)
            | nd::to_shared();
    
    auto bx  = primitive | nd::map(get_field) | component(0) | nd::midpoint_on_axis(0) | nd::extend_zero_gradient(0); 
    auto by  = primitive | nd::map(get_field) | component(1) | nd::midpoint_on_axis(1) | nd::extend_zero_gradient(1);
    auto bz  = primitive | nd::map(get_field) | component(2);

    return solution_t{ 
        0.0,
        0,
        vertices, 
        conserved, 
        bx.shared(), 
        by.shared(), 
        bz.shared() };
}




/**
 * @brief               Creates state object
 * 
 * @param   run_config  configuration object
 * 
 * @return              a state object
 */
mhd_2dCT::state_t mhd_2dCT::create_state( const mara::config_t& run_config )
{
    return state_t{
        create_solution(run_config),
        run_config
    };
}




mhd_2dCT::solution_t mhd_2dCT::advance( const solution_t& solution, mara::unit_time<double> dt)
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
        return [axis, riemann_solver=mara::mhd::riemann_hlle] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(riemann_solver, _1, _2, nh, gamma_law_index);
            return left_and_right_states | nd::apply(riemann);
        };
    };

    // Separate out only the hydro components of a flux array
    // ========================================================================
    auto just_euler_fluxes = [] ()
    {
        return nd::map([] (auto f) { return mara::mhd::flux_vector_euler_t{f[0], f[1], f[2], f[3], f[4]}; });
    };


    // ========================================================================

    auto v0  =  solution.vertices;
    auto u0  =  solution.conserved;
    auto bx0 =  solution.bfield_x;
    auto by0 =  solution.bfield_y;
    auto bz0 =  solution.bfield_z;

    auto dx  =  v0 | component(0) | nd::difference_on_axis(0);
    auto dy  =  v0 | component(1) | nd::difference_on_axis(1);
    auto dA  =  v0 | area_from_vertices;

    auto B   =  nd::zip(bx0|nd::midpoint_on_axis(0), by0|nd::midpoint_on_axis(1), bz0) | nd::apply(to_magnetic_vector);
    auto w0  =  nd::zip(u0,B) | nd::apply(recover_primitive);


    // Extend for ghost-cells and get fluxes with specified riemann solver
    // ========================================================================
    auto FX    =  w0 | nd::extend_periodic_on_axis(0) | nd::zip_adjacent2_on_axis(0) | intercell_flux(0);
    auto FY    =  w0 | nd::extend_periodic_on_axis(1) | nd::zip_adjacent2_on_axis(1) | intercell_flux(1);
    auto lx    =  FX | just_euler_fluxes() | nd::multiply(dy) | nd::difference_on_axis(0);
    auto ly    =  FY | just_euler_fluxes() | nd::multiply(dx) | nd::difference_on_axis(1);
    auto lx_bz =  FX | component(7) | nd::multiply(dy) | nd::difference_on_axis(0);
    auto ly_bz =  FY | component(7) | nd::multiply(dx) | nd::difference_on_axis(1); 

 
    // Meaningless 'charge' to convert from units of flux to units of field/force
    //=========================================================================
    auto e = mara::make_dimensional<4, 0, -2>(1.0);


    // Get z-directed electric fields and calculate corner-centered EMFs
    //=========================================================================
    auto emf_x_edges  = -FX | component(6) | nd::extend_periodic_on_axis(1) | nd::midpoint_on_axis(1); //avg flux of By in x-direction
    auto emf_y_edges  =  FY | component(5) | nd::extend_periodic_on_axis(0) | nd::midpoint_on_axis(0); //avg flux of Bx in y-direction
    auto emf_edges    = (emf_x_edges + emf_y_edges) * 0.5 * e; 


    // Updated conserved quantities and face-centered fields
    //=========================================================================
    auto u1  = u0  - (lx    + ly   ) * dt / dA;
    auto bz1 = bz0 - (lx_bz + ly_bz) * dt / dA * e;
    auto bx1 = bx0 - ( emf_edges|nd::difference_on_axis(1) ) / dy * dt;
    auto by1 = by0 + ( emf_edges|nd::difference_on_axis(0) ) / dx * dt;


    // Updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        v0,
        u1.shared(),
        bx1.shared(),
        by1.shared(),
        bz1.shared(),
    };

}




// ============================================================================
auto mhd_2dCT::simulation_should_continue( const state_t& state )
{
    return state.solution.time.value < state.run_config.get_double("tfinal");
}


mara::unit_time<double> get_timestep( const mhd_2dCT::solution_t& s, double cfl )
{
    //return 0.01;

    //auto recover_primitive = std::bind(mara::mhd::recover_primitive, std::placeholders::_1, 0.0);

    
    //=========================================================================
    auto u0 = s.conserved;
    auto B  = nd::zip(s.bfield_x|nd::midpoint_on_axis(0), s.bfield_y|nd::midpoint_on_axis(1), s.bfield_z) | nd::apply(to_magnetic_vector);
    auto primitive = nd::zip(u0, B) | nd::apply(recover_primitive);

    auto nx     = mara::unit_vector_t::on_axis(0);
    auto ny     = mara::unit_vector_t::on_axis(1);
    auto min_dx = s.vertices | component(0) | nd::difference_on_axis(0) | nd::min();
    auto min_dy = s.vertices | component(1) | nd::difference_on_axis(1) | nd::min();

    auto fast_wave_x = primitive  | nd::map([nx] (auto p ) { return p.fast_wave_speeds(nx, gamma_law_index); });
    auto fast_wave_y = primitive  | nd::map([ny] (auto p ) { return p.fast_wave_speeds(ny, gamma_law_index); });
    auto s_x = fast_wave_x | nd::map([] (auto fw) {return std::max( std::abs(fw.p.value), std::abs(fw.m.value) );}) | nd::max();
    auto s_y = fast_wave_x | nd::map([] (auto fw) {return std::max( std::abs(fw.p.value), std::abs(fw.m.value) );}) | nd::max();
    
    auto s_max = mara::make_velocity( std::max(s_x, s_y) );
    return std::min(min_dx, min_dy) / s_max * cfl;
}

// void test_nan(const mara::mhd_2dCT::solution_t& s, int n )
// {
//     bool is_nan = (auto u) [] 
//             {
//                 if( std::isnan(u[0].value) ) return 1;
//                 if( std::isnan(u[1].value) ) return 1;
//                 if( std::isnan(u[2].value) ) return 1;
//                 if( std::isnan(u[3].value) ) return 1;
//                 if( std::isnan(u[4].value) ) return 1;

//                 return 0;
//             };

//     auto u  = s.conserved;

//     if( nd::any( u|nd::map(is_nan) ) )
//         printf("Conserved went nan... %d\n", n);

//     return;
// }

mhd_2dCT::solution_t mhd_2dCT::next_solution( const state_t& state )
{
    auto s0 = state.solution;

    int n = 0;
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

mhd_2dCT::state_t mhd_2dCT::next_state( const mhd_2dCT::state_t& state )
{
    return mhd_2dCT::state_t{
        mhd_2dCT::next_solution( state ),
        state.run_config
    };
}


// Diagnostics -> right now just div_b
//=============================================================================

mhd_2dCT::diagnostic_fields_t mhd_2dCT::diagnostic_fields( const solution_t& solution, mara::config_t& run_config)
{
    auto v0  =  solution.vertices;
    auto dx  =  v0 | component(0) | nd::difference_on_axis(0);
    auto dy  =  v0 | component(1) | nd::difference_on_axis(1);

    auto bx       =  solution.bfield_x;
    auto by       =  solution.bfield_y;
    auto div_b_x  =  bx | nd::difference_on_axis(0);
    auto div_b_y  =  by | nd::difference_on_axis(1);
    auto div_b    =  div_b_x + div_b_y;

    return diagnostic_fields_t{
        run_config,
        solution.time,
        solution.vertices,
        div_b.shared()
    };
}


// Outputting
//=============================================================================

void output_solution_h5( const mhd_2dCT::solution_t& s, std::string fname )
{	
	std::cout << "   Outputting: " << fname << std::endl;
	auto h5f = h5::File( fname, "w" );

    // auto u0 = s.conserved;
    // auto B  = nd::zip(s.bfield_x|nd::midpoint_on_axis(0), s.bfield_y|nd::midpoint_on_axis(1), s.bfield_z) | nd::apply(to_magnetic_vector);
    // auto primitive = nd::zip(u0, B) | nd::apply(recover_primitive);

    //auto recover_primitive = std::bind(mara::mhd::recover_primitive, std::placeholders::_1, 0.0);
	h5f.write( "time"      , s.time         );
	h5f.write( "vertices"  , s.vertices     );
	h5f.write( "conserved", s.conserved    );
    h5f.write( "gamma"     , gamma_law_index);

	h5f.close();
}

void output_diagnostic_h5( const mhd_2dCT::diagnostic_fields_t& diag, std::string fname )
{
    auto h5f = h5::File( fname, "w" );

    h5f.write("time" , diag.time );
    h5f.write("div_b", diag.div_b);
}

std::string get_checkpoint_filename( int nout )
{
    char            buffer[256]; 
    sprintf(        buffer, "%03d", nout );
    std::string num(buffer);

    return "checkpoint_" + num + ".h5";
}

std::string get_diagnostic_filename( int dout )
{
    char            buffer[256];
    sprintf        (buffer, "%03d", dout);
    std::string num(buffer);

    return "diagnostic_" + num + ".h5";
}


// ============================================================================
int main(int argc, const char* argv[])
{
    auto run_config  = mhd_2dCT::create_run_config(argc, argv);
    auto state       = mhd_2dCT::create_state(run_config);
    auto diag        = mhd_2dCT::diagnostic_fields( state.solution, run_config );

    mara::pretty_print  ( std::cout, "config", run_config );
    output_solution_h5  ( state.solution, "checkpoint_000.h5" );
    output_diagnostic_h5( diag          , "diagnostic_000.h5" );

    int nout = 0;
    int dout = 0;
    double delta_n = run_config.get_double("tfinal") / run_config.get_int("cpi");
    double delta_d = run_config.get_double("tfinal") / run_config.get_int("doi");

    while( mhd_2dCT::simulation_should_continue(state) )
    {
        state = mhd_2dCT::next_state(state);
        diag  = mhd_2dCT::diagnostic_fields( state.solution, run_config );

        // if( state.solution.time.value % 10 == 0 )
        printf( " %d : t = %0.2f \n", state.solution.iteration.as_integral(), state.solution.time.value );

        if( state.solution.time.value / delta_n - nout > 1.0  )
        {
            output_solution_h5( state.solution, get_checkpoint_filename(++nout) );
        }
        if( state.solution.time.value / delta_d - dout > 1.0  )
        {
            //printf("Am I ever actually here??\n");
            output_diagnostic_h5( diag, get_diagnostic_filename(++dout) );
        }
    }

    output_solution_h5( state.solution, "output.h5" );
    
    diag = mhd_2dCT::diagnostic_fields( state.solution, run_config );
    output_diagnostic_h5( diag, get_diagnostic_filename(++dout) );

    return 0;
}
