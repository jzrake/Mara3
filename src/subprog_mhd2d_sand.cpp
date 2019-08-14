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

#include "app_compile_opts.hpp"
#if MARA_COMPILE_SUBPROGRAM_MHD2D_SAND


#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <cassert>
#include <cmath>

#include "app_config.hpp"
#include "app_schedule.hpp"
#include "app_serialize.hpp"
#include "app_subprogram.hpp"
#include "app_filesystem.hpp"
#include "app_performance.hpp"
#include "app_compile_opts.hpp"

#include "core_hdf5.hpp"
#include "core_ndarray.hpp"
#include "core_rational.hpp"
#include "core_ndarray_ops.hpp"
#include "core_linked_list.hpp"

#include "physics_mhd.hpp"
#include "physics_mhd_hlld.hpp"


#define gamma_law_index 1.4

// ============================================================================
//                                  Header 
// ============================================================================


namespace mhd_2d
{
    // Type definitions for simplicity later
    // ========================================================================
    using location_2d_t    =  mara::arithmetic_sequence_t<mara::dimensional_value_t<1 , 0,  0, double>, 2>;
    using velocity_2d_t    =  mara::arithmetic_sequence_t<mara::dimensional_value_t<1 , 0, -1, double>, 2>;
    using flux_function_t  =  std::function<mara::mhd::flux_vector_t(mara::mhd::primitive_t, mara::mhd::primitive_t, mara::unit_vector_t, double)>;
    using init_function_t  =  std::function<mara::mhd::primitive_t(mhd_2d::location_2d_t)>;

    
    // Solver structs
    // ========================================================================
    struct solver_data_t
    {
        double                              cfl;
        nd::shared_array<location_2d_t, 2>  vertices;
        flux_function_t                     riemann_solver;
        bool                                ct_flag;
    };


    struct solution_t
    {
        mara::unit_time<double>                                     time=0.0;
        mara::rational_number_t                                     iteration=0;
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
                // vertices,
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
                // vertices,
                conserved * scale.as_double() | nd::to_shared(),
                bfield_x  * scale.as_double() | nd::to_shared(),
                bfield_y  * scale.as_double() | nd::to_shared(),
                bfield_z  * scale.as_double() | nd::to_shared(),
            };
        }
    };


    struct time_series_sample_t
    {
        mara::unit_time<double>  time;
        mara::mhd::unit_field    max_div_b;
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
        solution_t                                solution;
        mara::schedule_t                          schedule;
        mara::linked_list_t<time_series_sample_t> time_series;
        mara::config_t                            run_config;

        state_t with(const solution_t&                                new_solution)    const { return {new_solution, schedule, time_series, run_config}; }
        state_t with(const mara::schedule_t&                          new_schedule)    const { return {solution, new_schedule, time_series, run_config}; }
        state_t with(const mara::config_t&                            new_run_config)  const { return {solution, schedule, time_series, new_run_config}; }
        state_t with(const mara::linked_list_t<time_series_sample_t>& new_time_series) const { return {solution, schedule, new_time_series, run_config}; }
    };


    // Declaration of necessary functions
    //=========================================================================
    mara::config_template_t             create_config_template();
    mara::config_t                      create_run_config     (int argc, const char* argv[]);
    nd::shared_array<location_2d_t, 2>  create_vertices       (const mara::config_t&);
    solver_data_t                       create_solver_data    (const mara::config_t&);
    solution_t                          create_solution       (const mara::config_t&);
    mara::schedule_t                    create_schedule       (const mara::config_t&);
    state_t                             create_state          (const mara::config_t&);

    
    state_t                             next_state            (const state_t&, const solver_data_t&);
    solution_t                          next_solution         (const state_t&, const solver_data_t&);
    mara::schedule_t                    next_schedule         (const state_t&);
    solution_t                          advance               (const solution_t&, const solver_data_t&, mara::unit_time<double>);
    diagnostic_fields_t                 diagnostic_fields     (const solver_data_t&, const solution_t&, const mara::config_t&);


    auto run_tasks                 (const state_t&, const solver_data_t&);
    void prepare_filesystem        (const mara::config_t&);
    void print_run_loop_message    (const state_t&, mara::perf_diagnostics_t);
    auto simulation_should_continue(const state_t&);
}


// ============================================================================
//                              Config
// ============================================================================
                             

mara::config_template_t mhd_2d::create_config_template()
{
    return mara::make_config_template()
     .item("outdir",           "./")    // directory where data products are written
     .item("cpi",               1.0)    // checkpoint interval (orbits; chkpt.????.h5 - snapshot of app_state)
     .item("dfi",               1.0)    // diagnostic field interval (orbits; diagnostics.????.h5)
     .item("tsi",               1.0)    // time series interval (orbits)
     .item("ct_flag",             0)    // flag for CT
     .item("rk_order",            1)    // timestepping order
     .item("riemann_solver", "hlle")    // riemann solver (hlle or hlld)
     .item("tfinal",            1.0)    
     .item("cfl",               0.4)    // courant number 
     .item("domain_radius",     1.0)    // half-size of square domain
     .item("N",                 100)    // number of cells in each direction
     .item("initial_condition", "orzsag_tang_vortex");
}


mara::config_t mhd_2d::create_run_config( int argc, const char* argv[] )
{
    auto args = mara::argv_to_string_map( argc, argv );
    return create_config_template().create().update(args);
}


// ============================================================================
//                           Helper Functions 
// ============================================================================

/**
 *   Get array of components from array of tuples
 */
auto component(std::size_t cmpnt)
{
    //WHEN PULL NEW VERSION:
    //   going to need to become a template that instead of std::size_t will need something else...
    return nd::map([cmpnt] (auto p) { return p[cmpnt]; });
}


/**
 *   Create a magnetic field vector out of 3 field component arrays
 */
auto to_magnetic_vector(mara::mhd::unit_field b1, mara::mhd::unit_field b2, mara::mhd::unit_field b3)
{
    return mara::mhd::magnetic_field_t{b1, b2, b3};
};


/**
 *   Recover primitives from hydro conserveds and magnetif field vector
*/
auto recover_primitive(const mara::mhd::conserved_density_euler_t& conserved,
                       const mara::mhd::magnetic_field_t& B)
{
    double temp_floor = 1e-4;
    return mara::mhd::recover_primitive(conserved, B, gamma_law_index, temp_floor);
}   


/**
 *   Create grid vertices
 */
nd::shared_array<mhd_2d::location_2d_t, 2> mhd_2d::create_vertices( const mara::config_t& run_config )
{
    auto N      = run_config.get_int("N");
    auto radius = run_config.get_double("domain_radius");

    auto x_points = nd::linspace(-radius, radius, N+1);
    auto y_points = nd::linspace(-radius, radius, N+1);

    return nd::cartesian_product( x_points, y_points )
    | nd::apply([] (double x, double y) { return mhd_2d::location_2d_t{x, y}; })
    | nd::to_shared();
}


// ============================================================================
//                        Initial Condition Functions 
// ============================================================================


/**
 *     The Brio-Wu-Shocktube
 */
auto brio_wu_shocktube(mhd_2d::location_2d_t position)
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


/**
 *     MHD Kelvin-Helmoltz Instability
 */
auto kelvin_helmholtz(mhd_2d::location_2d_t position)
{
    if ( gamma_law_index!= 1.4 )
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


/**
 *     The Orszag-Tang Vortex
 */
auto orzsag_tang_vortex(mhd_2d::location_2d_t position)
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


/**
 *     An MHD Blast-Wave
 */
auto mhd_blast_wave(mhd_2d::location_2d_t position)
{
    if ( gamma_law_index!= 1.4 )
        throw std::invalid_argument("wrong gamma: for this problem gamma=1.4");

    auto x = position[0];
    auto y = position[1];
    auto r = std::sqrt(x.value * x.value + y.value * y.value);

    auto blast    = r < 0.1;
    auto density  =     1.0;
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
 *    The MHD Flywheel Test for HLLD and CT
 *    
 *       - from DISCO code paper (Duffel, 2016)
 */
auto duffel_flywheel(mhd_2d::location_2d_t position)
{
    auto x = position[0].value;
    auto y = position[1].value;

    auto r     = std::sqrt(x * x + y * y);
    auto theta = std::atan2(y , x);
    auto sinth = std::sin(theta);
    auto costh = std::cos(theta);

    auto R       = 0.3;
    auto Om      = 1.5;
    auto density = 1.0;

    auto w  =  Om * std::exp(-0.5 * r * r / R / R);
    auto vp =  w * r;
    auto vx = -vp * sinth;
    auto vy =  vp * costh;

    auto B0 =  vp * std::sqrt(density);
    auto bx = -B0 * sinth;
    auto by =  B0 * costh;

    auto P0       = 1.1 * 0.5 * std::exp(-1) * Om * Om * R * R * density;
    auto pressure = P0 - 0.5 * B0 * B0;

    return mara::mhd::primitive_t()
     .with_mass_density(density)
     .with_gas_pressure(pressure)
     .with_velocity_1(vx)
     .with_velocity_2(vy)
     .with_velocity_3(0.0)
     .with_bfield_1(bx)
     .with_bfield_2(by)
     .with_bfield_3(0.0);
}

auto get_init_map()
{
    std::map<std::string, mhd_2d::init_function_t> init_map;
    init_map.insert(std::make_pair("brio_wu_shocktube" , brio_wu_shocktube ));
    init_map.insert(std::make_pair("kelvin_helmholtz"  , kelvin_helmholtz  ));
    init_map.insert(std::make_pair("orzsag_tang_vortex", orzsag_tang_vortex));
    init_map.insert(std::make_pair("mhd_blast_wave"    , mhd_blast_wave    ));
    init_map.insert(std::make_pair("duffel_flywheel"   , duffel_flywheel   ));

    return init_map;
}


// ============================================================================
//                              Create Solver
// ============================================================================


mhd_2d::solver_data_t mhd_2d::create_solver_data( const mara::config_t& run_config )
{
    std::map<std::string, flux_function_t> solver_map;
    solver_map.insert(std::make_pair("hlle", mara::mhd::riemann_hlle));
    solver_map.insert(std::make_pair("hlld", mara::mhd::riemann_hlld));
    
    auto vertices       = create_vertices(run_config);
    auto riemann_solver = solver_map[run_config.get_string("riemann_solver")];
    double cfl          = run_config.get_double("cfl");
    bool ct_flag        = run_config.get_int("ct_flag");

    return solver_data_t{
        cfl,
        vertices,
        riemann_solver,
        ct_flag,
    };
}


mhd_2d::solution_t mhd_2d::create_solution( const mara::config_t& run_config )
{
    // helper function to do cons2prim
    // ========================================================================
    auto to_conserved = [] (auto p) { return p.to_conserved_density_euler(gamma_law_index); };
    
    // helper fxn to get cell-centered magnetic fields
    // ========================================================================
    auto get_field    = [] (auto p) { return p.get_magnetic_field_vector(); };

    // get initial condition
    // ========================================================================
    auto init_map = get_init_map();
    auto initial_condition = init_map[run_config.get_string("initial_condition")];

    auto vertices  = create_vertices(run_config);
    auto primitive = vertices 
            | nd::midpoint_on_axis(0) 
            | nd::midpoint_on_axis(1) 
            | nd::map(initial_condition);
    
    auto conserved = primitive 
            | nd::map(to_conserved)
            | nd::to_shared();
    
    auto bx  = primitive | nd::map(get_field) | component(0);
    auto by  = primitive | nd::map(get_field) | component(1);
    auto bz  = primitive | nd::map(get_field) | component(2);
    if ( run_config.get_int("ct_flag") )
    {
        return solution_t{
            0.0,
            0,
            conserved,
            bx | nd::midpoint_on_axis(0) | nd::extend_zero_gradient(0) | nd::to_shared(),
            by | nd::midpoint_on_axis(1) | nd::extend_zero_gradient(1) | nd::to_shared(),
            bz.shared()
        };
    }

    return solution_t{ 
        0.0,
        0,
        conserved, 
        bx.shared(), 
        by.shared(), 
        bz.shared() 
    };
}


mara::schedule_t mhd_2d::create_schedule(const mara::config_t& run_config)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_checkpoint");
    schedule.create_and_mark_as_due("write_diagnostics");
    schedule.create_and_mark_as_due("record_time_series");
    return schedule;
}


mhd_2d::state_t mhd_2d::create_state( const mara::config_t& run_config )
{
    return state_t{
        create_solution(run_config),
        create_schedule(run_config),
        {},
        run_config
    };
}


// ============================================================================
//                              The Solver
// ============================================================================
 

mhd_2d::solution_t mhd_2d::advance( const solution_t& solution,
                                        const solver_data_t& solver,
                                        mara::unit_time<double> dt )
{

    // Meaningless 'charge' to convert from units of flux to units of field/force
    //=========================================================================
    auto e = mara::make_dimensional<4, 0, -2>(1.0);


    /**
     *     Extract fluxes of just hydro quantities
     */
    auto just_euler_fluxes = [] ()
    {
        return nd::map([] (auto f) { return mara::mhd::flux_vector_euler_t{f[0], f[1], f[2], f[3], f[4]}; });
    };


    /**
     *     Grid areas calculated from the vertices
     */
    auto area_from_vertices = [] (auto vertices)
    {
        auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };


    /*
     *     Calculate flux between cells
     */
    auto intercell_flux = [riemann_solver=solver.riemann_solver] (std::size_t axis)
    {
        return [axis, riemann_solver] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(riemann_solver, _1, _2, nh, gamma_law_index);
            return left_and_right_states | nd::apply(riemann);
        };
    };


    /**
     *     Zip_adjacent function specifically for MHD solvers
     *
     *        - includes longitudinal field as solver parameter
     */
    auto zip_adjacent2_mhd = [] (auto b_face, std::size_t axis)
    {
        auto set_parallel_field = [] (std::size_t axis) 
        {
            return nd::apply( [axis] (auto P, auto b) {return P.with_b_along(b, axis);} );
        };

        return [=] (auto array)
        {
            auto Pl  = array | nd::select_axis(axis).from(0).to(1).from_the_end();
            auto Pr  = array | nd::select_axis(axis).from(1).to(0).from_the_end();
            auto Plp = nd::zip(Pl, b_face) | set_parallel_field(axis);
            auto Prp = nd::zip(Pr, b_face) | set_parallel_field(axis);
            return zip(Plp,Prp);
        };
    };


    //=========================================================================
    auto v0  =  solver.vertices;
    auto u0  =  solution.conserved;
    auto bx0 =  solution.bfield_x;
    auto by0 =  solution.bfield_y;
    auto bz0 =  solution.bfield_z;


    //=========================================================================
    auto dx  =  v0 | component(0) | nd::difference_on_axis(0);
    auto dy  =  v0 | component(1) | nd::difference_on_axis(1);
    auto dA  =  v0 | area_from_vertices;


    //=========================================================================
    if ( solver.ct_flag )
    {
        // For CT bx0, by0 are face-centered fields
        //    1. Get cell centered field
        //=====================================================================
        auto B   =  nd::zip(bx0  | nd::midpoint_on_axis(0), 
                            by0  | nd::midpoint_on_axis(1), 
                            bz0) | nd::apply(to_magnetic_vector);


        //    2. Pass face-centered field to zip_adjacent2 stencil
        //=====================================================================
        auto w0  =  nd::zip(u0,B) | nd::apply(recover_primitive);
        auto FX  =  w0 | nd::extend_periodic_on_axis(0) | zip_adjacent2_mhd(bx0, 0) | intercell_flux(0);
        auto FY  =  w0 | nd::extend_periodic_on_axis(1) | zip_adjacent2_mhd(by0, 1) | intercell_flux(1);


        //    3. Extract euler fluxes for hydro quantities and B_z
        //=========================================================================
        auto lx    =  FX | just_euler_fluxes() | nd::multiply(dy) | nd::difference_on_axis(0);
        auto ly    =  FY | just_euler_fluxes() | nd::multiply(dx) | nd::difference_on_axis(1);
        auto lx_bz =  FX | component(7) | nd::multiply(dy) | nd::difference_on_axis(0);
        auto ly_bz =  FY | component(7) | nd::multiply(dx) | nd::difference_on_axis(1); 


        //    4. Get z-directed electric fields and calculate corner-centered EMFs
        //=========================================================================
        auto emf_x_edges  = -FX | component(6) | nd::extend_periodic_on_axis(1) | nd::midpoint_on_axis(1); //avg flux of By in x-direction
        auto emf_y_edges  =  FY | component(5) | nd::extend_periodic_on_axis(0) | nd::midpoint_on_axis(0); //avg flux of Bx in y-direction
        auto emf_edges    = (emf_x_edges + emf_y_edges) * 0.5 * e; 


        //    5. Updated conserved quantities and face-centered fields
        //=========================================================================
        auto u1  = u0  - (lx    + ly   ) * dt / dA;
        auto bz1 = bz0 - (lx_bz + ly_bz) * dt / dA * e;
        auto bx1 = bx0 - ( emf_edges|nd::difference_on_axis(1) ) / dy * dt;
        auto by1 = by0 + ( emf_edges|nd::difference_on_axis(0) ) / dx * dt;


        //    6. Return new solution
        //=========================================================================
        return solution_t{
            solution.time + dt,
            solution.iteration + 1,
            u1.shared(),
            bx1.shared(),
            by1.shared(),
            bz1.shared(),
        };
    }
    else
    {
        // For no-CT bx0, by0, bz0 are cell-centered fields
        //    1. Need face centered fields to send to zip_adjacent2 stencil
        //=====================================================================
        auto B   =  nd::zip(bx0, by0, bz0) | nd::apply(to_magnetic_vector);
        auto w0  =  nd::zip(u0,B) | nd::apply(recover_primitive);
        auto bxc =  bx0 | nd::extend_periodic_on_axis(0) | nd::midpoint_on_axis(0);
        auto byc =  by0 | nd::extend_periodic_on_axis(1) | nd::midpoint_on_axis(1);
        auto FX  =  w0  | nd::extend_periodic_on_axis(0) | zip_adjacent2_mhd(bxc, 0) | intercell_flux(0);
        auto FY  =  w0  | nd::extend_periodic_on_axis(1) | zip_adjacent2_mhd(byc, 1) | intercell_flux(1);


        //    2. Get Fluxes in all quantities
        //=========================================================================
        auto lx    =  FX | just_euler_fluxes() | nd::multiply(dy) | nd::difference_on_axis(0);
        auto ly    =  FY | just_euler_fluxes() | nd::multiply(dx) | nd::difference_on_axis(1);
        auto lx_bx =  FX | component(5) | nd::multiply(dy) | nd::difference_on_axis(0);
        auto ly_bx =  FY | component(5) | nd::multiply(dx) | nd::difference_on_axis(1); 
        auto lx_by =  FX | component(6) | nd::multiply(dy) | nd::difference_on_axis(0);
        auto ly_by =  FY | component(6) | nd::multiply(dx) | nd::difference_on_axis(1);
        auto lx_bz =  FX | component(7) | nd::multiply(dy) | nd::difference_on_axis(0);
        auto ly_bz =  FY | component(7) | nd::multiply(dx) | nd::difference_on_axis(1); 


        //    3. Updated conserved quantities and face-centered fields
        //=========================================================================
        auto u1  = u0  - (lx    + ly   ) * dt / dA;
        auto bx1 = bx0 - (lx_bx + ly_bx) * dt / dA * e;
        auto by1 = by0 - (lx_by + ly_by) * dt / dA * e;
        auto bz1 = bz0 - (lx_bz + ly_bz) * dt / dA * e;


        //    4. Return new solution
        //=========================================================================
        return solution_t{
            solution.time + dt,
            solution.iteration + 1,
            u1.shared(),
            bx1.shared(),
            by1.shared(),
            bz1.shared(),
        };
    }

    // Updated solution state
    //=========================================================================
}


/**
 *     Calculate maximum tolerable timestep
 */
mara::unit_time<double> get_timestep( const mhd_2d::solution_t& s, const mhd_2d::solver_data_t& solver )
{
    //=========================================================================
    auto nx     = mara::unit_vector_t::on_axis(0);
    auto ny     = mara::unit_vector_t::on_axis(1);
    auto min_dx = solver.vertices | component(0) | nd::difference_on_axis(0) | nd::min();
    auto min_dy = solver.vertices | component(1) | nd::difference_on_axis(1) | nd::min();


    //=========================================================================
    auto u0  =  s.conserved;
    if (solver.ct_flag)
    {
        auto B  =  nd::zip(s.bfield_x  | nd::midpoint_on_axis(0), 
                           s.bfield_y  | nd::midpoint_on_axis(1),
                           s.bfield_z) | nd::apply(to_magnetic_vector);
        auto primitive    =  nd::zip(u0, B) | nd::apply(recover_primitive);
        auto fast_wave_x  =  primitive  | nd::map([nx] (auto p ) { return p.fast_wave_speeds(nx, gamma_law_index); });
        auto fast_wave_y  =  primitive  | nd::map([ny] (auto p ) { return p.fast_wave_speeds(ny, gamma_law_index); });
        auto s_x   =  fast_wave_x | nd::map([] (auto fw) {return std::max( std::abs(fw.p.value), std::abs(fw.m.value) );}) | nd::max();
        auto s_y   =  fast_wave_y | nd::map([] (auto fw) {return std::max( std::abs(fw.p.value), std::abs(fw.m.value) );}) | nd::max();
        auto s_max =  mara::make_velocity( std::max(s_x, s_y) );
        return std::min(min_dx, min_dy) / s_max * solver.cfl;
    }
    else
    {
        auto B  =  nd::zip(s.bfield_x, s.bfield_y, s.bfield_z) | nd::apply(to_magnetic_vector);
        auto primitive    =  nd::zip(u0, B) | nd::apply(recover_primitive);
        auto fast_wave_x  =  primitive  | nd::map([nx] (auto p ) { return p.fast_wave_speeds(nx, gamma_law_index); });
        auto fast_wave_y  =  primitive  | nd::map([ny] (auto p ) { return p.fast_wave_speeds(ny, gamma_law_index); });
        auto s_x   =  fast_wave_x | nd::map([] (auto fw) {return std::max( std::abs(fw.p.value), std::abs(fw.m.value) );}) | nd::max();
        auto s_y   =  fast_wave_y | nd::map([] (auto fw) {return std::max( std::abs(fw.p.value), std::abs(fw.m.value) );}) | nd::max();
        auto s_max =  mara::make_velocity( std::max(s_x, s_y) );
        return std::min(min_dx, min_dy) / s_max * solver.cfl;
    }
    // Maximum allowable timestep
    //=========================================================================
}


//=============================================================================
//                         Next Functions
//=============================================================================


mhd_2d::solution_t mhd_2d::next_solution(const state_t& state, const solver_data_t& solver_data)
{
    auto s0 = state.solution;
    auto sd = solver_data;
    auto dt = get_timestep( s0, sd );

    switch( state.run_config.get_int("rk_order") )
    {
        case 1:
        {
            return advance(s0, sd, dt);
        }
        case 2:
        {
            auto b0 = mara::make_rational(1, 2);
            auto s1 = advance(s0, sd, dt);
            auto s2 = advance(s1, sd, dt);
            return s0 * b0 + s2 * (1 - b0);
        }
    }
    throw std::invalid_argument("mhd::next_solution");
}


mara::schedule_t mhd_2d::next_schedule(const state_t& state)
{
    return mara::mark_tasks_in(state, state.solution.time.value,
        {{"write_checkpoint",   state.run_config.get_double("cpi") * 2 * M_PI},
         {"write_diagnostics",  state.run_config.get_double("dfi") * 2 * M_PI},
         {"record_time_series", state.run_config.get_double("tsi") * 2 * M_PI}});
}


mhd_2d::state_t mhd_2d::next_state(const state_t& state, const solver_data_t& solver_data)
{
    return state_t{
        next_solution(state, solver_data),
        next_schedule(state),
        state.time_series,
        state.run_config
    };
}


//=============================================================================
//                             Diagnostics 
//                                 -> right now just div_b
//=============================================================================

nd::shared_array<mara::mhd::unit_field, 2>
compute_div_b(const mhd_2d::solver_data_t& solver, const mhd_2d::solution_t& solution, bool ct_flag)
{
    auto v0  =  solver.vertices;
    auto dx  =  v0 | component(0) | nd::difference_on_axis(0);
    auto dy  =  v0 | component(1) | nd::difference_on_axis(1);

    auto bx  =  solution.bfield_x;
    auto by  =  solution.bfield_y;

    if( ct_flag ){
        auto div_b_x  =  bx | nd::difference_on_axis(0);
        auto div_b_y  =  by | nd::difference_on_axis(1);
        auto div_b    =  div_b_x + div_b_y;
        return div_b.shared();
    }
    else
    {
        auto div_b_x  =  bx | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto div_b_y  =  by | nd::difference_on_axis(1) | nd::midpoint_on_axis(0); 
        auto div_b    =  div_b_x + div_b_y;
        return div_b.shared();
    }
}

mhd_2d::diagnostic_fields_t 
mhd_2d::diagnostic_fields(const solver_data_t& solver, const solution_t& solution, const mara::config_t& run_config)
{
    auto div_b = compute_div_b(solver, solution, run_config.get_int("ct_flag"));

    return diagnostic_fields_t{
        run_config,
        solution.time,
        solver.vertices,
        div_b
    };
}



//=============================================================================
//                              Outputting
//=============================================================================


template<>
struct h5::hdf5_type_info<mhd_2d::time_series_sample_t>
{
    using native_type = mhd_2d::time_series_sample_t;
    static auto make_datatype_for(const native_type& value)
    {
        return h5::Datatype::compound<native_type>({
            h5_compound_type_member(native_type, time),
            h5_compound_type_member(native_type, max_div_b),
        });
    }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto convert_to_writable(const native_type& value) { return value; }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(const native_type& value) { return &value; }
    static auto get_address(native_type& value) { return &value; }
};

// -----------------------------------------------------------------------
template<>
void mara::write<mhd_2d::solution_t>(h5::Group& group, std::string name, const mhd_2d::solution_t& solution)
{
    auto location = group.require_group(name);
    mara::write(location, "time",      solution.time);
    mara::write(location, "iteration", solution.iteration);
    mara::write(location, "conserved", solution.conserved);
}

template<>
void mara::write<mhd_2d::state_t>(h5::Group& group, std::string name, const mhd_2d::state_t& state)
{
    auto location = group.require_group(name);
    mara::write(location, "solution",    state.solution);
    mara::write(location, "schedule",    state.schedule);
    mara::write(location, "time_series", state.time_series.reverse());
    mara::write(location, "run_config",  state.run_config);
}

template<>
void mara::write<mhd_2d::diagnostic_fields_t>(h5::Group& group, std::string name, const mhd_2d::diagnostic_fields_t& diagnostics)
{
    auto location = group.require_group(name);
    mara::write(location, "run_config",        diagnostics.run_config);
    mara::write(location, "time",              diagnostics.time);
    mara::write(location, "vertices",          diagnostics.vertices);
    mara::write(location, "div_b",             diagnostics.div_b);
}



// void output_solution_h5( const mhd_2d::solution_t& s, const mhd_2d::solver_data_t solver, std::string fname )
// {	
// 	std::cout << "   Outputting: " << fname << std::endl;
// 	auto h5f = h5::File( fname, "w" );

//     // auto u0 = s.conserved;
//     // auto B  = nd::zip(s.bfield_x|nd::midpoint_on_axis(0), s.bfield_y|nd::midpoint_on_axis(1), s.bfield_z) | nd::apply(to_magnetic_vector);
//     // auto primitive = nd::zip(u0, B) | nd::apply(recover_primitive);

//     //auto recover_primitive = std::bind(mara::mhd::recover_primitive, std::placeholders::_1, 0.0);
// 	h5f.write( "time"      , s.time         );
// 	h5f.write( "vertices"  , solver.vertices);
// 	h5f.write( "conserved" , s.conserved    );
//     h5f.write( "gamma"     , gamma_law_index);

// 	h5f.close();
// }


// void output_diagnostic_h5( const mhd_2d::diagnostic_fields_t& diag, std::string fname )
// {
//     auto h5f = h5::File( fname, "w" );

//     h5f.write("time" , diag.time );
//     h5f.write("div_b", diag.div_b);
// }


// std::string get_checkpoint_filename( int nout )
// {
//     char            buffer[256]; 
//     sprintf(        buffer, "%03d", nout );
//     std::string num(buffer);

//     return "checkpoint_" + num + ".h5";
// }


// std::string get_diagnostic_filename( int dout )
// {
//     char            buffer[256];
//     sprintf        (buffer, "%03d", dout);
//     std::string num(buffer);

//     return "diagnostic_" + num + ".h5";
// }


// ============================================================================
// int main(int argc, const char* argv[])
// {
//     auto run_config  = mhd_2d::create_run_config(argc, argv);
//     auto solver_data = mhd_2d::create_solver_data(run_config);
//     auto state       = mhd_2d::create_state(run_config);
//     auto diag        = mhd_2d::diagnostic_fields(solver_data, state.solution, run_config );


//     mara::pretty_print  ( std::cout, "config", run_config );
//     output_diagnostic_h5( diag,       "diagnostic_000.h5" );
//     output_solution_h5  ( state.solution, solver_data, "checkpoint_000.h5" );


//     int nout = 0;
//     int dout = 0;
//     double delta_n = run_config.get_double("tfinal") / run_config.get_int("cpi");
//     double delta_d = run_config.get_double("tfinal") / run_config.get_int("dfi");


//     while( mhd_2d::simulation_should_continue(state) )
//     {
//         state = mhd_2d::next_state(state, solver_data);
//         diag  = mhd_2d::diagnostic_fields(solver_data, state.solution, run_config );


//         printf( " %d : t = %0.2f \n", state.solution.iteration.as_integral(), state.solution.time.value );
//         if ( state.solution.time.value / delta_n - nout > 1.0  )
//         {
//             output_solution_h5( state.solution, solver_data, get_checkpoint_filename(++nout) );
//         }
//         if ( state.solution.time.value / delta_d - dout > 1.0  )
//         {
//             //printf("Am I ever actually here??\n");
//             output_diagnostic_h5( diag, get_diagnostic_filename(++dout) );
//         }
//     }


//     output_solution_h5( state.solution, solver_data, "output.h5" );

    
//     diag = mhd_2d::diagnostic_fields(solver_data, state.solution, run_config );
//     output_diagnostic_h5( diag, get_diagnostic_filename(++dout) );


//     return 0;
// }


//=============================================================================
//                   Scheduling, Outputting, and Performance
//=============================================================================

auto mhd_2d::run_tasks(const state_t& state, const solver_data_t& solver_data)
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
    auto write_diagnostics = [&solver_data] (const state_t& state)
    {
        auto outdir = state.run_config.get_string("outdir");
        auto count  = state.schedule.num_times_performed("write_diagnostics");
        auto fname  = mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5"));
        auto group  = h5::File(fname, "w").open_group("/");
        mara::write(group, "/", diagnostic_fields(solver_data, state.solution, state.run_config));
        std::printf("write diagnostics: %s\n", fname.data());
        return mara::complete_task_in(state, "write_diagnostics");
    };


    //=========================================================================
    auto record_time_series = [&solver_data] (state_t state)
    {
        auto sample       =  time_series_sample_t();
        sample.time       =  state.solution.time;
        sample.max_div_b  =  compute_div_b(solver_data, state.solution, state.run_config.get_int("ct_flag")) | nd::max();

        state.time_series = state.time_series.prepend(sample);
        return mara::complete_task_in(state, "record_time_series");
    };


    return mara::run_scheduled_tasks(state, {
        {"write_diagnostics", write_diagnostics},
        {"record_time_series", record_time_series},
        {"write_checkpoint",  write_checkpoint}});
}


auto mhd_2d::simulation_should_continue( const state_t& state )
{
    return state.solution.time.value < state.run_config.get_double("tfinal");
}


void mhd_2d::prepare_filesystem(const mara::config_t& run_config)
{
    auto outdir = run_config.get_string("outdir");
    mara::filesystem::require_dir(outdir);
}


void mhd_2d::print_run_loop_message(const state_t& state, mara::perf_diagnostics_t perf)
{
    auto kzps = state.solution.conserved.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", 
                state.solution.iteration.as_integral(),
                state.solution.time.value,
                kzps);
}


// ============================================================================
//                               Main
// ============================================================================

class subprog_mhd2d_sand : public mara::sub_program_t
{
public:

    int main(int argc, const char* argv[]) override
    {
        auto run_config  = mhd_2d::create_run_config(argc, argv);
        auto solver_data = mhd_2d::create_solver_data(run_config);
        auto state       = mhd_2d::create_state(run_config);
        auto next        = std::bind(mhd_2d::next_state, std::placeholders::_1, solver_data);
        auto tasks       = std::bind(mhd_2d::run_tasks , std::placeholders::_1, solver_data);
        auto perf        = mara::perf_diagnostics_t();

        mhd_2d::prepare_filesystem(run_config);
        mara::pretty_print(std::cout, "config", run_config);
        state = tasks(state);

        while (mhd_2d::simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(mara::compose(tasks, next), state);
            mhd_2d::print_run_loop_message(state, perf);
        }

        tasks(next(state));
        return 0;
    }

    std::string name() const override
    {
        return "mhd_2d";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_mhd2d_sand()
{
    return std::make_unique<subprog_mhd2d_sand>();
}

#endif //MARA_COMPILE_SUBPROGRAM_MHD2D_SAND
