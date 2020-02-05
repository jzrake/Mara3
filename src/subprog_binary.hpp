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
#include "app_config.hpp"
#include "app_performance.hpp"
#include "app_schedule.hpp"
#include "app_serialize.hpp"
#include "core_hdf5.hpp"
#include "core_linked_list.hpp"
#include "core_ndarray.hpp"
#include "core_rational.hpp"
#include "core_tree.hpp"
#include "model_two_body.hpp"
#include "physics_iso2d.hpp"




//=============================================================================
namespace binary
{


    //=========================================================================
    using location_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0,  0, double>, 2>;
    using velocity_2d_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -1, double>, 2>;
    using accel_2d_t    = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 0, -2, double>, 2>;
    using force_2d_t    = mara::arithmetic_sequence_t<mara::dimensional_value_t<1, 1, -2, double>, 2>;
    using primitive_field_t = std::function<mara::iso2d::primitive_t(location_2d_t)>;

    template<typename ArrayValueType>
    using quad_tree_t = mara::arithmetic_binary_tree_t<nd::shared_array<ArrayValueType, 2>, 2>;


    //=========================================================================
    enum class reconstruct_method_t
    {
        plm,
        pcm,
    };


    //=========================================================================
    enum class riemann_solver_t
    {
        hlle,
        hllc,
    };


    //=========================================================================
    struct solver_data_t
    {
        mara::unit_rate  <double>                             sink_rate;
        mara::unit_time  <double>                             recommended_time_step;
        mara::unit_length<double>                             softening_radius;
        mara::unit_length<double>                             gst_suppr_radius;
        mara::unit_length<double>                             domain_radius;
        mara::unit_length<double>                             sink_radius;
        mara::dimensional_value_t<-2, 1, 0, double>           density_floor;
        double                                                mach_number;
        double                                                alpha;
        double                                                alpha_cutoff_radius;
        double                                                nu;
        double                                                plm_theta;
        double                                                cfl_number;
        double                                                begin_live_binary;
        int                                                   rk_order;
        bool                                                  axisymmetric_cs2;
        bool                                                  conserve_linear_p;
        bool                                                  fixed_dt;
        bool                                                  no_accretion_force;
        std::size_t                                           block_size;
        reconstruct_method_t                                  reconstruct_method;
        riemann_solver_t                                      riemann_solver;
        quad_tree_t<location_2d_t>                            vertices;
        quad_tree_t<location_2d_t>                            cell_centers;
        quad_tree_t<mara::unit_area<double>>                  cell_areas;
        quad_tree_t<mara::iso2d::conserved_per_area_t>        initial_conserved_u;
        quad_tree_t<mara::iso2d::conserved_angmom_per_area_t> initial_conserved_q;
        quad_tree_t<mara::unit_rate<double>>                  buffer_rate_field;
    };


    //=========================================================================
    struct solution_t
    {
        mara::unit_time<double>                                   time = 0.0;
        mara::rational_number_t                                   iteration = 0;
        quad_tree_t<mara::iso2d::conserved_per_area_t>            conserved_u;
        quad_tree_t<mara::iso2d::conserved_angmom_per_area_t>     conserved_q;

        mara::arithmetic_sequence_t<mara::unit_mass  <double>, 2> mass_accreted_on = {};
        mara::arithmetic_sequence_t<mara::unit_angmom<double>, 2> angular_momentum_accreted_on = {};
        mara::arithmetic_sequence_t<mara::unit_angmom<double>, 2> integrated_torque_on = {};
        mara::arithmetic_sequence_t<mara::unit_energy<double>, 2> work_done_on = {};
        mara::unit_mass  <double>                                 mass_ejected = {};
        mara::unit_angmom<double>                                 angular_momentum_ejected = {};
        mara::full_orbital_elements_t                             orbital_elements_acc;
        mara::full_orbital_elements_t                             orbital_elements_grav;
        mara::full_orbital_elements_t                             orbital_elements;
        solution_t operator+(const solution_t& other) const;
        solution_t operator*(mara::rational_number_t scale) const;
    };


    //=========================================================================
    struct diagnostic_fields_t
    {
        mara::config_t                                 run_config;
        mara::unit_time<double>                        time;
        quad_tree_t<location_2d_t>                     vertices;
        quad_tree_t<double>                            sigma;
        quad_tree_t<double>                            radial_velocity;
        quad_tree_t<double>                            phi_velocity;
        location_2d_t                                  position_of_mass1;
        location_2d_t                                  position_of_mass2;
    };


    //=========================================================================
    struct time_series_sample_t
    {
        mara::unit_time  <double>                                 time = 0.0;
        mara::unit_mass  <double>                                 disk_mass = 0.0;
        mara::unit_angmom<double>                                 disk_angular_momentum = 0.0;
        mara::unit_mass  <double>                                 mass_ejected = {};
        mara::unit_angmom<double>                                 angular_momentum_ejected = {};

        mara::arithmetic_sequence_t<mara::unit_mass  <double>, 2> mass_accreted_on = {};
        mara::arithmetic_sequence_t<mara::unit_angmom<double>, 2> angular_momentum_accreted_on = {};
        mara::arithmetic_sequence_t<mara::unit_angmom<double>, 2> integrated_torque_on = {};
        mara::arithmetic_sequence_t<mara::unit_energy<double>, 2> work_done_on = {};
        mara::full_orbital_elements_t                             orbital_elements_acc;
        mara::full_orbital_elements_t                             orbital_elements_grav;
        mara::full_orbital_elements_t                             orbital_elements;
        location_2d_t                                             position_of_mass1;
        location_2d_t                                             position_of_mass2;
    };


    //=========================================================================
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


    //=========================================================================
    mara::config_template_t      create_config_template();
    mara::config_t               create_run_config   (int argc, const char* argv[]);
    mara::schedule_t             create_schedule     (const mara::config_t& run_config);
    mara::orbital_elements_t     create_binary_params(const mara::config_t& run_config);
    quad_tree_t<location_2d_t>   create_vertices     (const mara::config_t& run_config);
    solution_t                   create_solution     (const mara::config_t& run_config);
    state_t                      create_state        (const mara::config_t& run_config);
    solver_data_t                create_solver_data  (const mara::config_t& run_config);
    primitive_field_t            create_disk_profile (const mara::config_t& run_config);

    diagnostic_fields_t          diagnostic_fields    (const solution_t& solution, const mara::config_t& run_config);
    solution_t                   advance_u            (const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode=false);
    solution_t                   advance_q            (const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode=false);
    solution_t                   advance              (const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode=false);
    mara::unit_mass  <double>    disk_mass            (const solution_t& solution, const solver_data_t& solver_data);
    mara::unit_angmom<double>    disk_angular_momentum(const solution_t& solution, const solver_data_t& solver_data);

    void set_scheme_globals    (const mara::config_t& run_config);
    void prepare_filesystem    (const mara::config_t& run_config);
    void set_scheme_globals    (const mara::config_t& run_config);
    void print_run_loop_message(const state_t& state, const solver_data_t& solver_data, mara::perf_diagnostics_t perf);

    quad_tree_t<mara::iso2d::primitive_t> recover_primitive(
        const solution_t& solution,
        const solver_data_t& solver_data);

    double maximum_timestep(
        const solution_t& solution,
        const solver_data_t& solver_data);
}




//=============================================================================
namespace mara
{
    template<> void write<binary::solution_t>         (h5::Group&, std::string, const binary::solution_t&);
    template<> void write<binary::state_t>            (h5::Group&, std::string, const binary::state_t&);
    template<> void write<binary::diagnostic_fields_t>(h5::Group&, std::string, const binary::diagnostic_fields_t&);
    template<> void read<binary::solution_t>          (h5::Group&, std::string, binary::solution_t&);
    template<> void read<binary::state_t>             (h5::Group&, std::string, binary::state_t&);
    template<> void read<binary::diagnostic_fields_t> (h5::Group&, std::string, binary::diagnostic_fields_t&);
}
