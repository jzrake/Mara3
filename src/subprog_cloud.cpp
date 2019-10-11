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
#if MARA_COMPILE_SUBPROGRAM_CLOUD




#include <iostream>
#include "core_hdf5.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "core_geometric.hpp"
#include "core_rational.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_parallel.hpp"
#include "app_subprogram.hpp"
#include "physics_srhd.hpp"
#include "model_atmosphere.hpp"
#include "model_jet_nozzle.hpp"
#include "post_shock_locator.hpp"




#define gamma_law_index (4. / 3)
#define light_speed_cgs 2.998e10
#define solar_mass_cgs  1.989e33




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart",      std::string())   // name of a restart file (create new run if empty)
    .item("outdir",              "data")   // directory to put output files in
    .item("nr",                     256)   // number of radial zones, per decade
    .item("tfinal",                 1.0)   // time to stop the simulation (code time units: inner_radius / light_speed)
    .item("cpi",                   10.0)   // checkpoint interval (code time units)
    .item("tsi",                    0.1)   // time-series interval (code time units)
    .item("dfi",                    1.0)   // diagnostic field interval (primitives and other data products, code time units)
    .item("num_decades",            2.0)   // number of radial decades to include in the domain
    .item("inner_radius",          3e08)   // inner boundary radius (in cm)
    .item("cloud_cutoff",          3e10)   // cloud radius rc (cm) where the density index changes from n to n2
    .item("cloud_mass",            2e-2)   // cloud mass (in solar masses)
    .item("density_index",          2.0)   // index n of the cloud density profile, rho ~ r^(-n) where r < rc
    .item("density_index2",         6.0)   // index n2 of the density beyond rc
    .item("jet_delay_time",         1.0)   // time for which the envelop propagates before engine oneset
    .item("jet_total_energy",      1e50)   // total energy (solid-angle and time-integrated) to be injected (erg)
    .item("jet_duration",           1.0)   // engine duration (in seconds)
    .item("jet_gamma_beta",        10.0)   // jet gamma-beta on-axis
    .item("jet_opening_angle",      0.1)   // jet opening-angle, theta_j
    .item("jet_structure_exp",      2.0)   // jet structure exponent alpha: exp[-(theta/theta_j)^alpha]
    .item("cfl_number",             0.4)   // CLF parameter
    .item("rk_order",                 1)   // RK time stepping mode (1 or 2)
    .item("reconstruct_method",       2)   // spatial reconstruction method: 1 for PCM, 2 for PLM
    .item("plm_theta",              1.2)   // PLM theta parameter
    .item("temperature_floor",     1e-8);  // temperature floor (0.0 means no floor is applied)
}




//=============================================================================
struct CloudProblem
{


    //=========================================================================
    using radial_vertex_array_t = nd::shared_array<mara::unit_length<double>, 1>;
    using polar_vertex_array_t  = nd::shared_array<double, 1>;
    using primitive_array_1d_t  = nd::shared_array<mara::srhd::primitive_t, 1>;


    //=========================================================================
    struct solution_t
    {
        double                                       time = 0.0;
        mara::rational_number_t                      iteration = mara::make_rational(0, 1);
        radial_vertex_array_t                        radial_vertices;
        polar_vertex_array_t                         polar_vertices;
        nd::shared_array<mara::srhd::conserved_t, 2> conserved;


        //=============================================================================
        solution_t operator+(const solution_t& other) const
        {
            return {
                time       + other.time,
                iteration  + other.iteration,
                radial_vertices,
                polar_vertices,
                (conserved + other.conserved).shared(),
            };
        }

        solution_t operator*(mara::rational_number_t scale) const
        {
            return {
                time       * scale.as_double(),
                iteration  * scale,
                radial_vertices,
                polar_vertices,
                (conserved * scale.as_double()).shared(),
            };
        }
    };


    //=========================================================================
    struct diagnostic_fields_t
    {
        double time = 0.0;
        nd::shared_array<double, 2> mass_density;
        nd::shared_array<double, 2> gas_pressure;
        nd::shared_array<double, 2> specific_entropy;
        nd::shared_array<double, 2> radial_gamma_beta;
        nd::shared_array<double, 2> radial_energy_flow;
        nd::shared_array<double, 1> total_energy_at_theta;
        nd::shared_array<double, 1> solid_angle_at_theta;
        nd::shared_array<double, 1> shock_midpoint_radius;
        nd::shared_array<double, 1> shock_upstream_radius;
        nd::shared_array<double, 1> shock_pressure_radius;
        nd::shared_array<double, 1> shock_luminosity_radius;
        nd::shared_array<double, 1> postshock_flow_gamma;
        nd::shared_array<double, 1> postshock_flow_power;
        nd::shared_array<double, 1> postshock_flow_power02;
        nd::shared_array<double, 1> postshock_flow_power04;
        nd::shared_array<double, 1> postshock_flow_power08;
        nd::shared_array<double, 1> postshock_flow_power16;
        nd::shared_array<double, 1> postshock_flow_power32;
        nd::shared_array<double, 1> postshock_flow_power64;
        nd::shared_array<double, 1> postshock_flow_power_max;
        radial_vertex_array_t       radial_vertices;
        polar_vertex_array_t        polar_vertices;
    };


    //=========================================================================
    struct app_state_t
    {
        solution_t solution;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };


    //=========================================================================
    struct unit_system_t
    {
        unit_system_t with_length(double new_length) const { return {new_length, _mass, _time}; }
        unit_system_t with_mass  (double new_mass)   const { return {_length, new_mass, _time}; }
        unit_system_t with_time  (double new_time)   const { return {_length, _mass, new_time}; }

        double length()         const { return _length; }
        double mass()           const { return _mass; }
        double time()           const { return _time; }
        double velocity()       const { return light_speed_cgs; }
        double energy()         const { return mass() * std::pow(velocity(), 2); }
        double mass_density()   const { return mass() / std::pow(length(), 3); }
        double energy_density() const { return energy() / std::pow(length(), 3); }
        double power()          const { return energy() / time(); }

        double _length  = 1.0; /** cm       */
        double _mass    = 1.0; /** g        */
        double _time    = 1.0; /** s        */
    };


    //=========================================================================
    static auto make_cloud_envelop_model(const mara::config_t& cfg);
    static auto make_atmosphere_model(const mara::config_t& cfg);
    static auto make_jet_nozzle_model(const mara::config_t& cfg);
    static auto make_reference_units(const mara::config_t& cfg);
    static auto make_diagnostic_fields(const solution_t& state, const mara::config_t& cfg);


    //=========================================================================
    static auto radial_face_areas(radial_vertex_array_t, polar_vertex_array_t);
    static auto polar_face_areas (radial_vertex_array_t, polar_vertex_array_t);
    static auto cell_volumes     (radial_vertex_array_t, polar_vertex_array_t);
    static auto cell_centroids   (radial_vertex_array_t, polar_vertex_array_t);


    //=========================================================================
    static auto intercell_flux(std::size_t axis);
    static auto estimate_gradient_plm(double plm_theta);
    static auto extend_zero_gradient_inner();
    static auto extend_zero_gradient_outer();
    static auto extend_inflow_nozzle_inner(const app_state_t& app_state);
    static auto advance(const app_state_t& app_state, const solution_t& solution, mara::unit_time<double> dt);


    //=============================================================================
    static void write_solution(h5::Group&& group, const solution_t& state);
    static auto read_solution(h5::Group&& group);
    static auto new_solution(const mara::config_t& cfg);
    static auto create_solution(const mara::config_t& cfg);
    static auto next_solution(const app_state_t& app_state);


    //=============================================================================
    static auto new_schedule(const mara::config_t& cfg);
    static auto create_schedule(const mara::config_t& cfg);
    static auto next_schedule(const mara::schedule_t& schedule, const mara::config_t& cfg, double time);


    //=========================================================================
    static void write_checkpoint(const app_state_t& state, std::string outdir);
    static void write_diagnostics(const app_state_t& state, std::string outdir);
    static void write_time_series(const app_state_t& state, std::string outdir);
    static auto create_app_state(mara::config_t cfg);
    static auto next(const app_state_t& state);
    static auto simulation_should_continue(const app_state_t& state);
    static auto run_tasks(const app_state_t& state);


    //=========================================================================
    static void print_run_loop_message(const solution_t& solution, mara::perf_diagnostics_t perf);
    static void print_run_dimensions(std::ostream& output, const mara::config_t& cfg);
    static void prepare_filesystem(const mara::config_t& cfg);


    template<typename ArrayType> static auto sin(ArrayType array) { return array | nd::map([] (auto x) { return std::sin(x); }); }
    template<typename ArrayType> static auto cos(ArrayType array) { return array | nd::map([] (auto x) { return std::cos(x); }); }
};




//=============================================================================
auto CloudProblem::radial_face_areas(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    auto [r, q] = nd::meshgrid(r_vertices, q_vertices);
    auto rc = r | nd::midpoint_on_axis(1);
    auto dm = -cos(q) | nd::difference_on_axis(1);
    return rc * rc * dm * 2 * M_PI;
}

auto CloudProblem::polar_face_areas(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    auto [r, q] = nd::meshgrid(r_vertices, q_vertices);
    auto dr = r | nd::difference_on_axis(0);
    auto rc = r | nd::midpoint_on_axis(0);
    auto qc = q | nd::midpoint_on_axis(0);
    return rc * dr * sin(qc) * 2 * M_PI;
}

auto CloudProblem::cell_volumes(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    auto [r, q] = nd::meshgrid(r_vertices, q_vertices);
    auto dv = r | nd::map([] (auto r) { return r * r * r; }) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
    auto dm = -cos(q) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
    return dv * dm * 2 * M_PI / 3.0;
}

auto CloudProblem::cell_centroids(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    return nd::cartesian_product(
        r_vertices | nd::midpoint_on_axis(0),
        q_vertices | nd::midpoint_on_axis(0));
}




//=============================================================================
auto CloudProblem::make_cloud_envelop_model(const mara::config_t& cfg)
{
    return mara::cloud_and_envelop_model()
    .with_inner_radius  (cfg.get_double("inner_radius"))
    .with_cloud_index   (cfg.get_double("density_index"));
}

auto CloudProblem::make_atmosphere_model(const mara::config_t& cfg)
{
    return mara::power_law_atmosphere_model()
    .with_inner_radius  (cfg.get_double("inner_radius"))
    .with_cutoff_radius (cfg.get_double("cloud_cutoff"))
    .with_inner_index   (cfg.get_double("density_index"))
    .with_outer_index   (cfg.get_double("density_index2"))
    .with_total_mass    (cfg.get_double("cloud_mass") * solar_mass_cgs);
}

auto CloudProblem::make_jet_nozzle_model(const mara::config_t& cfg)
{
    return mara::jet_nozzle_model()
    .with_inner_radius      (cfg.get_double("inner_radius"))
    .with_total_energy      (cfg.get_double("jet_total_energy"))
    .with_jet_duration      (cfg.get_double("jet_duration"))
    .with_structure_exponent(cfg.get_double("jet_structure_exp"))
    .with_opening_angle     (cfg.get_double("jet_opening_angle"))
    .with_lorentz_factor    (cfg.get_double("jet_gamma_beta"));
}

auto CloudProblem::make_reference_units(const mara::config_t& cfg)
{
    auto atmosphere_model = make_atmosphere_model(cfg);

    return unit_system_t()
    .with_length(atmosphere_model.r0)
    .with_mass(atmosphere_model.total_mass())
    .with_time(atmosphere_model.r0 / light_speed_cgs);
}

auto CloudProblem::make_diagnostic_fields(const solution_t& state, const mara::config_t& cfg)
{
    using namespace std::placeholders;

    auto reference    = make_reference_units(cfg);
    auto dv           = cell_volumes     (state.radial_vertices, state.polar_vertices);
    auto dAr          = radial_face_areas(state.radial_vertices, state.polar_vertices);
    auto rhat         = mara::unit_vector_t::on_axis_1();
    auto cons_to_prim = std::bind(mara::srhd::recover_primitive, _1, gamma_law_index, cfg.get_double("temperature_floor"));
    auto radial_cells = state.radial_vertices | nd::midpoint_on_axis(0);
    auto primitive    = state.conserved | nd::divide(dv) | nd::map(cons_to_prim);

    auto solid_angle_at_theta     = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto total_energy_at_theta    = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_midpoint_radius    = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_upstream_radius    = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_pressure_radius    = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_luminosity_radius  = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_gamma     = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power     = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power02   = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power04   = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power08   = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power16   = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power32   = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power64   = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power_max = nd::make_unique_array<double>(state.polar_vertices.size() - 1);

    for (std::size_t j = 0; j < state.polar_vertices.size() - 1; ++j)
    {
        auto pj = primitive       | nd::freeze_axis(1).at_index(j) | nd::to_shared();
        auto uj = state.conserved | nd::freeze_axis(1).at_index(j);
        auto Aj = dAr | nd::freeze_axis(1).at_index(j) | nd::midpoint_on_axis(0);
        auto Lj = pj | nd::map([rhat] (auto p) { return p.flux(rhat, gamma_law_index)[4]; })
        | nd::multiply(Aj)
        | nd::multiply(reference.power())
        | nd::map([] (auto L) { return L.value; });

        auto midpoint_index   = mara::find_shock_index(pj, gamma_law_index)[0];
        auto upstream_index   = mara::find_index_of_pressure_plateau_ahead(pj, midpoint_index);
        auto pressure_index   = mara::find_index_of_maximum_pressure_behind(pj, midpoint_index);
        auto luminosity_index = mara::find_index_of_maximum_behind(Lj, midpoint_index);
        auto i02 = midpoint_index >  2 ? midpoint_index -  2 : 0;
        auto i04 = midpoint_index >  4 ? midpoint_index -  4 : 0;
        auto i08 = midpoint_index >  8 ? midpoint_index -  8 : 0;
        auto i16 = midpoint_index > 16 ? midpoint_index - 16 : 0;
        auto i32 = midpoint_index > 32 ? midpoint_index - 32 : 0;
        auto i64 = midpoint_index > 64 ? midpoint_index - 64 : 0;

        solid_angle_at_theta(j)     = dAr(0, j) / state.radial_vertices(0) / state.radial_vertices(0);
        total_energy_at_theta(j)    = uj | nd::map([] (auto u) { return u[4].value; }) | nd::multiply(reference.energy()) | nd::sum();
        shock_midpoint_radius(j)    = radial_cells(midpoint_index).value * reference.length();
        shock_upstream_radius(j)    = radial_cells(upstream_index).value * reference.length();
        shock_pressure_radius(j)    = radial_cells(pressure_index).value * reference.length();
        shock_luminosity_radius(j)  = radial_cells(luminosity_index).value * reference.length();
        postshock_flow_gamma(j)     = primitive(pressure_index, j).lorentz_factor();
        postshock_flow_power(j)     = Lj(pressure_index);
        postshock_flow_power02(j)   = Lj(i02);
        postshock_flow_power04(j)   = Lj(i04);
        postshock_flow_power08(j)   = Lj(i08);
        postshock_flow_power16(j)   = Lj(i16);
        postshock_flow_power32(j)   = Lj(i32);
        postshock_flow_power64(j)   = Lj(i64);
        postshock_flow_power_max(j) = Lj(luminosity_index);
    }

    auto result = diagnostic_fields_t();

    result.radial_vertices    = state.radial_vertices * reference.length() | nd::to_shared();
    result.polar_vertices     = state.polar_vertices;
    result.solid_angle_at_theta     = std::move(solid_angle_at_theta)    .shared();
    result.total_energy_at_theta    = std::move(total_energy_at_theta)   .shared();
    result.shock_midpoint_radius    = std::move(shock_midpoint_radius)   .shared();
    result.shock_upstream_radius    = std::move(shock_upstream_radius)   .shared();
    result.shock_pressure_radius    = std::move(shock_pressure_radius)   .shared();
    result.shock_luminosity_radius  = std::move(shock_luminosity_radius) .shared();
    result.postshock_flow_gamma     = std::move(postshock_flow_gamma)    .shared();
    result.postshock_flow_power     = std::move(postshock_flow_power)    .shared();
    result.postshock_flow_power02   = std::move(postshock_flow_power02)  .shared();
    result.postshock_flow_power04   = std::move(postshock_flow_power04)  .shared();
    result.postshock_flow_power08   = std::move(postshock_flow_power08)  .shared();
    result.postshock_flow_power16   = std::move(postshock_flow_power16)  .shared();
    result.postshock_flow_power32   = std::move(postshock_flow_power32)  .shared();
    result.postshock_flow_power64   = std::move(postshock_flow_power64)  .shared();
    result.postshock_flow_power_max = std::move(postshock_flow_power_max).shared();

    result.time               = state.time * reference.time();
    result.specific_entropy   = primitive | nd::map(std::bind(&mara::srhd::primitive_t::specific_entropy, _1, gamma_law_index)) | nd::to_shared();
    result.gas_pressure       = primitive | nd::map(std::mem_fn(&mara::srhd::primitive_t::gas_pressure)) | nd::multiply(reference.energy_density()) | nd::to_shared();
    result.mass_density       = primitive | nd::map(std::mem_fn(&mara::srhd::primitive_t::mass_density)) | nd::multiply(reference.mass_density())   | nd::to_shared();
    result.radial_gamma_beta  = primitive | nd::map(std::mem_fn(&mara::srhd::primitive_t::gamma_beta_1)) | nd::to_shared();
    result.radial_energy_flow = primitive
    | nd::map([rhat] (auto p) { return p.flux(rhat, gamma_law_index); })
    | nd::multiply(dAr | nd::select_axis(0).from(0).to(1).from_the_end())
    | nd::map([] (auto L) { return L[4].value; })
    | nd::multiply(reference.power())
    | nd::to_shared();

    return result;
}




//=============================================================================
auto CloudProblem::intercell_flux(std::size_t axis)
{
    return [axis] (auto left_and_right_states)
    {
        using namespace std::placeholders;
        auto nh = mara::unit_vector_t::on_axis(axis);
        auto riemann = std::bind(mara::srhd::riemann_hlle, _1, _2, nh, gamma_law_index);
        return left_and_right_states | nd::apply(riemann);
    };
}

auto CloudProblem::estimate_gradient_plm(double plm_theta)
{
    return [plm_theta] (double ul, double u0, double ur)
    {
        using std::min;
        using std::fabs;
        auto min3abs = [] (auto a, auto b, auto c) { return min(min(fabs(a), fabs(b)), fabs(c)); };
        auto sgn = [] (auto x) { return std::copysign(1, x); };

        auto a = plm_theta * (u0 - ul);
        auto b =       0.5 * (ur - ul);
        auto c = plm_theta * (ur - u0);
        return 0.25 * fabs(sgn(a) + sgn(b)) * (sgn(a) + sgn(c)) * min3abs(a, b, c);
    };
}

auto CloudProblem::extend_inflow_nozzle_inner(const app_state_t& app_state)
{
    auto jet         = make_jet_nozzle_model(app_state.run_config);
    auto reference   = make_reference_units(app_state.run_config);
    // auto atmosphere  = make_cloud_envelop_model(app_state.run_config);
    // auto cloud_u     = atmosphere.gamma_beta_at(atmosphere.inner_radius, app_state.run_config.get_double("jet_delay_time"));
    auto polar_cells = app_state.solution.polar_vertices | nd::midpoint_on_axis(0);
    auto t_seconds   = app_state.solution.time * reference.time();

    auto inflow_function = [jet, t=t_seconds, /*cloud_u,*/ reference_density=reference.mass_density()] (double q)
    {
        auto u = jet.gamma_beta(q, t) + jet.gamma_beta(M_PI - q, t);// + cloud_u;
        auto d = jet.density_at_base() / reference_density;

        return mara::srhd::primitive_t()
                .with_mass_density(d)
                .with_gamma_beta_1(u);
    };

    return [inflow_function, polar_cells] (auto array)
    {
        return polar_cells
        | nd::map(inflow_function)
        | nd::to_shared()
        | nd::reshape(1, polar_cells.size())
        | nd::concat(array);
    };
}

auto CloudProblem::extend_zero_gradient_inner()
{
    return [] (auto array)
    {
        return (array | nd::select_first(1, 0)) | nd::concat(array);
    };
}

auto CloudProblem::extend_zero_gradient_outer()
{
    return [] (auto array)
    {
        return array | nd::concat(array | nd::select_final(1, 0));
    };
}

auto CloudProblem::advance(const app_state_t& app_state, const solution_t& solution, mara::unit_time<double> dt)
{
    using namespace std::placeholders;

    auto source_terms = [] (auto primitive, auto position)
    {
        auto r = std::get<0>(position).value;
        auto q = std::get<1>(position);
        return primitive.spherical_geometry_source_terms(r, q, gamma_law_index);
    };

    auto temp_floor   = app_state.run_config.get_double("temperature_floor");
    auto cons_to_prim = std::bind(mara::srhd::recover_primitive, std::placeholders::_1, gamma_law_index, temp_floor);
    auto extend_bc    = mara::compose(extend_inflow_nozzle_inner(app_state), extend_zero_gradient_outer());
    auto evaluate     = mara::evaluate_on<MARA_PREFERRED_THREAD_COUNT>();

    auto rc  = cell_centroids   (solution.radial_vertices, solution.polar_vertices) | evaluate;
    auto dv  = cell_volumes     (solution.radial_vertices, solution.polar_vertices) | evaluate;
    auto dAr = radial_face_areas(solution.radial_vertices, solution.polar_vertices) | evaluate;
    auto dAq = polar_face_areas (solution.radial_vertices, solution.polar_vertices) | evaluate;

    auto u0 = solution.conserved;
    auto p0 = u0 / dv | nd::map(cons_to_prim) | evaluate;
    auto s0 = nd::zip(p0, rc) | nd::apply(source_terms) | nd::multiply(dv);

    if (app_state.run_config.get_int("reconstruct_method") == 1)
    {
        auto extrapolate = nd::zip_adjacent2_on_axis;

        auto lr = p0 | extend_bc | extrapolate(0) | intercell_flux(0)                       | nd::multiply(-dAr) | nd::difference_on_axis(0);
        auto lq = p0 |             extrapolate(1) | intercell_flux(1) | nd::extend_zeros(1) | nd::multiply(-dAq) | nd::difference_on_axis(1);
        auto u1 = u0 + (lr + lq + s0) * dt;

        return solution_t {
            solution.time + dt.value,
            solution.iteration + 1,
            solution.radial_vertices,
            solution.polar_vertices,
            u1 | evaluate };
    }

    if (app_state.run_config.get_int("reconstruct_method") == 2)
    {
        auto extrapolate = [plm_theta=app_state.run_config.get_double("plm_theta")] (std::size_t axis)
        {
            return [axis, plm_theta] (auto P)
            {
                auto L = nd::select_axis(axis).from(0).to(1).from_the_end();
                auto R = nd::select_axis(axis).from(1).to(0).from_the_end();
                auto G = P
                | nd::zip_adjacent3_on_axis(axis)
                | nd::apply(mara::lift(estimate_gradient_plm(plm_theta)))
                | nd::extend_zeros(axis)
                | nd::to_shared();

                return nd::zip(
                    (P | L) + (G | L) * 0.5,
                    (P | R) - (G | R) * 0.5);
            };
        };

        auto lr = p0 | extend_bc | extrapolate(0) | intercell_flux(0)                       | nd::multiply(-dAr) | nd::difference_on_axis(0);
        auto lq = p0 |             extrapolate(1) | intercell_flux(1) | nd::extend_zeros(1) | nd::multiply(-dAq) | nd::difference_on_axis(1);
        auto u1 = u0 + (lr + lq + s0) * dt;

        return solution_t {
            solution.time + dt.value,
            solution.iteration + 1,
            solution.radial_vertices,
            solution.polar_vertices,
            u1 | evaluate };
    }
    throw std::invalid_argument("reconstruct_method must be 1 or 2");
}




//=============================================================================
void CloudProblem::write_solution(h5::Group&& group, const solution_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("radial_vertices", state.radial_vertices);
    group.write("polar_vertices", state.polar_vertices);
    group.write("conserved", state.conserved);
}

auto CloudProblem::read_solution(h5::Group&& group)
{
    auto state = solution_t();
    group.read("time", state.time);
    group.read("iteration", state.iteration);
    group.read("radial_vertices", state.radial_vertices);
    group.read("polar_vertices", state.polar_vertices);
    group.read("conserved", state.conserved);
    return state;
}

auto CloudProblem::new_solution(const mara::config_t& cfg)
{
    using namespace std::placeholders;

    // auto initial_p_v1 = [
    //     atmosphere_model = make_atmosphere_model(cfg),
    //     reference        = make_reference_units(cfg)]
    // (auto r, auto q)
    // {
    //     auto temperature = 1e-6;
    //     auto density = atmosphere_model.density_at(r.value * reference.length()) / reference.mass_density();

    //     return mara::srhd::primitive_t()
    //     .with_mass_density(density)
    //     .with_gas_pressure(density * temperature);
    // };

    auto initial_p_v2 = [
        atmosphere_model  = make_cloud_envelop_model(cfg),
        reference         = make_reference_units(cfg),
        jet_delay_time    = cfg.get_double("jet_delay_time")]
    (auto r, auto q)
    {
        auto r_cm        = r.value * reference.length();
        auto temperature = 1e-6;
        auto density     = atmosphere_model.density_at   (r_cm, jet_delay_time) / reference.mass_density();
        auto gamma_beta  = atmosphere_model.gamma_beta_at(r_cm, jet_delay_time);

        return mara::srhd::primitive_t()
        .with_mass_density(density)
        .with_gas_pressure(density * temperature)
        .with_gamma_beta_1(gamma_beta);
    };

    auto to_conserved   = std::bind(&mara::srhd::primitive_t::to_conserved_density, _1, gamma_law_index);
    auto nr             = cfg.get_int("nr");
    auto num_decades    = cfg.get_double("num_decades");
    auto r_vertices = nd::linspace(0.0, num_decades, int(num_decades * nr) + 1)
    | nd::map([] (auto y) { return mara::make_length(std::pow(10.0, y)); })
    | nd::to_shared();

    auto q_vertices = nd::linspace(0.0, M_PI, nr + 1) | nd::to_shared();
    auto dv = cell_volumes(r_vertices, q_vertices);
    auto state = solution_t();

    state.time = 0.0;
    state.iteration = 0;
    state.radial_vertices = r_vertices;
    state.polar_vertices = q_vertices;
    state.conserved = cell_centroids(r_vertices, q_vertices)
    | nd::apply(initial_p_v2)
    | nd::map(to_conserved)
    | nd::multiply(dv)
    | nd::to_shared();

    return state;
}

auto CloudProblem::create_solution(const mara::config_t& cfg)
{
    auto restart = cfg.get_string("restart");
    return restart.empty()
    ? new_solution(cfg)
    : read_solution(h5::File(restart, "r").open_group("solution"));
}

auto CloudProblem::next_solution(const app_state_t& app_state)
{
    auto dr_min = app_state.solution.radial_vertices | nd::difference_on_axis(0) | nd::read_index(0);
    auto dt = dr_min / mara::make_velocity(1.0) * app_state.run_config.get_double("cfl_number");
    auto s0 = app_state.solution;

    switch (app_state.run_config.get_int("rk_order"))
    {
        case 1:
        {
            return advance(app_state, s0, dt);
        }
        case 2:
        {
            auto b0 = mara::make_rational(1, 2);
            auto s1 = advance(app_state, s0, dt);
            auto s2 = advance(app_state, s1, dt);
            return s0 * b0 + s2 * (1 - b0);
        }
    }
    throw std::invalid_argument("cloud::next_solution (invalid rk_order)");    
}




//=============================================================================
auto CloudProblem::new_schedule(const mara::config_t& cfg)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_checkpoint");
    schedule.create_and_mark_as_due("write_diagnostics");
    schedule.create_and_mark_as_due("write_time_series");
    return schedule;
}

auto CloudProblem::create_schedule(const mara::config_t& cfg)
{
    auto restart = cfg.get_string("restart");
    return restart.empty()
    ? new_schedule(cfg)
    : mara::read_schedule(h5::File(restart, "r").open_group("schedule"));
}

auto CloudProblem::next_schedule(const mara::schedule_t& schedule, const mara::config_t& cfg, double time)
{
    auto next_schedule = schedule;
    auto cpi = cfg.get_double("cpi");
    auto dfi = cfg.get_double("dfi");
    auto tsi = cfg.get_double("tsi");

    if (time - schedule.last_performed("write_checkpoint")  >= cpi) next_schedule.mark_as_due("write_checkpoint",  cpi);
    if (time - schedule.last_performed("write_diagnostics") >= dfi) next_schedule.mark_as_due("write_diagnostics", dfi);
    if (time - schedule.last_performed("write_time_series") >= tsi) next_schedule.mark_as_due("write_time_series", tsi);

    return next_schedule;
}




//=============================================================================
static auto new_run_config(const mara::config_string_map_t& args)
{
    return config_template().create().update(args);
}

static auto create_run_config(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);
    return args.count("restart")
    ? config_template()
            .create()
            .update(mara::read_config(h5::File(args.at("restart"), "r").open_group("config")))
            .update(args)
    : new_run_config(args);
}




//=============================================================================
void CloudProblem::write_checkpoint(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_checkpoint");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", count, "h5")), "w");
    write_solution(file.require_group("solution"), state.solution);
    mara::write_schedule(file.require_group("schedule"), state.schedule);
    mara::write_config(file.require_group("config"), state.run_config);

    std::printf("write checkpoint: %s\n", file.filename().data());
}

void CloudProblem::write_diagnostics(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_diagnostics");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5")), "w");
    auto diagnostics = make_diagnostic_fields(state.solution, state.run_config);

    file.write("time",                  diagnostics.time);
    file.write("gas_pressure",          diagnostics.gas_pressure);
    file.write("mass_density",          diagnostics.mass_density);
    file.write("specific_entropy",      diagnostics.specific_entropy);
    file.write("radial_energy_flow",    diagnostics.radial_energy_flow);
    file.write("radial_gamma_beta",     diagnostics.radial_gamma_beta);
    file.write("radial_vertices",       diagnostics.radial_vertices);
    file.write("polar_vertices",        diagnostics.polar_vertices);
    file.write("solid_angle_at_theta",  diagnostics.solid_angle_at_theta);
    file.write("total_energy_at_theta", diagnostics.total_energy_at_theta);
    file.write("shock_midpoint_radius", diagnostics.shock_midpoint_radius);
    file.write("shock_upstream_radius", diagnostics.shock_upstream_radius);
    file.write("shock_pressure_radius", diagnostics.shock_pressure_radius);
    file.write("shock_luminosity_radius", diagnostics.shock_luminosity_radius);
    file.write("postshock_flow_gamma",  diagnostics.postshock_flow_gamma);
    file.write("postshock_flow_power",  diagnostics.postshock_flow_power);
    file.write("postshock_flow_power02",  diagnostics.postshock_flow_power02);
    file.write("postshock_flow_power04",  diagnostics.postshock_flow_power04);
    file.write("postshock_flow_power08",  diagnostics.postshock_flow_power08);
    file.write("postshock_flow_power16",  diagnostics.postshock_flow_power16);
    file.write("postshock_flow_power32",  diagnostics.postshock_flow_power32);
    file.write("postshock_flow_power64",  diagnostics.postshock_flow_power64);
    file.write("postshock_flow_power_max",  diagnostics.postshock_flow_power_max);
    std::printf("write diagnostics: %s\n", file.filename().data());
}

void CloudProblem::write_time_series(const app_state_t& state, std::string outdir)
{
}

auto CloudProblem::create_app_state(mara::config_t cfg)
{
    auto state = app_state_t();
    state.run_config     = cfg;
    state.solution       = create_solution(cfg);
    state.schedule       = create_schedule(cfg);
    return state;
}

auto CloudProblem::next(const app_state_t& state)
{
    auto next_state = state;
    next_state.solution = next_solution(state);
    next_state.schedule = next_schedule(state.schedule, state.run_config, state.solution.time);
    return next_state;
}

auto CloudProblem::simulation_should_continue(const app_state_t& state)
{
    auto time = state.solution.time;
    auto tfinal = state.run_config.get_double("tfinal");
    return time < tfinal;
}

auto CloudProblem::run_tasks(const app_state_t& state)
{
    auto next_state = state;
    auto outdir = state.run_config.get_string("outdir");

    if (state.schedule.is_due("write_checkpoint"))
    {
        write_checkpoint(state, outdir);
        next_state.schedule.mark_as_completed("write_checkpoint");
    }
    if (state.schedule.is_due("write_diagnostics"))
    {
        write_diagnostics(state, outdir);
        next_state.schedule.mark_as_completed("write_diagnostics");
    }
    if (state.schedule.is_due("write_time_series"))
    {
        write_time_series(state, outdir);
        next_state.schedule.mark_as_completed("write_time_series");
    }
    return next_state;
}




//=============================================================================
void CloudProblem::print_run_loop_message(const solution_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.radial_vertices.size() * solution.polar_vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration.as_integral(), solution.time, kzps);
    std::fflush(stdout);
}

void CloudProblem::print_run_dimensions(std::ostream& output, const mara::config_t& cfg)
{
    auto c2 = light_speed_cgs * light_speed_cgs;
    // auto atmosphere_model = make_atmosphere_model(cfg);
    auto atmosphere_model = make_cloud_envelop_model(cfg);
    auto jet_nozzle_model = make_jet_nozzle_model(cfg);
    auto jet_delay_time   = cfg.get_double("jet_delay_time");

    output << "====================================================\n";
    output << "model description:\n\n";
    output << "\treference length.................. " << atmosphere_model.inner_radius << " cm" << std::endl;
    output << "\treference time.................... " << atmosphere_model.inner_radius / light_speed_cgs << " s" << std::endl;
    output << "\treference mass.................... " << atmosphere_model.total_mass(jet_delay_time) << " g" << std::endl;
    output << "\treference density................. " << atmosphere_model.total_mass(jet_delay_time) / std::pow(atmosphere_model.inner_radius, 3) << " g/cm^3" << std::endl;
    output << "\treference energy.................. " << atmosphere_model.total_mass(jet_delay_time) * c2 << " erg" << std::endl;
    output << "\ttotal atmosphere mass............. " << atmosphere_model.total_mass(jet_delay_time) / solar_mass_cgs << " M_solar" << std::endl;
    output << "\tcloud cutoff radius............... " << atmosphere_model.cloud_outer_boundary(jet_delay_time) << " cm" << std::endl;
    output << "\tcloud velocity.................... " << atmosphere_model.velocity_at(atmosphere_model.inner_radius, jet_delay_time) << " cm/s" << std::endl;
    output << "\tcloud four velocity............... " << atmosphere_model.gamma_beta_at(atmosphere_model.inner_radius, jet_delay_time) << std::endl;
    output << "\tdensity at cloud base............. " << atmosphere_model.density_at(atmosphere_model.inner_radius, jet_delay_time) << " g/cm^3" << std::endl;
    output << "\tdensity at cloud cutoff........... " << atmosphere_model.density_at(atmosphere_model.cloud_outer_boundary(jet_delay_time), jet_delay_time) << " g/cm^3" << std::endl;
    output << "\tjet mass density at base.......... " << jet_nozzle_model.density_at_base() << " g/cm^3" << std::endl;
    output << "\tjet Lorentz factor at q=0, t=0s... " << jet_nozzle_model.gamma_beta(0, 0) << std::endl;
    output << "\tjet Lorentz factor at q=0, t=1s... " << jet_nozzle_model.gamma_beta(0, 1) << std::endl;
    output << "\texplosion E / M................... " << jet_nozzle_model.Ej / (atmosphere_model.total_mass(jet_delay_time) * c2) << std::endl;
    output << std::endl;
}

void CloudProblem::prepare_filesystem(const mara::config_t& cfg)
{
    if (cfg.get_string("restart").empty())
    {
        auto outdir = cfg.get_string("outdir");
        mara::filesystem::require_dir(outdir);

        auto file = h5::File(mara::filesystem::join(outdir, "time_series.h5"), "w");
        auto plist = h5::PropertyList::dataset_create().set_chunk(1000);
        auto space = h5::Dataspace::unlimited(0);

        file.require_dataset("time", h5::Datatype::native_double(), space, plist);
        file.require_dataset("shock_radius", h5::Datatype::native_double(), space, plist);
        mara::write_config(file.require_group("config"), cfg);
    }
}




//=============================================================================
class subprog_cloud : public mara::sub_program_t
{
public:

    int run_main(const mara::config_t& cfg)
    {
        using prob             = CloudProblem;
        auto run_tasks_on_next = mara::compose(prob::run_tasks, prob::next);
        auto perf              = mara::perf_diagnostics_t();
        auto state             = prob::create_app_state(cfg);

        mara::pretty_print(std::cout, "config", cfg);
        prob::prepare_filesystem(cfg);
        prob::print_run_dimensions(std::cout, cfg);

        state = prob::run_tasks(state);

        while (prob::simulation_should_continue(state))
        {
            std::tie(state, perf) = mara::time_execution(run_tasks_on_next, state);
            prob::print_run_loop_message(state.solution, perf);
        }

        run_tasks_on_next(state);
        return 0;
    }

    int main(int argc, const char* argv[]) override
    {
        return run_main(create_run_config(argc, argv));
    }

    std::string name() const override
    {
        return "cloud";
    }
};

std::unique_ptr<mara::sub_program_t> make_subprog_cloud()
{
    return std::make_unique<subprog_cloud>();
}

#endif // MARA_COMPILE_SUBPROGRAM_CLOUD