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
#include <thread>
#include "ndmpi.hpp"
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "ndarray_ops.hpp"
#include "core_geometric.hpp"
#include "core_rational.hpp"
#include "app_config.hpp"
#include "app_filesystem.hpp"
#include "app_serialize.hpp"
#include "app_schedule.hpp"
#include "app_performance.hpp"
#include "app_subprogram.hpp"
#include "physics_srhd.hpp"




#define gamma_law_index (4. / 3)
#define cfl_number 0.4
#define light_speed_cgs 3e10
#define solar_mass_cgs 2e33




//=============================================================================
static auto config_template()
{
    return mara::make_config_template()
    .item("restart",      std::string())   // name of a restart file (create new run if empty)
    .item("outdir",              "data")   // directory to put output files in
    .item("nr",                     256)   // number of radial zones, per decade
    .item("tfinal",                 1.0)   // time to stop the simulation
    .item("cpi",                   10.0)   // checkpoint interval
    .item("tsi",                    0.1)   // time-series interval
    .item("dfi",                    1.0)   // diagnostic field interval (primitives and other data products)
    .item("num_decades",            2.0)   // number of radial decades to include in the domain
    .item("inner_radius",          3e08)   // inner boundary radius (in cm)
    .item("cloud_cutoff",          3e10)   // cloud radius rc (cm) where the density index changes from n to n2
    .item("cloud_mass",            2e-2)   // cloud mass (in solar masses)
    .item("density_index",          2.0)   // index n of the cloud density profile, rho ~ r^(-n) where r < rc
    .item("density_index2",         6.0)   // index n2 of the density beyond rc
    .item("jet_total_energy",      1e50)   // total energy (solid-angle and time-integrated) to be injected (erg)
    .item("jet_duration",           1.0)   // engine duration (in seconds)
    .item("jet_gamma_beta",        10.0)   // jet gamma-beta on-axis
    .item("jet_opening_angle",      0.1)
    .item("jet_structure_exp",      2.0);
}




//==============================================================================
template<std::size_t NumThreads>
static auto evaluate_on()
{
    return [] (auto array)
    {
        using value_type = typename decltype(array)::value_type;
        auto provider = nd::make_unique_provider<value_type>(array.shape());
        auto evaluate_partial = [&] (auto accessor)
        {
            return [accessor, array, &provider]
            {
                for (auto index : accessor)
                {
                    provider(index) = array(index);
                }
            };
        };
        auto threads = nd::basic_sequence_t<std::thread, NumThreads>();
        auto regions = nd::partition_shape<NumThreads>(array.shape());

        for (auto [n, accessor] : nd::enumerate(regions))
            threads[n] = std::thread(evaluate_partial(accessor));

        for (auto& thread : threads)
            thread.join();

        return nd::make_array(std::move(provider).shared());
    };
}




//=============================================================================
template<typename HydroSystem>
struct CloudProblem
{


    //=========================================================================
    using radial_vertex_array_t = nd::shared_array<mara::unit_length<double>, 1>;
    using polar_vertex_array_t  = nd::shared_array<double, 1>;
    using primitive_array_1d_t  = nd::shared_array<mara::srhd::primitive_t, 1>;


    //=========================================================================
    struct solution_state_t
    {
        double time = 0.0;
        mara::rational_number_t iteration = mara::make_rational(0, 1);
        radial_vertex_array_t radial_vertices;
        polar_vertex_array_t polar_vertices;
        nd::shared_array<typename HydroSystem::conserved_t, 2> conserved;
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
        nd::shared_array<double, 1> postshock_flow_power;
        nd::shared_array<double, 1> postshock_flow_gamma;
        radial_vertex_array_t       radial_vertices;
        polar_vertex_array_t        polar_vertices;
    };


    //=========================================================================
    struct app_state_t
    {
        solution_state_t solution_state;
        mara::schedule_t schedule;
        mara::config_t run_config;
    };


    /**
     * @brief      Model of a density profile that is a broken power-law in the
     *             spherical radius:
     *             
     *                      | f0 * (r / r0)^(-n1)           if r < rc
     *             rho(r) = |
     *                      | rho(rc) * (r / rc)^(-n2)      otherwise
     */
    struct power_law_atmosphere_model
    {
        power_law_atmosphere_model with_coefficient  (double new_f0) const { return {new_f0, r0, rc, n1, n2}; }
        power_law_atmosphere_model with_inner_radius (double new_r0) const { return {f0, new_r0, rc, n1, n2}; }
        power_law_atmosphere_model with_cutoff_radius(double new_rc) const { return {f0, r0, new_rc, n1, n2}; }
        power_law_atmosphere_model with_inner_index  (double new_n1) const { return {f0, r0, rc, new_n1, n2}; }
        power_law_atmosphere_model with_outer_index  (double new_n2) const { return {f0, r0, rc, n1, new_n2}; }
        power_law_atmosphere_model with_total_mass(double new_total_mass) const
        {
            return with_coefficient(new_total_mass / total_mass());
        }

        double mass_within_cutoff() const
        {
            return n1 == 3.0
            ? 4 * M_PI * (density_at(rc) * std::pow(rc, 3) * std::log(rc / r0))
            : 4 * M_PI * (density_at(rc) * std::pow(rc, 3) - density_at(r0) * std::pow(r0, 3)) / (3 - n1);
        }

        double mass_beyond_cutoff() const
        {
            if (n2 <= 3.0)
            {
                throw std::invalid_argument("power_law_atmosphere: outer index (n2) must be greater than 3");
            }
            return 4 * M_PI * density_at(rc) * std::pow(rc, 3) / (n2 - 3);
        }

        double total_mass() const
        {
            return mass_within_cutoff() + mass_beyond_cutoff();
        }

        double density_at(double r) const
        {
            return r <= rc ? f0 * std::pow(r / r0, -n1) : density_at(rc) * std::pow(r / rc, -n2);
        }

        double f0 = 1.0; /** coefficient (g / cm^3) */
        double r0 = 1.0; /** inner radius (cm) */
        double rc = 1e2; /** cutoff radius (cm) where index switches from n1 to n2 */
        double n1 = 2.0; /** power-law index where r < rc */
        double n2 = 6.0; /** power-law index where r > rc */
    };


    /**
     * @brief      Model of an ultra-relativistic, cold jet inflow with a
     *             Gaussian-like angular structure,
     *
     *             L(q, t) = dj G0^2 r0^2 c^3 exp(-(q / qj)^as) exp(-t / tj)
     *
     *             where L(q, t) is the luminosity per steradian at polar angle
     *             q and time t. Here, dj is the comoving mass density at the
     *             jet base, determined from the total energy and base Lorentz
     *             factor G0.
     */
    struct jet_nozzle_model
    {
        jet_nozzle_model with_total_energy      (double new_Ej) const { return {new_Ej, G0, tj, qj, as, r0}; }
        jet_nozzle_model with_lorentz_factor    (double new_G0) const { return {Ej, new_G0, tj, qj, as, r0}; }
        jet_nozzle_model with_jet_duration      (double new_tj) const { return {Ej, G0, new_tj, qj, as, r0}; }
        jet_nozzle_model with_opening_angle     (double new_qj) const { return {Ej, G0, tj, new_qj, as, r0}; }
        jet_nozzle_model with_structure_exponent(double new_as) const { return {Ej, G0, tj, qj, new_as, r0}; }
        jet_nozzle_model with_inner_radius      (double new_r0) const { return {Ej, G0, tj, qj, as, new_r0}; }


        /**
         * @brief      Return the luminosity per steradian of each jet.
         *
         * @param[in]  q     The polar angle
         * @param[in]  t     The time
         *
         * @return     The luminosity (erg / s / Sr)
         */
        double luminosity_per_steradian(double q, double t) const
        {
            return density_at_base() *
            std::pow(G0, 2) *
            std::pow(r0, 2) *
            std::pow(light_speed_cgs, 3) *
            std::exp(-std::pow(q / qj, as)) * std::exp(-t / tj);
        }
    

        /**
         * @brief      Return the jet gamma-beta (also the Lorentz factor given
         *             the ultra-relativistic assumption) at the jet base at the
         *             given polar angle q and time t. If including the
         *             counter-jet as well, this should be called as
         *             lorentz_factor(q) + lorentz_factor(M_PI - q).
         *
         * @param[in]  q     The polar angle
         * @param[in]  t     The time
         *
         * @return     The Lorentz factor
         */
        double gamma_beta(double q, double t) const
        {
            return G0 *
            std::exp(-0.5 * std::pow(q / qj, as)) *
            std::exp(-0.5 * t / tj);
        }


        /**
         * @brief      Estimate the comoving mass density at the jet base (r0)
         *             necessary for the jet (plus counter-jet) to have the
         *             total energy.
         *
         * @return     The density (g / cm^3)
         *
         * @note       This is estimate is accurate when the jet is cold (h - 1
         *             << 1) and ultra-relativistic (G0 >> 1), and when the
         *             structure exponent (as) is 2. Expect errors at the ~10%
         *             level for different values of as.
         */
        double density_at_base() const
        {
            return Ej / (2 * M_PI * std::pow(G0 * r0 * qj, 2) * tj * std::pow(light_speed_cgs, 3));
        }

        double Ej = 1.0; /** total explosion energy (erg) */
        double G0 = 2.0; /** Lorentz factor on-axis and at t=0 */
        double tj = 1.0; /** engine duration (s) */
        double qj = 0.1; /** engine opening angle (radian) */
        double as = 2.0; /** structure exponent */
        double r0 = 1.0; /** inner radius */
    };


    //=========================================================================
    struct reference_dimensions_t
    {
        reference_dimensions_t with_length(double new_length) const { return {new_length, _mass, _time}; }
        reference_dimensions_t with_mass  (double new_mass)   const { return {_length, new_mass, _time}; }
        reference_dimensions_t with_time  (double new_time)   const { return {_length, _mass, new_time}; }

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
    static auto make_atmosphere_model(const mara::config_t& cfg);
    static auto make_jet_nozzle_model(const mara::config_t& cfg);
    static auto make_reference_dimensions(const mara::config_t& cfg);


    //=========================================================================
    static auto radial_face_areas(radial_vertex_array_t, polar_vertex_array_t);
    static auto polar_face_areas (radial_vertex_array_t, polar_vertex_array_t);
    static auto cell_volumes     (radial_vertex_array_t, polar_vertex_array_t);
    static auto cell_centroids   (radial_vertex_array_t, polar_vertex_array_t);


    //=========================================================================
    static auto intercell_flux(std::size_t axis);
    static auto extend_reflecting_inner();
    static auto extend_zero_gradient_outer();
    static auto extend_inflow_nozzle_inner(const app_state_t& app_state);


    //=============================================================================
    static auto find_shock_index(primitive_array_1d_t primitive);
    static auto find_index_of_maximum_pressure_behind(primitive_array_1d_t primitive, std::size_t index);
    static auto find_index_of_pressure_plateau_ahead(primitive_array_1d_t primitive, std::size_t index);
    static auto make_diagnostic_fields(const solution_state_t& state, const mara::config_t& cfg);


    //=============================================================================
    static void write_solution(h5::Group&& group, const solution_state_t& state);
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
    static void print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf);
    static void print_run_dimensions(std::ostream& output, const mara::config_t& cfg);
    static void prepare_filesystem(const mara::config_t& cfg);


    template<typename ArrayType> static auto sin(ArrayType array) { return array | nd::map([] (auto x) { return std::sin(x); }); }
    template<typename ArrayType> static auto cos(ArrayType array) { return array | nd::map([] (auto x) { return std::cos(x); }); }
};




//=============================================================================
template<typename HydroSystem>
auto CloudProblem<HydroSystem>::make_atmosphere_model(const mara::config_t& cfg)
{
    return power_law_atmosphere_model()
    .with_inner_radius  (cfg.get_double("inner_radius"))
    .with_cutoff_radius (cfg.get_double("cloud_cutoff"))
    .with_inner_index   (cfg.get_double("density_index"))
    .with_outer_index   (cfg.get_double("density_index2"))
    .with_total_mass    (cfg.get_double("cloud_mass") * solar_mass_cgs);
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::make_jet_nozzle_model(const mara::config_t& cfg)
{
    return jet_nozzle_model()
    .with_inner_radius      (cfg.get_double("inner_radius"))
    .with_total_energy      (cfg.get_double("jet_total_energy"))
    .with_jet_duration      (cfg.get_double("jet_duration"))
    .with_structure_exponent(cfg.get_double("jet_structure_exp"))
    .with_opening_angle     (cfg.get_double("jet_opening_angle"))
    .with_lorentz_factor    (cfg.get_double("jet_gamma_beta"));
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::make_reference_dimensions(const mara::config_t& cfg)
{
    auto atmosphere_model = make_atmosphere_model(cfg);

    return reference_dimensions_t()
    .with_length(atmosphere_model.r0)
    .with_mass(atmosphere_model.total_mass())
    .with_time(atmosphere_model.r0 / light_speed_cgs);
}




//=============================================================================
template<typename HydroSystem>
auto CloudProblem<HydroSystem>::radial_face_areas(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    auto [r, q] = nd::meshgrid(r_vertices, q_vertices);
    auto rc = r | nd::midpoint_on_axis(1);
    auto dm = -cos(q) | nd::difference_on_axis(1);
    return rc * rc * dm * 2 * M_PI;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::polar_face_areas(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    auto [r, q] = nd::meshgrid(r_vertices, q_vertices);
    auto dr = r | nd::difference_on_axis(0);
    auto rc = r | nd::midpoint_on_axis(0);
    auto qc = q | nd::midpoint_on_axis(0);
    return rc * dr * sin(qc);
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::cell_volumes(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    auto [r, q] = nd::meshgrid(r_vertices, q_vertices);
    auto dv = r | nd::map([] (auto r) { return r * r * r; }) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
    auto dm = -cos(q) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
    return dv * dm * 2 * M_PI / 3.0;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::cell_centroids(radial_vertex_array_t r_vertices, polar_vertex_array_t q_vertices)
{
    return nd::cartesian_product(
        r_vertices | nd::midpoint_on_axis(0),
        q_vertices | nd::midpoint_on_axis(0));
}




//=============================================================================
template<typename HydroSystem>
auto CloudProblem<HydroSystem>::intercell_flux(std::size_t axis)
{
    return [axis] (auto array)
    {
        using namespace std::placeholders;
        auto L = array | nd::select_axis(axis).from(0).to(1).from_the_end();
        auto R = array | nd::select_axis(axis).from(1).to(0).from_the_end();
        auto nh = mara::unit_vector_t::on_axis(axis);
        auto riemann = std::bind(HydroSystem::riemann_hlle, _1, _2, nh, gamma_law_index);
        return nd::zip_arrays(L, R) | nd::apply(riemann);
    };
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::extend_reflecting_inner()
{
    return [] (auto array)
    {
        auto xl = array
        | nd::select_first(1, 0)
        | nd::map([] (auto p) { return p.with_gamma_beta_1(-p.gamma_beta_1()); });
        return xl | nd::concat(array);
    };
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::extend_inflow_nozzle_inner(const app_state_t& app_state)
{
    auto jet         = make_jet_nozzle_model(app_state.run_config);
    auto reference   = make_reference_dimensions(app_state.run_config);
    auto polar_cells = app_state.solution_state.polar_vertices | nd::midpoint_on_axis(0);
    auto t_seconds   = app_state.solution_state.time * reference.time();

    auto inflow_function = [jet, t=t_seconds, reference_density=reference.mass_density()] (double q)
    {
        auto u = jet.gamma_beta(q, t) + jet.gamma_beta(M_PI - q, t);
        auto d = jet.density_at_base() / reference_density;

        return typename HydroSystem::primitive_t()
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

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::extend_zero_gradient_outer()
{
    return [] (auto array)
    {
        return array | nd::concat(array | nd::select_final(1, 0));
    };
}




//=============================================================================
template<typename HydroSystem>
auto CloudProblem<HydroSystem>::find_shock_index(primitive_array_1d_t primitive)
{
    using namespace std::placeholders;

    auto s0 = primitive | nd::map(std::bind(&HydroSystem::primitive_t::specific_entropy, _1, gamma_law_index));
    auto ds = s0 | nd::difference_on_axis(0);
    auto shock_index = nd::where(ds == nd::min(ds)) | nd::read_index(0);
    return shock_index;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::find_index_of_maximum_pressure_behind(primitive_array_1d_t primitive, std::size_t index)
{
    auto p = primitive
    | nd::map(std::mem_fn(&HydroSystem::primitive_t::gas_pressure))
    | nd::bounds_check();

    try {
        while (p(index - 1) > p(index))
        {
            --index;
        }
        return index;
    }
    catch (const std::exception& e)
    {
        // std::printf("find_index_of_maximum_pressure_behind: %s\n", e.what());
        return std::size_t(0);
    }
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::find_index_of_pressure_plateau_ahead(primitive_array_1d_t primitive, std::size_t index)
{
    auto dlogp = primitive
    | nd::map(std::mem_fn(&HydroSystem::primitive_t::gas_pressure))
    | nd::map([] (auto p) { return std::log(p); })
    | nd::difference_on_axis(0)
    | nd::bounds_check();

    try {
        while (dlogp(index - 1) < 0.5 * dlogp(index - 2))
        {
            ++index;
        }
        return index;
    }
    catch (const std::exception& e)
    {
        // std::printf("find_index_of_pressure_plateau_ahead: %s\n", e.what());
        return std::size_t(0);
    }
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::make_diagnostic_fields(const solution_state_t& state, const mara::config_t& cfg)
{
    using namespace std::placeholders;
    auto cons_to_prim = std::bind(HydroSystem::recover_primitive, _1, gamma_law_index);
    auto dv = cell_volumes(state.radial_vertices, state.polar_vertices);
    auto primitive = state.conserved | nd::divide(dv) | nd::map(cons_to_prim);
    auto result = diagnostic_fields_t();
    auto reference = make_reference_dimensions(cfg);

    auto solid_angle_at_theta  = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto total_energy_at_theta = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_midpoint_radius = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_upstream_radius = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto shock_pressure_radius = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_power  = nd::make_unique_array<double>(state.polar_vertices.size() - 1);
    auto postshock_flow_gamma  = nd::make_unique_array<double>(state.polar_vertices.size() - 1);

    auto dAr = radial_face_areas(state.radial_vertices, state.polar_vertices);
    auto radial_cells = state.radial_vertices | nd::midpoint_on_axis(0);
    auto rhat = mara::unit_vector_t::on_axis_1();

    for (std::size_t j = 0; j < state.polar_vertices.size() - 1; ++j)
    {
        auto pj = primitive       | nd::freeze_axis(1).at_index(j) | nd::to_shared();
        auto uj = state.conserved | nd::freeze_axis(1).at_index(j);
        auto midpoint_index = find_shock_index(pj)[0];
        auto upstream_index = find_index_of_pressure_plateau_ahead(pj, midpoint_index);
        auto pressure_index = find_index_of_maximum_pressure_behind(pj, midpoint_index);
        auto pshock = primitive(pressure_index, j);

        solid_angle_at_theta(j) = dAr(0, j) / state.radial_vertices(0) / state.radial_vertices(0);
        total_energy_at_theta(j) = uj | nd::map([] (auto u) { return u[4].value; }) | nd::multiply(reference.energy()) | nd::sum();
        shock_midpoint_radius(j) = radial_cells(midpoint_index).value * reference.length();
        shock_upstream_radius(j) = radial_cells(upstream_index).value * reference.length();
        shock_pressure_radius(j) = radial_cells(pressure_index).value * reference.length();

        postshock_flow_power(j) =(pshock.flux(rhat, gamma_law_index)[4] * dAr(pressure_index, j)).value * reference.power();
        postshock_flow_gamma(j) = pshock.lorentz_factor();
    }

    result.time               = state.time;
    result.specific_entropy   = primitive | nd::map(std::bind(&HydroSystem::primitive_t::specific_entropy, _1, gamma_law_index)) | nd::to_shared();
    result.gas_pressure       = primitive | nd::map(std::mem_fn(&HydroSystem::primitive_t::gas_pressure)) | nd::multiply(reference.energy_density()) | nd::to_shared();
    result.mass_density       = primitive | nd::map(std::mem_fn(&HydroSystem::primitive_t::mass_density)) | nd::multiply(reference.mass_density())   | nd::to_shared();
    result.radial_gamma_beta  = primitive | nd::map(std::mem_fn(&HydroSystem::primitive_t::gamma_beta_1)) | nd::to_shared();

    result.radial_energy_flow = primitive
    | nd::map([rhat] (auto p) { return p.flux(rhat, gamma_law_index); })
    | nd::multiply(dAr | nd::select_axis(0).from(0).to(1).from_the_end())
    | nd::map([] (auto L) { return L[4].value; })
    | nd::multiply(reference.power())
    | nd::to_shared();

    result.radial_vertices    = state.radial_vertices * reference.length() | nd::to_shared();
    result.polar_vertices     = state.polar_vertices;
    result.solid_angle_at_theta  = std::move(solid_angle_at_theta).shared();
    result.total_energy_at_theta = std::move(total_energy_at_theta).shared();
    result.shock_midpoint_radius = std::move(shock_midpoint_radius).shared();
    result.shock_upstream_radius = std::move(shock_upstream_radius).shared();
    result.shock_pressure_radius = std::move(shock_pressure_radius).shared();
    result.postshock_flow_power  = std::move(postshock_flow_power).shared();
    result.postshock_flow_gamma  = std::move(postshock_flow_gamma).shared();

    return result;
}




//=============================================================================
template<typename HydroSystem>
void CloudProblem<HydroSystem>::write_solution(h5::Group&& group, const solution_state_t& state)
{
    group.write("time", state.time);
    group.write("iteration", state.iteration);
    group.write("radial_vertices", state.radial_vertices);
    group.write("polar_vertices", state.polar_vertices);
    group.write("conserved", state.conserved);
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::read_solution(h5::Group&& group)
{
    auto state = solution_state_t();
    group.read("time", state.time);
    group.read("iteration", state.iteration);
    group.read("radial_vertices", state.radial_vertices);
    group.read("polar_vertices", state.polar_vertices);
    group.read("conserved", state.conserved);
    return state;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::new_solution(const mara::config_t& cfg)
{
    using namespace std::placeholders;

    auto initial_p = [
        atmosphere_model = make_atmosphere_model(cfg),
        reference        = make_reference_dimensions(cfg)] (auto r, auto q)
    {
        auto temperature = 1e-6;
        auto density = atmosphere_model.density_at(r.value * reference.length()) / reference.mass_density();

        return typename HydroSystem::primitive_t()
        .with_mass_density(density)
        .with_gas_pressure(density * temperature);
    };

    auto to_conserved   = std::bind(&HydroSystem::primitive_t::to_conserved_density, _1, gamma_law_index);
    auto nr             = cfg.get_int("nr");
    auto num_decades    = cfg.get_double("num_decades");

    auto r_vertices = nd::linspace(0.0, num_decades, int(num_decades * nr) + 1)
    | nd::map([] (auto y) { return mara::make_length(std::pow(10.0, y)); })
    | nd::to_shared();

    auto q_vertices = nd::linspace(0, M_PI, nr + 1) | nd::to_shared();
    auto dv = cell_volumes(r_vertices, q_vertices);
    auto state = solution_state_t();

    state.time = 0.0;
    state.iteration = 0;
    state.radial_vertices = r_vertices;
    state.polar_vertices = q_vertices;
    state.conserved = cell_centroids(r_vertices, q_vertices)
    | nd::apply(initial_p)
    | nd::map(to_conserved)
    | multiply(dv)
    | nd::to_shared();

    return state;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::create_solution(const mara::config_t& cfg)
{
    auto restart = cfg.get_string("restart");
    return restart.empty()
    ? new_solution(cfg)
    : read_solution(h5::File(restart, "r").open_group("solution"));
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::next_solution(const app_state_t& app_state)
{
    using namespace std::placeholders;

    auto source_terms = [] (auto primitive, auto position)
    {
        auto r = std::get<0>(position).value;
        auto q = std::get<1>(position);
        return primitive.spherical_geometry_source_terms(r, q, gamma_law_index);
    };
    auto cons_to_prim = std::bind(HydroSystem::recover_primitive, std::placeholders::_1, gamma_law_index);
    auto extend_bc = mara::compose(extend_inflow_nozzle_inner(app_state), extend_zero_gradient_outer());

    auto evaluate = evaluate_on<12>();
    auto state = app_state.solution_state;

    auto dr_min = state.radial_vertices | nd::difference_on_axis(0) | nd::read_index(0);
    auto dt  = dr_min / mara::make_velocity(1.0) * cfl_number;
    auto rc  = cell_centroids   (state.radial_vertices, state.polar_vertices) | evaluate;
    auto dv  = cell_volumes     (state.radial_vertices, state.polar_vertices) | evaluate;
    auto dAr = radial_face_areas(state.radial_vertices, state.polar_vertices) | evaluate;
    auto dAq = polar_face_areas (state.radial_vertices, state.polar_vertices) | evaluate;

    auto u0 = state.conserved;
    auto p0 = u0 / dv | nd::map(cons_to_prim) | evaluate;
    auto s0 = nd::zip_arrays(p0, rc) | nd::apply(source_terms) | multiply(dv);
    auto lr = p0 | extend_bc | intercell_flux(0) | multiply(-dAr) | nd::difference_on_axis(0);
    auto lq = p0 | intercell_flux(1) | nd::extend_zero_gradient(1) | multiply(-dAq) | nd::difference_on_axis(1);
    auto u1 = u0 + (lr + lq + s0) * dt;

    return solution_state_t {
        state.time + dt.value,
        state.iteration + 1,
        state.radial_vertices,
        state.polar_vertices,
        u1 | evaluate };
}




//=============================================================================
template<typename HydroSystem>
auto CloudProblem<HydroSystem>::new_schedule(const mara::config_t& cfg)
{
    auto schedule = mara::schedule_t();
    schedule.create_and_mark_as_due("write_checkpoint");
    schedule.create_and_mark_as_due("write_diagnostics");
    schedule.create_and_mark_as_due("write_time_series");
    return schedule;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::create_schedule(const mara::config_t& cfg)
{
    auto restart = cfg.get_string("restart");
    return restart.empty()
    ? new_schedule(cfg)
    : mara::read_schedule(h5::File(restart, "r").open_group("schedule"));
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::next_schedule(const mara::schedule_t& schedule, const mara::config_t& cfg, double time)
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
            .update(mara::read_config(h5::File(args.at("restart"), "r").open_group("cfg")))
            .update(args)
    : new_run_config(args);
}




//=============================================================================
template<typename HydroSystem>
void CloudProblem<HydroSystem>::write_checkpoint(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_checkpoint");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("chkpt", count, "h5")), "w");
    write_solution(file.require_group("solution"), state.solution_state);
    mara::write_schedule(file.require_group("schedule"), state.schedule);
    mara::write_config(file.require_group("cfg"), state.run_config);

    std::printf("write checkpoint: %s\n", file.filename().data());
}

template<typename HydroSystem>
void CloudProblem<HydroSystem>::write_diagnostics(const app_state_t& state, std::string outdir)
{
    auto count = state.schedule.num_times_performed("write_diagnostics");
    auto file = h5::File(mara::filesystem::join(outdir, mara::create_numbered_filename("diagnostics", count, "h5")), "w");
    auto diagnostics = make_diagnostic_fields(state.solution_state, state.run_config);

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
    file.write("postshock_flow_power",  diagnostics.postshock_flow_power);
    file.write("postshock_flow_gamma",  diagnostics.postshock_flow_gamma);

    std::printf("write diagnostics: %s\n", file.filename().data());
}

template<typename HydroSystem>
void CloudProblem<HydroSystem>::write_time_series(const app_state_t& state, std::string outdir)
{
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::create_app_state(mara::config_t cfg)
{
    auto state = app_state_t();
    state.run_config     = cfg;
    state.solution_state = create_solution(cfg);
    state.schedule       = create_schedule(cfg);
    return state;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::next(const app_state_t& state)
{
    auto next_state = state;
    next_state.solution_state = next_solution(state);
    next_state.schedule       = next_schedule(state.schedule, state.run_config, state.solution_state.time);
    return next_state;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::simulation_should_continue(const app_state_t& state)
{
    auto time = state.solution_state.time;
    auto tfinal = state.run_config.get_double("tfinal");
    return time < tfinal;
}

template<typename HydroSystem>
auto CloudProblem<HydroSystem>::run_tasks(const app_state_t& state)
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
template<typename HydroSystem>
void CloudProblem<HydroSystem>::print_run_loop_message(const solution_state_t& solution, mara::perf_diagnostics_t perf)
{
    auto kzps = solution.radial_vertices.size() * solution.polar_vertices.size() / perf.execution_time_ms;
    std::printf("[%04d] t=%3.7lf kzps=%3.2lf\n", solution.iteration.as_integral(), solution.time, kzps);
}

template<typename HydroSystem>
void CloudProblem<HydroSystem>::print_run_dimensions(std::ostream& output, const mara::config_t& cfg)
{
    auto c2 = light_speed_cgs * light_speed_cgs;
    auto atmosphere_model = make_atmosphere_model(cfg);
    auto jet_nozzle_model = make_jet_nozzle_model(cfg);

    output << "====================================================\n";
    output << "model description:\n\n";
    output << "\treference length.................. " << atmosphere_model.r0 << " cm" << std::endl;
    output << "\treference time.................... " << atmosphere_model.r0 / light_speed_cgs << " s" << std::endl;
    output << "\treference mass.................... " << atmosphere_model.total_mass() << " g" << std::endl;
    output << "\treference density................. " << atmosphere_model.total_mass() / std::pow(atmosphere_model.r0, 3) << " g/cm^3" << std::endl;
    output << "\treference energy.................. " << atmosphere_model.total_mass() * c2 << " erg" << std::endl;
    output << "\tdensity at cloud base............. " << atmosphere_model.density_at(atmosphere_model.r0) << " g/cm^3" << std::endl;
    output << "\tdensity at cloud cutoff........... " << atmosphere_model.density_at(atmosphere_model.rc) << " g/cm^3" << std::endl;
    output << "\tjet mass density at base.......... " << jet_nozzle_model.density_at_base() << " g/cm^3" << std::endl;
    output << "\tjet Lorentz factor at q=0, t=0s... " << jet_nozzle_model.gamma_beta(0, 0) << std::endl;
    output << "\tjet Lorentz factor at q=0, t=1s... " << jet_nozzle_model.gamma_beta(0, 1) << std::endl;
    output << "\texplosion E / M................... " << jet_nozzle_model.Ej / (atmosphere_model.total_mass() * c2) << std::endl;
    output << std::endl;
}

template<typename HydroSystem>
void CloudProblem<HydroSystem>::prepare_filesystem(const mara::config_t& cfg)
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
        mara::write_config(file.require_group("cfg"), cfg);
    }
}




//=============================================================================
class subprog_cloud : public mara::sub_program_t
{
public:

    template<typename HydroSystem>
    int run_main(const mara::config_t& cfg)
    {
        using prob             = CloudProblem<HydroSystem>;
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
            prob::print_run_loop_message(state.solution_state, perf);
        }

        run_tasks_on_next(state);
        return 0;
    }

    int main(int argc, const char* argv[]) override
    {
        return run_main<mara::srhd>(create_run_config(argc, argv));
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