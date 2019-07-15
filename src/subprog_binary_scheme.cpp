#include <iostream>
#include "core_ndarray_ops.hpp"
#include "math_interpolation.hpp"
#include "mesh_prolong_restrict.hpp"
#include "mesh_tree_operators.hpp"
#include "subprog_binary.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY
static std::launch tree_launch = std::launch::deferred;




//=============================================================================
namespace binary
{
    auto grav_vdot_field(const solver_data_t& solver_data, location_2d_t body_location, mara::unit_mass<double> body_mass);
    auto sink_rate_field(const solver_data_t& solver_data, location_2d_t sink_location, mara::unit_time<double> dt);
}




//=============================================================================
void binary::set_scheme_globals(const mara::config_t& run_config)
{
    tree_launch = run_config.get_int("threaded") == 0 ? std::launch::deferred : std::launch::async;
}




//=============================================================================
template<std::size_t I>
static auto component()
{
    return nd::map([] (auto p) { return mara::get<I>(p); });
};

auto binary::grav_vdot_field(const solver_data_t& solver_data, location_2d_t body_location, mara::unit_mass<double> body_mass)
{
    auto accel = [body_location, body_mass, softening_radius=solver_data.softening_radius](location_2d_t field_point)
    {
        auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
        auto dr  = field_point - body_location;
        auto dr2 = dr[0] * dr[0] + dr[1] * dr[1];
        auto rs2 = softening_radius * softening_radius;
        return -dr / (dr2 + rs2).pow<3, 2>() * G * body_mass;
    };

    return solver_data.cell_centers.map([accel] (auto block)
    {
        return block | nd::map(accel);
    });
}

auto binary::sink_rate_field(const solver_data_t& solver_data, location_2d_t sink_location, mara::unit_time<double> dt)
{
    auto sink = [dt, sink_location, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (location_2d_t field_point)
    {
        auto dr = field_point - sink_location;
        auto s2 = sink_radius * sink_radius;
        auto a2 = (dr * dr).sum() / s2 / 2.0;
        auto tau_inverse = sink_rate * 0.5 * std::exp(-a2);
        auto tau_inverse_exact = tau_inverse * (1.0 - std::exp(-(dt * tau_inverse).scalar()));
        return tau_inverse_exact;
    };

    return solver_data.cell_centers.map([sink] (auto block)
    {
        return block | nd::map(sink);
    });
}




//=============================================================================
binary::solution_t binary::advance(const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode)
{

    // Note: this scheme still uses globally constant sound speed!
    auto cs2 = std::pow(solver_data.mach_number, -2.0);
    auto sr2 = solver_data.gst_suppr_radius.pow<2>();


    /*
     * @brief      Extend each block in a given tree of cell-wise values, with
     *             two guard zones on the given axis.
     *
     * @param[in]  tree  The tree whose blocks need to be extended
     * @param[in]  axis  The axis to extend on
     *
     * @return     A function that operates on trees
     */
    auto extend = [] (auto tree, std::size_t axis)
    {
        return tree.indexes().map([tree, axis] (auto index)
        {
            auto C = tree.at(index);
            auto L = mara::get_cell_block(tree, index.prev_on(axis), mara::compose(nd::to_shared(), nd::select_final(2, axis)));
            auto R = mara::get_cell_block(tree, index.next_on(axis), mara::compose(nd::to_shared(), nd::select_first(2, axis)));
            return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis);
        }, tree_launch);
    };


    /*
     * @brief      Apply a piecewise linear extrapolation, along the given axis,
     *             to the cells in each array of a tree.
     *
     * @param[in]  axis  The axis to extrapolate on
     *
     * @return     A function that operates on trees
     */
    auto extrapolate = [plm_theta=safe_mode ? 0.0 : solver_data.plm_theta] (std::size_t axis)
    {
        return [plm_theta, axis] (auto P)
        {
            using namespace std::placeholders;
            auto L1 = nd::select_axis(axis).from(0).to(1).from_the_end();
            auto R1 = nd::select_axis(axis).from(1).to(0).from_the_end();
            auto L2 = nd::select_axis(axis).from(1).to(2).from_the_end();
            auto R2 = nd::select_axis(axis).from(2).to(1).from_the_end();

            auto G = P
            | nd::zip_adjacent3_on_axis(axis)
            | nd::apply([plm_theta] (auto a, auto b, auto c) { return mara::plm_gradient(a, b, c, plm_theta); })
            | nd::multiply(0.5)
            | nd::to_shared();

            return nd::zip(
                (P | L2) + (G | L1),
                (P | R2) - (G | R1));
        };
    };


    /*
     * @brief      Return an array of intercell fluxes by calling the specified
     *             riemann solver
     *
     * @param[in]  axis  The axis to get the fluxes on
     *
     * @return     An array operator that returns arrays of fluxes
     */
    auto intercell_flux = [cs2] (std::size_t axis)
    {
        return [axis, cs2] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(mara::iso2d::riemann_hlle, _1, _2, cs2, cs2, nh);
            return left_and_right_states | nd::apply(riemann);
        };
    };


    /*
     * @brief      Return a tree of primitive variables given a tree of
     *             angular-momentum conserving variables and a tree of the
     *             cell-center coordinates.
     *
     * @param[in]  q0    The tree of angular-momentum conserving variables
     * @param[in]  xc    The tree of cell-center coordinates
     *
     * @return     A tree of primitives
     */
    auto recover_primitive = [] (const auto& q0, const auto& xc)
    {
        auto q_to_p = [] (auto q, auto x) { return mara::iso2d::recover_primitive(q, x); };

        return q0
        .pair(xc)
        .apply([q_to_p] (auto Q, auto X)
        {
            return nd::zip(Q, X) | nd::apply(q_to_p) | nd::to_shared();
        }, tree_launch);
    };


    /*
     * @brief      Return a tree of fluxes of angular-momentum conserving
     *             variables (Q), given a tree of linear-momentum conserving
     *             variables (U) and the associated coordinates.
     *
     * @param[in]  F     The tree of U-fluxes
     * @param[in]  X     The tree of coordinates
     *
     * @return     The tree of Q-fluxes
     */
    auto to_angmom_fluxes = [rd=solver_data.domain_radius] (std::size_t axis)
    {
        return [axis, rd] (const auto& F, const auto& X)
        {
            return nd::zip(F, X) | nd::apply([axis, rd] (auto f, auto x)
            {
                auto flux_sigma = mara::get<0>(f);
                auto flux_px    = mara::get<1>(f);
                auto flux_py    = mara::get<2>(f);
                auto flux_sr    = x[0] * flux_px + x[1] * flux_py;
                auto flux_lz    = x[0] * flux_py - x[1] * flux_px;

                if (axis == 0 && (x[0] == -rd || x[0] == rd)) flux_lz = 0.0;
                if (axis == 1 && (x[1] == -rd || x[1] == rd)) flux_lz = 0.0;

                return mara::make_arithmetic_tuple(flux_sigma, flux_sr, flux_lz);
            });
        };
    };


    // Minor helper functions
    //=========================================================================
    auto evaluate = nd::to_shared();
    auto force_to_source_terms = [] (force_2d_t f, location_2d_t x)
    {
        auto sigma_dot = mara::make_dimensional<0, 1, -1>(0.0);
        auto sr_dot    = x[0] * f[0] + x[1] * f[1];
        auto lz_dot    = x[0] * f[1] - x[1] * f[0];
        return mara::make_arithmetic_tuple(sigma_dot, sr_dot, lz_dot);
    };
    auto force_to_source_terms_tree = [force_to_source_terms] (auto F, auto X)
    {
        return nd::zip(F, X) | nd::apply(force_to_source_terms);
    };
    auto geometrical_source_terms_tree = [cs2, sr2] (auto P, auto X)
    {
        auto ramp = X | nd::map([sr2] (auto x)
        {
            return 1.0 - std::exp(-(x * x).sum() / sr2);
        });

        return P
        | nd::map([cs2] (auto p) { return p.source_terms_conserved_angmom(cs2); })
        | nd::multiply(ramp);
    };


    // Binary parameters
    //=========================================================================
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);
    auto body1_pos = location_2d_t{binary.body1.position_x, binary.body1.position_y};
    auto body2_pos = location_2d_t{binary.body2.position_x, binary.body2.position_y};
    // auto body1_vel = velocity_2d_t{binary.body1.velocity_x, binary.body1.velocity_y};
    // auto body2_vel = velocity_2d_t{binary.body2.velocity_x, binary.body2.velocity_y};


    // Intermediate scheme data
    //=========================================================================
    auto v0  =  solver_data.vertices;
    auto dA  =  solver_data.cell_areas;
    auto q0  =  solution.conserved;
    auto p0  =  recover_primitive(solution.conserved, solver_data.cell_centers);
    auto dx  =  v0.map([] (auto v) { return v | component<0>() | nd::difference_on_axis(0); });
    auto dy  =  v0.map([] (auto v) { return v | component<1>() | nd::difference_on_axis(1); });
    auto xf  =  v0.map([] (auto v) { return v | nd::midpoint_on_axis(1); });
    auto yf  =  v0.map([] (auto v) { return v | nd::midpoint_on_axis(0); });
    auto fx  =  extend(p0, 0).map(extrapolate(0), tree_launch).map(intercell_flux(0));
    auto fy  =  extend(p0, 1).map(extrapolate(1), tree_launch).map(intercell_flux(1));
    auto gx  =  fx.pair(xf).apply(to_angmom_fluxes(0)) * dy;
    auto gy  =  fy.pair(yf).apply(to_angmom_fluxes(1)) * dx;
    auto lx  = -gx.map(nd::difference_on_axis(0));
    auto ly  = -gy.map(nd::difference_on_axis(1));
    auto m0  =  q0.map(component<0>()) * dA; // cell masses


    // Gravitational force, sink fields, and buffer zone source term
    //=========================================================================
    auto fg1 =        grav_vdot_field(solver_data, body1_pos, binary.body1.mass) * m0;
    auto fg2 =        grav_vdot_field(solver_data, body2_pos, binary.body2.mass) * m0;
    auto sg1 =   fg1.pair(solver_data.cell_centers).apply(force_to_source_terms_tree);
    auto sg2 =   fg2.pair(solver_data.cell_centers).apply(force_to_source_terms_tree);
    auto ss1 =  -q0 * sink_rate_field(solver_data, body1_pos, dt) * dA;
    auto ss2 =  -q0 * sink_rate_field(solver_data, body2_pos, dt) * dA;
    auto st  =   p0.pair(solver_data.cell_centers).apply(geometrical_source_terms_tree) * dA;
    auto bz  =  (solver_data.initial_conserved - q0) * solver_data.buffer_rate_field * dA;


    // The updated conserved densities
    //=========================================================================
    auto q1 = q0 + (lx + ly + ss1 + ss2 + sg1 + sg2 + bz + st) * dt / dA;
    auto next_conserved = q1.map(evaluate, tree_launch);

    if (! safe_mode && (next_conserved.map(component<0>()) < 0.0).map(nd::any()).any())
    {
        next_conserved
        .pair(solver_data.cell_centers)
        .sink([] (auto ux)
        {
            for (auto [u, x] : nd::zip(std::get<0>(ux), std::get<1>(ux)))
            {
                if (mara::get<0>(u) < 0.0)
                {
                    std::cout << "negative density " << mara::get<0>(u).value << " at " << mara::to_string(x) << std::endl;
                }
            }
        });

        std::cout << "binary::advance (negative density; re-trying in safe mode)" << std::endl;
        return advance(solution, solver_data, dt, true);
    }


    // The total force on each component, Mdot's, Ldot's, and Edot's.
    //=========================================================================
    auto mdot1 = -ss1.map(component<0>()).map(nd::sum(), tree_launch).sum();
    auto mdot2 = -ss2.map(component<0>()).map(nd::sum(), tree_launch).sum();
    auto kdot1 = -ss1.map(component<2>()).map(nd::sum(), tree_launch).sum();
    auto kdot2 = -ss2.map(component<2>()).map(nd::sum(), tree_launch).sum();
    auto ldot1 = -sg1.map(component<2>()).map(nd::sum(), tree_launch).sum();
    auto ldot2 = -sg2.map(component<2>()).map(nd::sum(), tree_launch).sum();
    auto m0_ejection_rate = -bz.map(component<0>()).map(nd::sum(), tree_launch).sum();
    auto lz_ejection_rate = -bz.map(component<2>()).map(nd::sum(), tree_launch).sum();

    auto Mdot = mara::make_sequence(mdot1, mdot2);
    auto Kdot = mara::make_sequence(kdot1, kdot2);
    auto Ldot = mara::make_sequence(ldot1, ldot2);


    // The full updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        next_conserved,

        solution.mass_accreted_on             + Mdot * dt,
        solution.angular_momentum_accreted_on + Kdot * dt,
        solution.integrated_torque_on         + Ldot * dt,
        solution.work_done_on                 , // + Edot * dt,
        solution.mass_ejected                 + m0_ejection_rate * dt,
        solution.angular_momentum_ejected     + lz_ejection_rate * dt,
    };
}




//=============================================================================
binary::solution_t binary::solution_t::operator+(const solution_t& other) const
{
    return {
        time       + other.time,
        iteration  + other.iteration,
        (conserved + other.conserved).map(nd::to_shared(), tree_launch),

        mass_accreted_on               + other.mass_accreted_on,
        angular_momentum_accreted_on   + other.angular_momentum_accreted_on,
        integrated_torque_on           + other.integrated_torque_on,
        work_done_on                   + other.work_done_on,
        mass_ejected                   + other.mass_ejected,
        angular_momentum_ejected       + other.angular_momentum_ejected,
    };
}

binary::solution_t binary::solution_t::operator*(mara::rational_number_t scale) const
{
    return {
        time       * scale.as_double(),
        iteration  * scale,
        (conserved * scale.as_double()).map(nd::to_shared(), tree_launch),

        mass_accreted_on               * scale.as_double(),
        angular_momentum_accreted_on   * scale.as_double(),
        integrated_torque_on           * scale.as_double(),
        work_done_on                   * scale.as_double(),
        mass_ejected                   * scale.as_double(),
        angular_momentum_ejected       * scale.as_double(),
    };
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
