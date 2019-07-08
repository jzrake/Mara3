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
    auto sink_rate_field(const solver_data_t& solver_data, location_2d_t sink_location);
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

    return solver_data.vertices.map([accel] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(accel);
    });
}

auto binary::sink_rate_field(const solver_data_t& solver_data, location_2d_t sink_location)
{
    auto sink = [sink_location, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (location_2d_t field_point)
    {
        auto dr = field_point - sink_location;
        auto s2 = sink_radius * sink_radius;
        auto a2 = (dr * dr).sum() / s2 / 2.0;
        return sink_rate * 0.5 * std::exp(-a2);
    };

    return solver_data.vertices.map([sink] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(sink);
    });
}




//=============================================================================
binary::solution_t binary::advance(const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode)
{
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
     * @param[in]  riemann_solver  The riemann solver to use
     * @param[in]  axis            The axis to get the fluxes on
     *
     * @return     An array operator that returns arrays of fluxes
     */
    auto intercell_flux = [cs2=std::pow(solver_data.mach_number, -2.0)] (auto riemann_solver, std::size_t axis)
    {
        return [axis, riemann_solver, cs2] (auto left_and_right_states)
        {
            using namespace std::placeholders;
            auto nh = mara::unit_vector_t::on_axis(axis);
            auto riemann = std::bind(riemann_solver, _1, _2, cs2, cs2, nh);
            return left_and_right_states | nd::apply(riemann);
        };
    };


    // Minor helper functions
    //=========================================================================
    auto evaluate = nd::to_shared();
    auto force_to_source_terms = [] (force_2d_t f)
    {
        return mara::iso2d::conserved_t() / mara::make_time(1.0);
        // return mara::iso2d::flow_t{0.0, f[0].value, f[1].value};
        // TODO
    };
    auto recover_primitive = [] (auto&& u) { return mara::iso2d::recover_primitive(u); };
    auto cross_prod_z = [] (auto r, auto f) { return r[0] * f[1] - r[1] * f[0]; };


    // Binary parameters
    //=========================================================================
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);
    auto body1_pos = location_2d_t{binary.body1.position_x, binary.body1.position_y};
    auto body2_pos = location_2d_t{binary.body2.position_x, binary.body2.position_y};
    auto body1_vel = velocity_2d_t{binary.body1.velocity_x, binary.body1.velocity_y};
    auto body2_vel = velocity_2d_t{binary.body2.velocity_x, binary.body2.velocity_y};


    // Intermediate scheme data
    //=========================================================================
    auto v0  =  solver_data.vertices;
    auto dA  =  solver_data.cell_areas;
    auto u0  =  solution.conserved;
    auto p0  =  u0.map(nd::map(recover_primitive)).map(evaluate, tree_launch);
    auto dx  =  v0.map([] (auto v) { return v | component<0>() | nd::difference_on_axis(0); });
    auto dy  =  v0.map([] (auto v) { return v | component<1>() | nd::difference_on_axis(1); });
    auto fx  =  extend(p0, 0).map(extrapolate(0), tree_launch).map(intercell_flux(mara::iso2d::riemann_hlle, 0)) * dy;
    auto fy  =  extend(p0, 1).map(extrapolate(1), tree_launch).map(intercell_flux(mara::iso2d::riemann_hlle, 1)) * dx;
    auto lx  = -fx.map(nd::difference_on_axis(0));
    auto ly  = -fy.map(nd::difference_on_axis(1));
    auto m0  =  u0.map(component<0>()) * dA; // cell masses


    // Gravitational force, sink fields, and buffer zone source term
    //=========================================================================
    auto fg1 =        grav_vdot_field(solver_data, body1_pos, binary.body1.mass) * m0;
    auto fg2 =        grav_vdot_field(solver_data, body2_pos, binary.body2.mass) * m0;
    auto ss1 =  -u0 * sink_rate_field(solver_data, body1_pos) * dA;
    auto ss2 =  -u0 * sink_rate_field(solver_data, body2_pos) * dA;
    auto sg  =  (fg1 + fg2).map(nd::map(force_to_source_terms));
    auto ss  =  (ss1 + ss2);
    auto bz  =  (solver_data.initial_conserved - u0) * solver_data.buffer_rate_field * dA;


    // The updated conserved densities
    //=========================================================================
    auto u1 = u0 + (lx + ly + ss + sg + bz) * dt / dA;
    auto next_conserved = u1.map(evaluate, tree_launch);

    if (! safe_mode && (next_conserved.map(component<0>()) < 0.0).map(nd::any()).any())
    {
        std::cout << "binary::advance (negative density; re-trying in safe mode)" << std::endl;
        return advance(solution, solver_data, dt, true);
    }


    // The total force on each component, Mdot's, Ldot's, and Edot's.
    //=========================================================================
    auto xc = solver_data.cell_centers.map(component<0>());
    auto yc = solver_data.cell_centers.map(component<1>());
    auto fg1_tot = -fg1.map(nd::sum(), tree_launch).sum();
    auto fg2_tot = -fg2.map(nd::sum(), tree_launch).sum();
    auto ss1_tot = -ss1.map(component<0>()).map(nd::sum(), tree_launch).sum();
    auto ss2_tot = -ss2.map(component<0>()).map(nd::sum(), tree_launch).sum();
    auto px_accrete1_rate = -ss1.map(component<1>());
    auto py_accrete1_rate = -ss1.map(component<2>());
    auto px_accrete2_rate = -ss2.map(component<1>());
    auto py_accrete2_rate = -ss2.map(component<2>());
    auto px_ejection_rate =  -bz.map(component<1>());
    auto py_ejection_rate =  -bz.map(component<2>());
    auto m0_ejection_rate =  -bz.map(component<0>()).map(nd::sum(), tree_launch).sum();
    auto lz_ejection_rate = (xc * py_ejection_rate - yc * px_ejection_rate).map(nd::sum(), tree_launch).sum();
    auto lz_accrete1_rate = (xc * py_accrete1_rate - yc * px_accrete1_rate).map(nd::sum(), tree_launch).sum();
    auto lz_accrete2_rate = (xc * py_accrete2_rate - yc * px_accrete2_rate).map(nd::sum(), tree_launch).sum();

    auto Mdot = mara::make_sequence(ss1_tot, ss2_tot);
    auto Kdot = mara::make_sequence(lz_accrete1_rate, lz_accrete2_rate);
    auto Ldot = mara::make_sequence(cross_prod_z(body1_pos, fg1_tot), cross_prod_z(body2_pos, fg2_tot));
    auto Edot = mara::make_sequence((fg1_tot * body1_vel).sum(), (fg2_tot * body2_vel).sum());


    // The full updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        next_conserved,

        solution.mass_accreted_on             + Mdot * dt,
        solution.angular_momentum_accreted_on + Kdot * dt,
        solution.integrated_torque_on         + Ldot * dt,
        solution.work_done_on                 + Edot * dt,
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
