#include <iostream>
#include "core_ndarray_ops.hpp"
#include "math_interpolation.hpp"
#include "mesh_prolong_restrict.hpp"
#include "mesh_tree_operators.hpp"
#include "subprog_binary.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




static std::launch tree_launch = std::launch::deferred;
using prim_pair_t = std::tuple<mara::iso2d::primitive_t, mara::iso2d::primitive_t>;
using force_per_area_t = mara::arithmetic_sequence_t<mara::dimensional_value_t<-1, 1, -2, double>, 2>;




#define hard_coded_alpha 0.1




//=============================================================================
namespace binary
{
    // auto grav_vdot_field(const solver_data_t& solver_data, location_2d_t body_location, mara::unit_mass<double> body_mass);
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




//=============================================================================
static auto strip_axis(std::size_t axis, std::size_t count)
{
    return [axis, count] (auto A)
    {
        return A | nd::select_axis(axis).from(count).to(count).from_the_end();
    };
};




//=============================================================================
// auto binary::grav_vdot_field(const solver_data_t& solver_data, location_2d_t body_location, mara::unit_mass<double> body_mass)
// {
//     auto accel = [body_location, body_mass, softening_radius=solver_data.softening_radius](location_2d_t field_point)
//     {
//         auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
//         auto dr  = field_point - body_location;
//         auto dr2 = dr[0] * dr[0] + dr[1] * dr[1];
//         auto rs2 = softening_radius * softening_radius;
//         return -dr / (dr2 + rs2).pow<3, 2>() * G * body_mass;
//     };

//     return solver_data.cell_centers.map([accel] (auto block)
//     {
//         return block | nd::map(accel);
//     });
// }




//=============================================================================
auto binary::sink_rate_field(const solver_data_t& solver_data, location_2d_t sink_location, mara::unit_time<double> dt)
{
    auto sink = [dt, sink_location, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (location_2d_t field_point)
    {
        auto dr = field_point - sink_location;
        auto s2 = sink_radius * sink_radius;
        auto a2 = (dr * dr).sum() / s2 / 2.0;
        auto tau_inverse = sink_rate * 0.5 * std::exp(-a2);
        auto tau_inverse_exact = mara::make_dimensional<0,0,0>(1.0 - std::exp(-(dt * tau_inverse).scalar())) / dt;
        return tau_inverse_exact;
    };

    return solver_data.cell_centers.map([sink] (auto block)
    {
        return block | nd::map(sink);
    });
}




//=============================================================================
static auto grav_vdot_field(const binary::solver_data_t& solver_data, binary::location_2d_t body_location, mara::unit_mass<double> body_mass)
{
    return [body_location, body_mass, softening_radius=solver_data.softening_radius](binary::location_2d_t field_point)
    {
        auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
        auto dr  = field_point - body_location;
        auto dr2 = dr[0] * dr[0] + dr[1] * dr[1];
        auto rs2 = softening_radius * softening_radius;
        return -dr / (dr2 + rs2).pow<3, 2>() * G * body_mass;
    };
}




//=============================================================================
static auto sink_rate_field(const binary::solver_data_t& solver_data, binary::location_2d_t sink_location)
{
    return [sink_location, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (binary::location_2d_t field_point)
    {
        auto dr = field_point - sink_location;
        auto s2 = sink_radius * sink_radius;
        auto a2 = (dr * dr).sum() / s2 / 2.0;
        return sink_rate * 0.5 * std::exp(-a2);
    };
}




/*
 * @brief      Extend each block in a given tree of cell-wise values, with two
 *             guard zones on the given axis.
 *
 * @param[in]  tree         The tree whose blocks need to be extended
 * @param[in]  axis         The axis to extend on
 * @param[in]  guard_count  The number of guard zones to extend by on each axis
 *
 * @tparam     TreeType     { description }
 *
 * @return     A function that operates on trees
 */
template<typename TreeType>
static auto extend(TreeType tree, std::size_t axis, std::size_t guard_count)
{
    return tree.indexes().map([tree, axis, guard_count] (auto index)
    {
        auto C = tree.at(index);
        auto L = mara::get_cell_block(tree, index.prev_on(axis), mara::compose(nd::to_shared(), nd::select_final(guard_count, axis)));
        auto R = mara::get_cell_block(tree, index.next_on(axis), mara::compose(nd::to_shared(), nd::select_first(guard_count, axis)));
        return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis);
    }, tree_launch);
};




//=============================================================================
template<typename PrimitiveArray>
static auto estimate_gradient(PrimitiveArray p0, std::size_t axis, double plm_theta)
{
    return p0
    | nd::zip_adjacent3_on_axis(axis)
    | nd::apply([plm_theta] (auto a, auto b, auto c) { return mara::plm_gradient(a, b, c, plm_theta); });
};




//=============================================================================
static auto cs2_at_position(binary::location_2d_t x, double mach_number)
{
    auto GM = 1.0;
    auto r2 = (x * x).sum().value;
    return GM / std::sqrt(r2) / mach_number / mach_number;
};




//=============================================================================
static auto scale_height_at_position(binary::location_2d_t x, double mach_number)
{
    auto r = std::sqrt((x * x).sum().value);
    return r / mach_number;
};




//=============================================================================
static auto to_angmom_fluxes(std::size_t axis, mara::unit_length<double> domain_radius)
{
    return [axis, rd=domain_radius] (binary::location_2d_t x, mara::iso2d::flux_t f)
    {
        auto flux_sigma = mara::get<0>(f);
        auto flux_px    = mara::get<1>(f);
        auto flux_py    = mara::get<2>(f);
        auto flux_sr    = x[0] * flux_px + x[1] * flux_py;
        auto flux_lz    = x[0] * flux_py - x[1] * flux_px;

        if (axis == 0 && (x[0] == -rd || x[0] == rd)) flux_lz = 0.0;
        if (axis == 1 && (x[1] == -rd || x[1] == rd)) flux_lz = 0.0;

        return mara::make_arithmetic_tuple(flux_sigma, flux_sr, flux_lz);
    };
};




//=============================================================================
static mara::iso2d::flux_t viscous_flux(std::size_t axis, prim_pair_t g_long, prim_pair_t g_tran, double mu, double grid_spacing)
{
    auto [gl, gr] = g_long;
    auto [hl, hr] = g_tran;

    switch (axis)
    {
        case 0:
        {
            auto dx_ux = 0.5 * (gl.velocity_x() + gr.velocity_x());
            auto dx_uy = 0.5 * (gl.velocity_y() + gr.velocity_y());
            auto dy_ux = 0.5 * (hl.velocity_x() + hr.velocity_x());
            auto dy_uy = 0.5 * (hl.velocity_y() + hr.velocity_y());

            auto tauxx = mu * (dx_ux - dy_uy) / grid_spacing;
            auto tauxy = mu * (dx_uy + dy_ux) / grid_spacing;

            return mara::make_arithmetic_tuple(
                mara::make_dimensional<-1, 1, -1>(0.0),
                mara::make_dimensional< 0, 1,-2>(tauxx),
                mara::make_dimensional< 0, 1,-2>(tauxy));
        }
        case 1:
        {
            auto dx_ux = 0.5 * (hl.velocity_x() + hr.velocity_x());
            auto dx_uy = 0.5 * (hl.velocity_y() + hr.velocity_y());
            auto dy_ux = 0.5 * (gl.velocity_x() + gr.velocity_x());
            auto dy_uy = 0.5 * (gl.velocity_y() + gr.velocity_y());

            auto tauyx =  mu * (dx_uy + dy_ux) / grid_spacing;
            auto tauyy = -mu * (dx_ux - dy_uy) / grid_spacing;

            return mara::make_arithmetic_tuple(
                mara::make_dimensional<-1, 1, -1>(0.0),
                mara::make_dimensional< 0, 1,-2>(tauyx),
                mara::make_dimensional< 0, 1,-2>(tauyy));
        }
    }
    throw;
}




//=============================================================================
static auto intercell_flux(std::size_t axis, const binary::solver_data_t& solver_data)
{
    auto to_angmom = to_angmom_fluxes(axis, solver_data.domain_radius);

    return [axis, solver_data, to_angmom] (binary::location_2d_t xf, prim_pair_t p0, prim_pair_t g_long, prim_pair_t g_tran)
    {
        auto grid_spacing = 1.0; // TODO!
        auto [pl, pr] = p0;
        auto [gl, gr] = g_long;

        auto pl_hat = pl + gl * 0.5;
        auto pr_hat = pr - gr * 0.5;
        auto cs2 = cs2_at_position(xf, solver_data.mach_number);
        auto nu = hard_coded_alpha * std::sqrt(cs2) * scale_height_at_position(xf, solver_data.mach_number);
        auto mu = 0.5 * nu * (pl_hat.sigma() + pr_hat.sigma());

        auto nhat = mara::unit_vector_t::on_axis(axis);
        auto fhat = mara::iso2d::riemann_hlle(pl_hat, pr_hat, cs2, cs2, nhat);
        auto fhat_visc = viscous_flux(axis, g_long, g_tran, mu, grid_spacing);

        return to_angmom(xf, fhat + fhat_visc);
    };
}




static auto force_to_source_terms(binary::location_2d_t x, force_per_area_t f)
{
    auto sigma_dot = mara::make_dimensional<-2, 1, -1>(0.0);
    auto sr_dot    = x[0] * f[0] + x[1] * f[1];
    auto lz_dot    = x[0] * f[1] - x[1] * f[0];
    return mara::make_arithmetic_tuple(sigma_dot, sr_dot, lz_dot);
};




//=============================================================================
static auto source_terms(
    const binary::solution_t& solution,
    const binary::solver_data_t& solver_data,
    mara::tree_index_t<2> tree_index,
    mara::unit_time<double> dt)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);
    auto body1_pos = binary::location_2d_t{binary.body1.position_x, binary.body1.position_y};
    auto body2_pos = binary::location_2d_t{binary.body2.position_x, binary.body2.position_y};


    auto sr2 = solver_data.gst_suppr_radius.pow<2>();
    auto M   = solver_data.mach_number;
    auto xc  = solver_data.cell_centers.at(tree_index);
    auto dA  = solver_data.cell_areas.at(tree_index);
    auto br  = solver_data.buffer_rate_field.at(tree_index);
    auto q0  = solution.conserved.at(tree_index);


    // TODO: use cached primitives
    auto p0 = nd::zip(q0, xc)
    | nd::apply([] (auto q, auto x) { return mara::iso2d::recover_primitive(q, x); })
    | nd::to_shared();


    auto sigma = q0 | component<0>();
    auto fg1 = (xc | nd::map(grav_vdot_field(solver_data, body1_pos, binary.body1.mass))) * sigma;
    auto fg2 = (xc | nd::map(grav_vdot_field(solver_data, body2_pos, binary.body2.mass))) * sigma;

    auto s_grav_1 = (nd::zip(xc, fg1) | nd::apply(force_to_source_terms)) * dt;
    auto s_grav_2 = (nd::zip(xc, fg2) | nd::apply(force_to_source_terms)) * dt;
    auto s_sink_1 = -q0 * (xc | nd::map(sink_rate_field(solver_data, body1_pos))) * dt;
    auto s_sink_2 = -q0 * (xc | nd::map(sink_rate_field(solver_data, body2_pos))) * dt;
    auto s_buffer = (solver_data.initial_conserved.at(tree_index) - q0) * br * dt;
    auto s_geom = nd::zip(p0, xc) | nd::apply([sr2, M, dt] (auto p, auto x)
    {
        auto ramp = 1.0 - std::exp(-(x * x).sum() / sr2);
        auto cs2 = cs2_at_position(x, M);
        return p.source_terms_conserved_angmom(cs2) * ramp * dt;
    });

    return s_grav_1 + s_grav_2 + s_sink_1 + s_sink_2 + s_buffer + s_geom | nd::to_shared();
}




//=============================================================================
static auto block_update(
    const binary::solution_t& solution,
    const binary::solver_data_t& solver_data,
    mara::unit_time<double> dt,
    bool safe_mode)
{
    auto extended_xc = extend(extend(solver_data.cell_centers, 0, 2), 1, 2);

    return [solver_data, solution, dt, extended_xc, safe_mode] (auto&& tree_index, auto&& extended_q0)
    {
        auto xv = solver_data.vertices.at(tree_index);
        auto xc = solver_data.cell_centers.at(tree_index);
        auto dA = solver_data.cell_areas.at(tree_index);
        auto dx = xv | component<0>() | nd::difference_on_axis(0);
        auto dy = xv | component<1>() | nd::difference_on_axis(1);
        auto xf = xv | nd::midpoint_on_axis(1);
        auto yf = xv | nd::midpoint_on_axis(0);

        auto p0 = nd::zip(extended_q0, extended_xc.at(tree_index))
        | nd::apply([] (auto q, auto x) { return mara::iso2d::recover_primitive(q, x); })
        | nd::to_shared();

        auto plm = safe_mode ? 0.0 : solver_data.plm_theta;
        auto gx = estimate_gradient(p0, 0, plm) | nd::to_shared();
        auto gy = estimate_gradient(p0, 1, plm) | nd::to_shared();

        auto fhat_x = nd::zip(
            xf,
            p0 | strip_axis(0, 1) | strip_axis(1, 2) | nd::zip_adjacent2_on_axis(0),
            gx |                    strip_axis(1, 2) | nd::zip_adjacent2_on_axis(0),
            gy | strip_axis(0, 1) | strip_axis(1, 1) | nd::zip_adjacent2_on_axis(0))
        | nd::apply(intercell_flux(0, solver_data))
        | nd::multiply(dy)
        | nd::to_shared();

        auto fhat_y = nd::zip(
            yf,
            p0 | strip_axis(1, 1) | strip_axis(0, 2) | nd::zip_adjacent2_on_axis(1),
            gy |                    strip_axis(0, 2) | nd::zip_adjacent2_on_axis(1),
            gx | strip_axis(1, 1) | strip_axis(0, 1) | nd::zip_adjacent2_on_axis(1))
        | nd::apply(intercell_flux(1, solver_data))
        | nd::multiply(dx)
        | nd::to_shared();

        auto lx = fhat_x | nd::difference_on_axis(0);
        auto ly = fhat_y | nd::difference_on_axis(1);
        auto q0 = solution.conserved.at(tree_index);

        auto dq_source = source_terms(solution, solver_data, tree_index, dt);
        return q0 + (lx + ly) * dt / dA + dq_source | nd::to_shared();
    };
}




//=============================================================================
binary::solution_t binary::advance(const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode)
{
    auto extended_q0 = extend(extend(solution.conserved, 0, 2), 1, 2);
    

    auto dq = extended_q0.pair_indexes().apply(block_update(solution, solver_data, dt, safe_mode));


    // The full updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        solution.conserved,

        solution.mass_accreted_on             ,//+ Mdot * dt,
        solution.angular_momentum_accreted_on ,//+ Kdot * dt,
        solution.integrated_torque_on         ,//+ Ldot * dt,
        solution.work_done_on                 ,//, // + Edot * dt,
        solution.mass_ejected                 ,//+ m0_ejection_rate * dt,
        solution.angular_momentum_ejected     ,//+ lz_ejection_rate * dt,
    };
}




//=============================================================================
// binary::solution_t binary::advance(const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode)
// {
    // auto sr2 = solver_data.gst_suppr_radius.pow<2>();
    // auto evaluate = nd::to_shared();


    // /*
    //  * @brief      Extend each block in a given tree of cell-wise values, with
    //  *             two guard zones on the given axis.
    //  *
    //  * @param[in]  tree  The tree whose blocks need to be extended
    //  * @param[in]  axis  The axis to extend on
    //  *
    //  * @return     A function that operates on trees
    //  */
    // auto extend = [] (auto tree, std::size_t guard_count, std::size_t axis)
    // {
    //     return tree.indexes().map([tree, guard_count, axis] (auto index)
    //     {
    //         auto C = tree.at(index);
    //         auto L = mara::get_cell_block(tree, index.prev_on(axis), mara::compose(nd::to_shared(), nd::select_final(guard_count, axis)));
    //         auto R = mara::get_cell_block(tree, index.next_on(axis), mara::compose(nd::to_shared(), nd::select_first(guard_count, axis)));
    //         return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis);
    //     }, tree_launch);
    // };


    // /*
    //  * @brief      Apply a piecewise linear extrapolation, along the given axis,
    //  *             to the cells in each array of a tree.
    //  *
    //  * @param[in]  axis  The axis to extrapolate on
    //  *
    //  * @return     A function that operates on trees
    //  */
    // auto extrapolate = [plm_theta=safe_mode ? 0.0 : solver_data.plm_theta] (std::size_t axis)
    // {
    //     return [plm_theta, axis] (auto P)
    //     {
    //         using namespace std::placeholders;
    //         auto L1 = nd::select_axis(axis).from(0).to(1).from_the_end();
    //         auto R1 = nd::select_axis(axis).from(1).to(0).from_the_end();
    //         auto L2 = nd::select_axis(axis).from(1).to(2).from_the_end();
    //         auto R2 = nd::select_axis(axis).from(2).to(1).from_the_end();

    //         auto G = P
    //         | nd::zip_adjacent3_on_axis(axis)
    //         | nd::apply([plm_theta] (auto a, auto b, auto c) { return mara::plm_gradient(a, b, c, plm_theta); })
    //         | nd::multiply(0.5)
    //         | nd::to_shared();

    //         return nd::zip(
    //             (P | L2) + (G | L1),
    //             (P | R2) - (G | R1));
    //     };
    // };


    // auto cs2_at_position = [Ma=solver_data.mach_number] (location_2d_t x)
    // {
    //     // return 1.0 / Ma / Ma;
    //     auto GM = 1.0;
    //     auto r2 = (x * x).sum().value;
    //     return GM / std::sqrt(r2) / Ma / Ma;
    // };


    // /*
    //  * @brief      Return an array of intercell fluxes by calling the specified
    //  *             riemann solver
    //  *
    //  * @param[in]  axis  The axis to get the fluxes on
    //  *
    //  * @return     An array operator that returns arrays of fluxes
    //  */
    // auto intercell_flux = [cs2_at_position] (std::size_t axis)
    // {
    //     return [axis, cs2_at_position] (auto left_and_right_states, auto face_coordinates)
    //     {
    //         auto nh = mara::unit_vector_t::on_axis(axis);
    //         auto Pl = nd::get<0>(left_and_right_states);
    //         auto Pr = nd::get<1>(left_and_right_states);

    //         return nd::zip(Pl, Pr, face_coordinates)
    //         | nd::apply([nh, cs2_at_position] (auto pl, auto pr, auto xface)
    //         {
    //             auto cs2 = cs2_at_position(xface);
    //             return mara::iso2d::riemann_hlle(pl, pr, cs2, cs2, nh);
    //         });
    //     };
    // };


    // /*
    //  * @brief      Return a tree of primitive variables given a tree of
    //  *             angular-momentum conserving variables and a tree of the
    //  *             cell-center coordinates.
    //  *
    //  * @param[in]  q0    The tree of angular-momentum conserving variables
    //  * @param[in]  xc    The tree of cell-center coordinates
    //  *
    //  * @return     A tree of primitives
    //  */
    // auto recover_primitive = [] (const auto& q0, const auto& xc)
    // {
    //     auto q_to_p = [] (auto q, auto x) { return mara::iso2d::recover_primitive(q, x); };

    //     return q0
    //     .pair(xc)
    //     .apply([q_to_p] (auto Q, auto X)
    //     {
    //         return nd::zip(Q, X) | nd::apply(q_to_p) | nd::to_shared();
    //     }, tree_launch);
    // };


    // /*
    //  * @brief      Return a tree of fluxes of angular-momentum conserving
    //  *             variables (Q), given a tree of linear-momentum conserving
    //  *             variables (U) and the associated coordinates.
    //  *
    //  * @param[in]  F     The tree of U-fluxes
    //  * @param[in]  X     The tree of coordinates
    //  *
    //  * @return     The tree of Q-fluxes
    //  */
    // auto to_angmom_fluxes = [rd=solver_data.domain_radius] (std::size_t axis)
    // {
    //     return [axis, rd] (const auto& F, const auto& X)
    //     {
    //         return nd::zip(F, X) | nd::apply([axis, rd] (auto f, auto x)
    //         {
    //             auto flux_sigma = mara::get<0>(f);
    //             auto flux_px    = mara::get<1>(f);
    //             auto flux_py    = mara::get<2>(f);
    //             auto flux_sr    = x[0] * flux_px + x[1] * flux_py;
    //             auto flux_lz    = x[0] * flux_py - x[1] * flux_px;

    //             if (axis == 0 && (x[0] == -rd || x[0] == rd)) flux_lz = 0.0;
    //             if (axis == 1 && (x[1] == -rd || x[1] == rd)) flux_lz = 0.0;

    //             return mara::make_arithmetic_tuple(flux_sigma, flux_sr, flux_lz);
    //         });
    //     };
    // };


    // auto force_to_source_terms = [] (force_2d_t f, location_2d_t x)
    // {
    //     auto sigma_dot = mara::make_dimensional<0, 1, -1>(0.0);
    //     auto sr_dot    = x[0] * f[0] + x[1] * f[1];
    //     auto lz_dot    = x[0] * f[1] - x[1] * f[0];
    //     return mara::make_arithmetic_tuple(sigma_dot, sr_dot, lz_dot);
    // };


    // auto force_to_source_terms_tree = [force_to_source_terms] (auto F, auto X)
    // {
    //     return nd::zip(F, X) | nd::apply(force_to_source_terms);
    // };


    // auto geometrical_source_terms_tree = [sr2, cs2_at_position] (auto P, auto X)
    // {
    //     auto ramp = X | nd::map([sr2] (auto x)
    //     {
    //         return 1.0 - std::exp(-(x * x).sum() / sr2);
    //     });

    //     auto CS2 = X | nd::map(cs2_at_position);

    //     return nd::zip(P, CS2)
    //     | nd::apply([] (auto p, auto cs2) { return p.source_terms_conserved_angmom(cs2); })
    //     | nd::multiply(ramp);
    // };


    // // Binary parameters
    // //=========================================================================
    // auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);
    // auto body1_pos = location_2d_t{binary.body1.position_x, binary.body1.position_y};
    // auto body2_pos = location_2d_t{binary.body2.position_x, binary.body2.position_y};
    // // auto body1_vel = velocity_2d_t{binary.body1.velocity_x, binary.body1.velocity_y};
    // // auto body2_vel = velocity_2d_t{binary.body2.velocity_x, binary.body2.velocity_y};


    // auto viscous_fluxes_tree_x_direction = [] (auto P, auto X)
    // {
    //     auto centered_stencil = [] (auto a, auto b, auto c) { return c - a; };
    //     auto nu = mara::make_dimensional<2, 0, -1>(0.01);

    //     auto d0 = P | nd::map([] (auto p) { return mara::make_dimensional<-2, 1, 0>(p.sigma()); });
    //     auto ux = P | nd::map([] (auto p) { return mara::make_velocity(p.velocity_x()); });
    //     auto uy = P | nd::map([] (auto p) { return mara::make_velocity(p.velocity_y()); });
    //     auto xc = X | nd::map([] (auto p) { return p[0]; });
    //     auto yc = X | nd::map([] (auto p) { return p[1]; });
    //     auto mu = d0 * nu | nd::midpoint_on_axis(0) | nd::select_axis(1).from(1).to(1).from_the_end();
    //     auto dx_ux = ux | nd::difference_on_axis(0) | nd::select_axis(1).from(1).to(1).from_the_end();
    //     auto dx_uy = uy | nd::difference_on_axis(0) | nd::select_axis(1).from(1).to(1).from_the_end();
    //     auto dy_ux = ux | nd::midpoint_on_axis(0) | nd::zip_adjacent3_on_axis(1) | nd::apply(centered_stencil);
    //     auto dy_uy = uy | nd::midpoint_on_axis(0) | nd::zip_adjacent3_on_axis(1) | nd::apply(centered_stencil);

    //     auto dx = xc | nd::difference_on_axis(0) | nd::select_axis(1).from(1).to(1).from_the_end();
    //     auto dy = yc | nd::midpoint_on_axis(0) | nd::zip_adjacent3_on_axis(1) | nd::apply(centered_stencil);

    //     auto tauxx = mu * (dx_ux / dx - dy_uy / dy);
    //     auto tauxy = mu * (dx_uy / dx + dy_ux / dy);

    //     return nd::zip(tauxx, tauxy) | nd::apply([] (auto txx, auto txy)
    //     {
    //         return mara::make_arithmetic_tuple(mara::make_dimensional<-1, 1, -1>(0.0), txx, txy);
    //     });
    // };


    // auto viscous_fluxes_tree_y_direction = [] (auto P, auto X)
    // {
    //     auto centered_stencil = [] (auto a, auto b, auto c) { return c - a; };
    //     auto nu = mara::make_dimensional<2, 0, -1>(0.01);

    //     auto d0 = P | nd::map([] (auto p) { return mara::make_dimensional<-2, 1, 0>(p.sigma()); });
    //     auto ux = P | nd::map([] (auto p) { return mara::make_velocity(p.velocity_x()); });
    //     auto uy = P | nd::map([] (auto p) { return mara::make_velocity(p.velocity_y()); });
    //     auto xc = X | nd::map([] (auto p) { return p[0]; });
    //     auto yc = X | nd::map([] (auto p) { return p[1]; });
    //     auto mu = d0 * nu | nd::midpoint_on_axis(1) | nd::select_axis(0).from(1).to(1).from_the_end();
    //     auto dy_ux = ux | nd::difference_on_axis(1) | nd::select_axis(0).from(1).to(1).from_the_end();
    //     auto dy_uy = uy | nd::difference_on_axis(1) | nd::select_axis(0).from(1).to(1).from_the_end();
    //     auto dx_ux = ux | nd::midpoint_on_axis(1) | nd::zip_adjacent3_on_axis(0) | nd::apply(centered_stencil);
    //     auto dx_uy = uy | nd::midpoint_on_axis(1) | nd::zip_adjacent3_on_axis(0) | nd::apply(centered_stencil);

    //     auto dx = xc | nd::midpoint_on_axis(1) | nd::zip_adjacent3_on_axis(0) | nd::apply(centered_stencil);
    //     auto dy = yc | nd::difference_on_axis(1) | nd::select_axis(0).from(1).to(1).from_the_end();

    //     auto tauyx =  mu * (dx_uy / dx + dy_ux / dy);
    //     auto tauyy = -mu * (dx_ux / dx - dy_uy / dy);

    //     return nd::zip(tauyx, tauyy) | nd::apply([] (auto tyx, auto tyy)
    //     {
    //         return mara::make_arithmetic_tuple(mara::make_dimensional<-1, 1, -1>(0.0), tyx, tyy);
    //     });
    // };


    // // Intermediate scheme data
    // //=========================================================================
    // auto v0  =  solver_data.vertices;
    // auto dA  =  solver_data.cell_areas;
    // auto xc  =  solver_data.cell_centers;
    // auto q0  =  solution.conserved;
    // auto p0  =  recover_primitive(solution.conserved, solver_data.cell_centers);



    // auto dx  =  v0.map([] (auto v) { return v | component<0>() | nd::difference_on_axis(0); });
    // auto dy  =  v0.map([] (auto v) { return v | component<1>() | nd::difference_on_axis(1); });
    // auto xf  =  v0.map([] (auto v) { return v | nd::midpoint_on_axis(1); });
    // auto yf  =  v0.map([] (auto v) { return v | nd::midpoint_on_axis(0); });
    // auto fx  =  extend(p0, 2, 0).map(extrapolate(0), tree_launch).pair(xf).apply(intercell_flux(0));
    // auto fy  =  extend(p0, 2, 1).map(extrapolate(1), tree_launch).pair(yf).apply(intercell_flux(1));

    // // auto p0_extended_for_viscous_fluxes = extend(extend(p0, 1, 0), 1, 1);
    // // auto xc_extended_for_viscous_fluxes = extend(extend(xc, 1, 0), 1, 1);
    // // auto visc_fx = p0_extended_for_viscous_fluxes.pair(xc_extended_for_viscous_fluxes).apply(viscous_fluxes_tree_x_direction);
    // // auto visc_fy = p0_extended_for_viscous_fluxes.pair(xc_extended_for_viscous_fluxes).apply(viscous_fluxes_tree_y_direction);

    // auto gx  =  (fx /* + visc_fx*/).pair(xf).apply(to_angmom_fluxes(0)) * dy;
    // auto gy  =  (fy /* + visc_fy*/).pair(yf).apply(to_angmom_fluxes(1)) * dx;
    // auto lx  = -gx.map(nd::difference_on_axis(0));
    // auto ly  = -gy.map(nd::difference_on_axis(1));
    // auto m0  =  q0.map(component<0>()) * dA; // cell masses


    // // Gravitational force, sink fields, and buffer zone source term
    // //=========================================================================
    // auto fg1 =        grav_vdot_field(solver_data, body1_pos, binary.body1.mass) * m0;
    // auto fg2 =        grav_vdot_field(solver_data, body2_pos, binary.body2.mass) * m0;
    // auto sg1 =   fg1.pair(xc).apply(force_to_source_terms_tree);
    // auto sg2 =   fg2.pair(xc).apply(force_to_source_terms_tree);
    // auto ss1 =  -q0 * sink_rate_field(solver_data, body1_pos, dt) * dA;
    // auto ss2 =  -q0 * sink_rate_field(solver_data, body2_pos, dt) * dA;
    // auto st  =   p0.pair(solver_data.cell_centers).apply(geometrical_source_terms_tree) * dA;
    // auto bz  =  (solver_data.initial_conserved - q0) * solver_data.buffer_rate_field * dA;

    // auto mask = solver_data.cell_centers.map([rd=solver_data.domain_radius] (auto XC)
    // {
    //     return XC | nd::map([rd] (auto xc) { return (xc * xc).sum() < rd * rd; });
    // });


    // // The updated conserved densities
    // //=========================================================================
    // auto q1 = q0 + (lx + ly + ss1 + ss2 + sg1 + sg2 + bz + st) * dt / dA;
    // auto next_conserved = q1.map(evaluate, tree_launch);

    // if (! safe_mode && (next_conserved.map(component<0>()) < 0.0).map(nd::any()).any())
    // {
    //     next_conserved
    //     .pair(solver_data.cell_centers)
    //     .sink([] (auto ux)
    //     {
    //         for (auto [u, x] : nd::zip(std::get<0>(ux), std::get<1>(ux)))
    //         {
    //             if (mara::get<0>(u) < 0.0)
    //             {
    //                 std::cout << "negative density " << mara::get<0>(u).value << " at " << mara::to_string(x) << std::endl;
    //             }
    //         }
    //     });

    //     std::cout << "binary::advance (negative density; re-trying in safe mode)" << std::endl;
    //     return advance(solution, solver_data, dt, true);
    // }


    // // The total force on each component, Mdot's, Ldot's, and Edot's.
    // //=========================================================================
    // auto mdot1 = -ss1.map(component<0>()).map(nd::sum(), tree_launch).sum();
    // auto mdot2 = -ss2.map(component<0>()).map(nd::sum(), tree_launch).sum();
    // auto kdot1 = -ss1.map(component<2>()).map(nd::sum(), tree_launch).sum();
    // auto kdot2 = -ss2.map(component<2>()).map(nd::sum(), tree_launch).sum();
    // auto ldot1 = -sg1.map(component<2>()).map(nd::sum(), tree_launch).sum();
    // auto ldot2 = -sg2.map(component<2>()).map(nd::sum(), tree_launch).sum();
    // auto m0_ejection_rate = -bz.map(component<0>()).map(nd::sum(), tree_launch).sum();
    // auto lz_ejection_rate = -bz.map(component<2>()).map(nd::sum(), tree_launch).sum();

    // auto Mdot = mara::make_sequence(mdot1, mdot2);
    // auto Kdot = mara::make_sequence(kdot1, kdot2);
    // auto Ldot = mara::make_sequence(ldot1, ldot2);


    // // The full updated solution state
    // //=========================================================================
    // return solution_t{
    //     solution.time + dt,
    //     solution.iteration + 1,
    //     next_conserved,

    //     solution.mass_accreted_on             + Mdot * dt,
    //     solution.angular_momentum_accreted_on + Kdot * dt,
    //     solution.integrated_torque_on         + Ldot * dt,
    //     solution.work_done_on                 , // + Edot * dt,
    //     solution.mass_ejected                 + m0_ejection_rate * dt,
    //     solution.angular_momentum_ejected     + lz_ejection_rate * dt,
    // };
// }




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
