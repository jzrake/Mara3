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




//=============================================================================
namespace binary
{
    struct source_term_total_t
    {
        mara::arithmetic_sequence_t<mara::unit_mass  <double>, 2> mass_accreted_on = {};
        mara::arithmetic_sequence_t<mara::unit_angmom<double>, 2> angular_momentum_accreted_on = {};
        mara::arithmetic_sequence_t<mara::unit_angmom<double>, 2> integrated_torque_on = {};
        mara::arithmetic_sequence_t<mara::unit_energy<double>, 2> work_done_on = {};
        mara::unit_mass  <double>                                 mass_ejected = {};
        mara::unit_angmom<double>                                 angular_momentum_ejected = {};
        source_term_total_t operator+(const source_term_total_t& other) const;
    };
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
binary::source_term_total_t binary::source_term_total_t::operator+(const source_term_total_t& other) const
{
    return {
        mass_accreted_on + other.mass_accreted_on,
        angular_momentum_accreted_on + other.angular_momentum_accreted_on,
        integrated_torque_on + other.integrated_torque_on,
        work_done_on + other.work_done_on,
        mass_ejected + other.mass_ejected,
        angular_momentum_ejected + other.angular_momentum_ejected,
    };
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




//=============================================================================
template<typename TreeType>
static auto extend(TreeType tree, std::size_t axis, std::size_t guard_count)
{
    return tree.indexes().map([tree, axis, guard_count] (auto index)
    {
        auto C = tree.at(index);
        auto L = mara::get_cell_block(tree, index.prev_on(axis), mara::compose(nd::to_shared(), nd::select_final(guard_count, axis)));
        auto R = mara::get_cell_block(tree, index.next_on(axis), mara::compose(nd::to_shared(), nd::select_first(guard_count, axis)));
        return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis) | nd::to_shared();
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
static mara::iso2d::flux_t viscous_flux(std::size_t axis,
    prim_pair_t g_long,
    prim_pair_t g_tran,
    mara::unit_length<double> dx,
    mara::unit_length<double> dy,
    double mu)
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

            auto tauxx = mu * (dx_ux / dx.value - dy_uy / dy.value);
            auto tauxy = mu * (dx_uy / dx.value + dy_ux / dy.value);

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

            auto tauyx =  mu * (dx_uy / dx.value + dy_ux / dy.value);
            auto tauyy = -mu * (dx_ux / dx.value - dy_uy / dy.value);

            return mara::make_arithmetic_tuple(
                mara::make_dimensional<-1, 1, -1>(0.0),
                mara::make_dimensional< 0, 1,-2>(tauyx),
                mara::make_dimensional< 0, 1,-2>(tauyy));
        }
    }
    throw;
}




//=============================================================================
static auto intercell_flux(std::size_t axis, const binary::solver_data_t& solver_data, mara::tree_index_t<2> tree_index)
{
    auto to_angmom = to_angmom_fluxes(axis, solver_data.domain_radius);
    auto dx = (solver_data.vertices.at(tree_index) | nd::difference_on_axis(0) | nd::read_index(0, 0))[0];
    auto dy = (solver_data.vertices.at(tree_index) | nd::difference_on_axis(1) | nd::read_index(0, 0))[1];

    return [axis, solver_data, to_angmom, dx, dy] (binary::location_2d_t xf, prim_pair_t p0, prim_pair_t g_long, prim_pair_t g_tran)
    {
        auto [pl, pr] = p0;
        auto [gl, gr] = g_long;

        auto pl_hat = pl + gl * 0.5;
        auto pr_hat = pr - gr * 0.5;
        auto cs2 = cs2_at_position(xf, solver_data.mach_number);
        auto nu = solver_data.alpha * std::sqrt(cs2) * scale_height_at_position(xf, solver_data.mach_number);
        auto mu = 0.5 * nu * (pl_hat.sigma() + pr_hat.sigma());

        auto nhat = mara::unit_vector_t::on_axis(axis);
        auto fhat = mara::iso2d::riemann_hlle(pl_hat, pr_hat, cs2, cs2, nhat);
        auto fhat_visc = viscous_flux(axis, g_long, g_tran, dx, dy, mu);

        return to_angmom(xf, fhat + fhat_visc);
    };
}




//=============================================================================
static auto force_to_source_terms(binary::location_2d_t x, force_per_area_t f)
{
    auto sigma_dot = mara::make_dimensional<-2, 1, -1>(0.0);
    auto sr_dot    = x[0] * f[0] + x[1] * f[1];
    auto lz_dot    = x[0] * f[1] - x[1] * f[0];
    return mara::make_arithmetic_tuple(sigma_dot, sr_dot, lz_dot);
};




//=============================================================================
static auto source_terms = [] (auto solver_data, auto solution, auto p0, auto tree_index, auto dt)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);
    auto body1_pos = binary::location_2d_t{binary.body1.position_x, binary.body1.position_y};
    auto body2_pos = binary::location_2d_t{binary.body2.position_x, binary.body2.position_y};

    auto sr2 = solver_data.gst_suppr_radius.template pow<2>();
    auto M   = solver_data.mach_number;
    auto xc  = solver_data.cell_centers.at(tree_index);
    auto dA  = solver_data.cell_areas.at(tree_index);
    auto br  = solver_data.buffer_rate_field.at(tree_index);
    auto q0  = solution.conserved.at(tree_index);

    auto sigma = q0 | component<0>();
    auto fg1 = (xc | nd::map(grav_vdot_field(solver_data, body1_pos, binary.body1.mass))) * sigma;
    auto fg2 = (xc | nd::map(grav_vdot_field(solver_data, body2_pos, binary.body2.mass))) * sigma;

    auto s_grav_1 = (nd::zip(xc, fg1) | nd::apply(force_to_source_terms)) * dt;
    auto s_grav_2 = (nd::zip(xc, fg2) | nd::apply(force_to_source_terms)) * dt;
    auto s_sink_1 = -q0 * (xc | nd::map(sink_rate_field(solver_data, body1_pos))) * dt;
    auto s_sink_2 = -q0 * (xc | nd::map(sink_rate_field(solver_data, body2_pos))) * dt;
    auto s_buffer = (solver_data.initial_conserved.at(tree_index) - q0) * br * dt;
    auto s_geom = nd::zip(p0.at(tree_index), xc) | nd::apply([sr2, M, dt] (auto p, auto x)
    {
        auto ramp = 1.0 - std::exp(-(x * x).sum() / sr2);
        auto cs2 = cs2_at_position(x, M);
        return p.source_terms_conserved_angmom(cs2) * ramp * dt;
    });

    auto totals = binary::source_term_total_t();
    totals.mass_accreted_on[0]             = -(s_sink_1 | component<0>() | nd::multiply(dA) | nd::sum());
    totals.mass_accreted_on[1]             = -(s_sink_2 | component<0>() | nd::multiply(dA) | nd::sum());
    totals.angular_momentum_accreted_on[0] = -(s_sink_1 | component<2>() | nd::multiply(dA) | nd::sum());
    totals.angular_momentum_accreted_on[1] = -(s_sink_2 | component<2>() | nd::multiply(dA) | nd::sum());
    totals.integrated_torque_on[0]         = -(s_grav_1 | component<2>() | nd::multiply(dA) | nd::sum());
    totals.integrated_torque_on[1]         = -(s_grav_2 | component<2>() | nd::multiply(dA) | nd::sum());
    totals.mass_ejected                    = -(s_buffer | component<0>() | nd::multiply(dA) | nd::sum());
    totals.angular_momentum_ejected        = -(s_buffer | component<2>() | nd::multiply(dA) | nd::sum());

    return std::make_pair(s_grav_1 + s_grav_2 + s_sink_1 + s_sink_2 + s_buffer + s_geom | nd::to_shared(), totals);
};




//=============================================================================
static auto block_update = [] (
    auto solver_data,
    auto solution,
    auto p0,
    auto p0_ex,
    auto p0_ey,
    auto gx_ex,
    auto gx_ey,
    auto gy_ex,
    auto gy_ey,
    auto dt)
{
    return [=] (auto tree_index)
    {
        auto q0 = solution.conserved.at(tree_index);
        auto xv = solver_data.vertices.at(tree_index);
        auto xc = solver_data.cell_centers.at(tree_index);
        auto dA = solver_data.cell_areas.at(tree_index);
        auto dx = xv | component<0>() | nd::difference_on_axis(0);
        auto dy = xv | component<1>() | nd::difference_on_axis(1);
        auto xf = xv | nd::midpoint_on_axis(1);
        auto yf = xv | nd::midpoint_on_axis(0);

        auto fhat_x = nd::zip(
            xf,
            p0_ex.at(tree_index) | nd::zip_adjacent2_on_axis(0),
            gx_ex.at(tree_index) | nd::zip_adjacent2_on_axis(0),
            gy_ex.at(tree_index) | nd::zip_adjacent2_on_axis(0))
        | nd::apply(intercell_flux(0, solver_data, tree_index))
        | nd::multiply(dy)
        | nd::to_shared();

        auto fhat_y = nd::zip(
            yf,
            p0_ey.at(tree_index) | nd::zip_adjacent2_on_axis(1),
            gy_ey.at(tree_index) | nd::zip_adjacent2_on_axis(1),
            gx_ey.at(tree_index) | nd::zip_adjacent2_on_axis(1))
        | nd::apply(intercell_flux(1, solver_data, tree_index))
        | nd::multiply(dx)
        | nd::to_shared();

        auto lx = fhat_x | nd::difference_on_axis(0);
        auto ly = fhat_y | nd::difference_on_axis(1);

        auto s = source_terms(solver_data, solution, p0, tree_index, dt);
        return std::make_pair(q0 - (lx + ly) * dt / dA + s.first | nd::to_shared(), s.second);
    };
};




//=============================================================================
binary::solution_t binary::advance(const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt, bool safe_mode)
{

    // Compute pre-requisite data
    //=========================================================================
    auto th = solver_data.plm_theta;
    auto q0 = solution.conserved;
    auto p0 = q0.pair(solver_data.cell_centers).apply([] (auto Q, auto X)
    {
        return nd::zip(Q, X)
        | nd::apply([] (auto q, auto x) { return mara::iso2d::recover_primitive(q, x); })
        | nd::to_shared();
    });


    auto p0_ex = extend(p0, 0, 1);
    auto p0_ey = extend(p0, 1, 1);
    auto gx = p0_ex.map([th] (auto p) { return estimate_gradient(p, 0, th) | nd::to_shared(); });
    auto gy = p0_ey.map([th] (auto p) { return estimate_gradient(p, 1, th) | nd::to_shared(); });
    auto gx_ex = extend(gx, 0, 1);
    auto gx_ey = extend(gx, 1, 1);
    auto gy_ex = extend(gy, 0, 1);
    auto gy_ey = extend(gy, 1, 1);
    auto block_results = p0
    .indexes()
    .map(block_update(solver_data, solution, p0, p0_ex, p0_ey, gx_ex, gx_ey, gy_ex, gy_ey, dt));

    auto q1     = block_results.map([] (const auto& t) { return t.first; });
    auto totals = block_results.map([] (const auto& t) { return t.second; }).sum();


    // The full updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        q1,
        solution.mass_accreted_on             + totals.mass_accreted_on,
        solution.angular_momentum_accreted_on + totals.angular_momentum_accreted_on,
        solution.integrated_torque_on         + totals.integrated_torque_on,
        solution.work_done_on,
        solution.mass_ejected                 + totals.mass_ejected,
        solution.angular_momentum_ejected     + totals.angular_momentum_ejected,
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
