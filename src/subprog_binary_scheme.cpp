#include "subprog_binary.hpp"
#include "mesh_prolong_restrict.hpp"
#include "mesh_tree_operators.hpp"
#include "core_ndarray_ops.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




//=============================================================================
namespace binary
{
    auto grav_vdot_field(const solver_data_t& solver_data, location_2d_t body_location, mara::unit_mass<double> body_mass);
    auto sink_rate_field(const solver_data_t& solver_data, location_2d_t sink_location);
    auto estimate_gradient_plm(double plm_theta);
}




//=============================================================================
auto binary::estimate_gradient_plm(double plm_theta)
{
    return [plm_theta] (
        const mara::iso2d::primitive_t& pl,
        const mara::iso2d::primitive_t& p0,
        const mara::iso2d::primitive_t& pr)
    {
        using std::min;
        using std::fabs;
        auto min3abs = [] (auto a, auto b, auto c) { return min(min(fabs(a), fabs(b)), fabs(c)); };
        auto sgn = [] (auto x) { return std::copysign(1, x); };

        double th = plm_theta;
        auto result = mara::iso2d::primitive_t();

        for (std::size_t i = 0; i < 3; ++i)
        {
            auto a =  th * (p0[i] - pl[i]);
            auto b = 0.5 * (pr[i] - pl[i]);
            auto c =  th * (pr[i] - p0[i]);
            result[i] = 0.25 * std::fabs(sgn(a) + sgn(b)) * (sgn(a) + sgn(c)) * min3abs(a, b, c);
        }
        return result;
    };
}

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
binary::solution_t binary::advance(const solution_t& solution, const solver_data_t& solver_data, mara::unit_time<double> dt)
{


    /*
     * @brief      An operator on arrays of sequences: takes a single component
     *             of a sequence and returns an array.
     *
     * @param[in]  component  The component to take
     *
     * @return     An array whose value type is the sequence value type
     */
    auto component = [] (std::size_t component)
    {
        return nd::map([component] (auto p) { return p[component]; });
    };


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
            auto C = mara::get_cell_block(tree, index);
            auto L = mara::get_cell_block(tree, index.prev_on(axis)) | nd::select_final(2, axis);
            auto R = mara::get_cell_block(tree, index.next_on(axis)) | nd::select_first(2, axis);
            return L | nd::concat(C).on_axis(axis) | nd::concat(R).on_axis(axis);
        });
    };


    /*
     * @brief      Return an array of areas dx * dy from the given vertex
     *             locations.
     *
     * @param[in]  vertices  An array of vertices
     *
     * @return     A new array
     */
    auto area_from_vertices = [component] (auto vertices)
    {
        auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };


    /*
     * @brief      Apply a piecewise linear extrapolation, along the given axis,
     *             to the cells in each array of a tree.
     *
     * @param[in]  axis  The axis to extrapolate on
     *
     * @return     A function that operates on trees
     */
    auto extrapolate = [plm_theta=solver_data.plm_theta] (std::size_t axis)
    {
        return [plm_theta, axis] (auto P)
        {
            auto L1 = nd::select_axis(axis).from(0).to(1).from_the_end();
            auto R1 = nd::select_axis(axis).from(1).to(0).from_the_end();
            auto L2 = nd::select_axis(axis).from(1).to(2).from_the_end();
            auto R2 = nd::select_axis(axis).from(2).to(1).from_the_end();

            auto G = P
            | nd::zip_adjacent3_on_axis(axis)
            | nd::apply(estimate_gradient_plm(plm_theta))
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
    auto force_to_source_terms = [] (force_2d_t v) { return mara::iso2d::flow_t{0.0, v[0].value, v[1].value}; };
    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);
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
    auto u0  =  solution.conserved;
    auto p0  =  u0.map(nd::map(recover_primitive)).map(evaluate);
    auto dA  =  v0.map(area_from_vertices).map(evaluate);
    auto dx  =  v0.map([component] (auto v) { return v | component(0) | nd::difference_on_axis(0); });
    auto dy  =  v0.map([component] (auto v) { return v | component(1) | nd::difference_on_axis(1); });
    auto fx  =  extend(p0, 0).map(extrapolate(0)).map(intercell_flux(mara::iso2d::riemann_hlle, 0)) * dy;
    auto fy  =  extend(p0, 1).map(extrapolate(1)).map(intercell_flux(mara::iso2d::riemann_hlle, 1)) * dx;
    auto lx  = -fx.map(nd::difference_on_axis(0));
    auto ly  = -fy.map(nd::difference_on_axis(1));
    auto m0  =  u0.map(component(0)) * dA; // cell masses


    // Gravitational force, sink fields, and buffer zone source term
    //=========================================================================
    auto fg1 =       grav_vdot_field(solver_data, body1_pos, binary.body1.mass) * m0;
    auto fg2 =       grav_vdot_field(solver_data, body2_pos, binary.body2.mass) * m0;
    auto ss1 = -u0 * sink_rate_field(solver_data, body1_pos) * dA;
    auto ss2 = -u0 * sink_rate_field(solver_data, body2_pos) * dA;
    auto sg  =  (fg1 + fg2).map(nd::map(force_to_source_terms));
    auto ss  =  (ss1 + ss2);
    auto bz  =  (solver_data.initial_conserved - u0) * solver_data.buffer_rate_field * dA;


    // The updated conserved densities
    //=========================================================================
    auto u1  = u0 + (lx + ly + ss + sg + bz) * dt / dA;


    // The total force on each component, Mdot's, Ldot's, and Edot's.
    //=========================================================================
    auto fg1_tot = -fg1.map(nd::sum()).sum();
    auto fg2_tot = -fg2.map(nd::sum()).sum();
    auto ss1_tot = -ss1.map(component(0)).map(nd::sum()).sum();
    auto ss2_tot = -ss2.map(component(0)).map(nd::sum()).sum();
    auto Mdot = mara::make_sequence(ss1_tot, ss2_tot);
    auto Ldot = mara::make_sequence(cross_prod_z(fg1_tot, body1_pos), cross_prod_z(fg2_tot, body2_pos));
    auto Edot = mara::make_sequence((fg1_tot * body1_vel).sum(), (fg2_tot * body2_vel).sum());


    // The full updated solution state
    //=========================================================================
    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        u1.map(evaluate),
        solution.mass_accreted_on     + Mdot * dt,
        solution.integrated_torque_on + Ldot * dt,
        solution.work_done_on         + Edot * dt,
    };
}




//=============================================================================
binary::solution_t binary::solution_t::operator+(const solution_t& other) const
{
    return {
        time       + other.time,
        iteration  + other.iteration,
        (conserved + other.conserved).map(nd::to_shared()),
        mass_accreted_on     + other.mass_accreted_on,
        integrated_torque_on + other.integrated_torque_on,
        work_done_on         + other.work_done_on,
    };
}

binary::solution_t binary::solution_t::operator*(mara::rational_number_t scale) const
{
    return {
        time       * scale.as_double(),
        iteration  * scale,
        (conserved * scale.as_double()).map(nd::to_shared()),
        mass_accreted_on     * scale.as_double(),
        integrated_torque_on * scale.as_double(),
        work_done_on         * scale.as_double(),
    };
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
