#include "subprog_binary.hpp"
#include "mesh_prolong_restrict.hpp"
#include "mesh_tree_operators.hpp"
#include "core_ndarray_ops.hpp"




//=============================================================================
namespace binary
{
    auto gravitational_acceleration_field(mara::unit_time<double> time, const solver_data_t& solver_data);
    auto sink_rate_field(mara::unit_time<double> time, const solver_data_t& solver_data);    
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

auto binary::gravitational_acceleration_field(mara::unit_time<double> time, const solver_data_t& solver_data)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, time.value);
    auto accel = [softening_radius=solver_data.softening_radius] (const mara::point_mass_t& body, location_2d_t field_point)
    {
        auto mass_location = location_2d_t { body.position_x, body.position_y };

        auto G   = mara::dimensional_value_t<3, -1, -2, double>(1.0);
        auto M   = mara::make_mass(body.mass);
        auto dr  = field_point - mass_location;
        auto dr2 = dr[0] * dr[0] + dr[1] * dr[1];
        auto rs2 = softening_radius * softening_radius;
        return -dr / (dr2 + rs2).pow<3, 2>() * G * M;
    };

    auto acceleration = [binary, accel] (location_2d_t p)
    {
        return accel(binary.body1, p) + accel(binary.body2, p);
    };

    return solver_data.vertices.map([acceleration] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(acceleration);
    });
}

auto binary::sink_rate_field(mara::unit_time<double> time, const solver_data_t& solver_data)
{
    auto binary = mara::compute_two_body_state(solver_data.binary_params, time.value);
    auto sink = [binary, sink_radius=solver_data.sink_radius, sink_rate=solver_data.sink_rate] (location_2d_t p)
    {
        auto dx1 = p[0] - mara::make_length(binary.body1.position_x);
        auto dy1 = p[1] - mara::make_length(binary.body1.position_y);
        auto dx2 = p[0] - mara::make_length(binary.body2.position_x);
        auto dy2 = p[1] - mara::make_length(binary.body2.position_y);

        auto s2 = sink_radius * sink_radius;
        auto a2 = (dx1 * dx1 + dy1 * dy1) / s2 / 2.0;
        auto b2 = (dx2 * dx2 + dy2 * dy2) / s2 / 2.0;

        return sink_rate * 0.5 * (std::exp(-a2) + std::exp(-b2));
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
    auto component = [] (std::size_t component)
    {
        return nd::map([component] (auto p) { return p[component]; });
    };

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

    auto area_from_vertices = [component] (auto vertices)
    {
        auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };

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

    auto force_to_source_terms = [] (force_2d_t v)
    {
        return mara::iso2d::flow_t{0.0, v[0].value, v[1].value};
    };

    auto evaluate = nd::to_shared();
    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);
    auto v0 = solver_data.vertices;
    auto u0 = solution.conserved;
    auto p0 = u0.map(nd::map(recover_primitive)).map(evaluate);
    auto dA = v0.map(area_from_vertices).map(evaluate);
    auto dx = v0.map([component] (auto v) { return v | component(0) | nd::difference_on_axis(0); });
    auto dy = v0.map([component] (auto v) { return v | component(1) | nd::difference_on_axis(1); });
    auto fx = extend(p0, 0).map(extrapolate(0)).map(intercell_flux(mara::iso2d::riemann_hlle, 0)) * dy;
    auto fy = extend(p0, 1).map(extrapolate(1)).map(intercell_flux(mara::iso2d::riemann_hlle, 1)) * dx;
    auto lx = -fx.map(nd::difference_on_axis(0));
    auto ly = -fy.map(nd::difference_on_axis(1));
    auto m0 = u0.map(component(0)) * dA; // cell masses
    auto ag = gravitational_acceleration_field(solution.time, solver_data);
    auto sg = (ag * m0).map(nd::map(force_to_source_terms));
    auto ss = -u0 * sink_rate_field(solution.time, solver_data) * dA;
    auto bz = (solver_data.initial_conserved - u0) * solver_data.buffer_rate_field * dA;
    auto u1 = u0 + (lx + ly + ss + sg + bz) * dt / dA;

    return solution_t{
        solution.time + dt,
        solution.iteration + 1,
        u1.map(evaluate),
    };
}




//=============================================================================
binary::solution_t binary::solution_t::operator+(const solution_t& other) const
{
    return {
        time       + other.time,
        iteration  + other.iteration,
        (conserved + other.conserved).map(nd::to_shared()),
    };
}

binary::solution_t binary::solution_t::operator*(mara::rational_number_t scale) const
{
    return {
        time       * scale.as_double(),
        iteration  * scale,
        (conserved * scale.as_double()).map(nd::to_shared()),
    };
}
