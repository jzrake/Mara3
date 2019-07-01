#include "subprog_binary.hpp"
#include "core_ndarray_ops.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




//=============================================================================
binary::diagnostic_fields_t binary::diagnostic_fields(const solution_t& solution, const mara::config_t& run_config)
{
    auto solver_data = create_solver_data(run_config);
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);

    auto component = [] (std::size_t component)
    {
        return nd::map([component] (auto p) { return p[component]; });
    };

    auto area_from_vertices = [component] (auto vertices)
    {
        auto dx = vertices | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = vertices | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    };

    auto recover_primitive = std::bind(mara::iso2d::recover_primitive, std::placeholders::_1, 0.0);
    auto v0 = solver_data.vertices;
    auto c0 = v0.map(nd::midpoint_on_axis(0)).map(nd::midpoint_on_axis(1));
    auto xc = c0.map(component(0));
    auto yc = c0.map(component(1));
    auto u0 = solution.conserved;
    auto p0 = u0.map(nd::map(recover_primitive)).map(nd::to_shared());
    auto dA = v0.map(area_from_vertices).map(nd::to_shared());

    auto rc = (xc * xc + yc * yc).map(nd::map([] (mara::unit_area<double> r2) { return r2.pow<1, 2>(); }));
    auto rhat_x =  xc / rc;
    auto rhat_y =  yc / rc;
    auto phat_x = -yc / rc;
    auto phat_y =  xc / rc;
    auto sigma = p0.map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::sigma)));
    auto vx    = p0.map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_x)));
    auto vy    = p0.map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_y)));
    auto vr    = vx * rhat_x + vy * rhat_y;
    auto vp    = vx * phat_x + vy * phat_y;

    return diagnostic_fields_t{
        run_config,
        solution.time,
        v0,
        sigma.map(nd::to_shared()),
        vr   .map(nd::to_shared()),
        vp   .map(nd::to_shared()),
        {binary.body1.position_x, binary.body1.position_y},
        {binary.body2.position_x, binary.body2.position_y},
    };
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
