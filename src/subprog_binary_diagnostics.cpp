#include "subprog_binary.hpp"
#include "core_ndarray_ops.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




//=============================================================================
template<std::size_t I>
static auto component()
{
    return nd::map([] (auto p) { return mara::get<I>(p); });
};




//=============================================================================
mara::unit_mass<double> binary::disk_mass(const solution_t& solution, const solver_data_t& solver_data)
{
    auto v0 = solver_data.vertices;
    auto dA = solver_data.cell_areas;
    auto u0 = solution.conserved;
    auto sigma = u0.map(component<0>());
    return (sigma * dA).map(nd::sum()).sum();
}

mara::unit_angmom<double> binary::disk_angular_momentum(const solution_t& solution, const solver_data_t& solver_data)
{
    auto v0 = solver_data.vertices;
    auto dA = solver_data.cell_areas;
    auto u0 = solution.conserved;
    auto c0 = v0.map(nd::midpoint_on_axis(0)).map(nd::midpoint_on_axis(1));
    auto xc = c0.map(component<0>());
    auto yc = c0.map(component<1>());
    auto px = u0.map(component<1>());
    auto py = u0.map(component<2>());
    auto Lz = xc * py - yc * px;
    return (Lz * dA).map(nd::sum()).sum();
}




//=============================================================================
binary::diagnostic_fields_t binary::diagnostic_fields(const solution_t& solution, const mara::config_t& run_config)
{
    auto solver_data = create_solver_data(run_config);
    auto binary = mara::compute_two_body_state(solver_data.binary_params, solution.time.value);

    auto recover_primitive = [] (auto&& u) { return mara::iso2d::recover_primitive(u); };
    auto v0 = solver_data.vertices;
    auto c0 = solver_data.cell_centers;
    auto xc = c0.map(component<0>());
    auto yc = c0.map(component<1>());
    auto u0 = solution.conserved;
    auto p0 = u0.map(nd::map(recover_primitive)).map(nd::to_shared());
    auto dA = solver_data.cell_areas;

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
