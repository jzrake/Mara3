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

    return solver_data.conserve_linear_p
    ? (solution.conserved_u.map(component<0>()) * dA).map(nd::sum()).sum()
    : (solution.conserved_q.map(component<0>()) * dA).map(nd::sum()).sum();
}

mara::unit_angmom<double> binary::disk_angular_momentum(const solution_t& solution, const solver_data_t& solver_data)
{
    auto dA = solver_data.cell_areas;

    if (solver_data.conserve_linear_p)
    {
        auto lz = solution.conserved_u
        .pair(solver_data.cell_centers)
        .apply([] (auto U, auto X) { return nd::zip(U, X) | nd::apply(mara::iso2d::angular_momentum); });

        return (lz * dA).map(nd::sum()).sum();
    }
    return (solution.conserved_q * dA).map(component<2>()).map(nd::sum()).sum();
}




//=============================================================================
binary::diagnostic_fields_t binary::diagnostic_fields(const solution_t& solution, const mara::config_t& run_config)
{
    auto solver_data = create_solver_data(run_config);
    auto binary = mara::compute_two_body_state(solution.orbital_elements, solution.time.value);

    auto v0 = solver_data.vertices;
    auto c0 = solver_data.cell_centers;
    auto dA = solver_data.cell_areas;
    auto xc = c0.map(component<0>());
    auto yc = c0.map(component<1>());
    auto q0 = solution.conserved_q;
    auto p0 = recover_primitive(solution, solver_data);

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
