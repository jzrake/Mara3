#include "subprog_binary.hpp"
#include "core_ndarray_ops.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




//=============================================================================
static auto component(std::size_t component)
{
    return nd::map([component] (auto p) { return p[component]; });
};




//=============================================================================
binary::solver_data_t binary::create_solver_data(const mara::config_t& run_config)
{
    auto vertices = create_vertices(run_config);

    auto cell_centers = vertices.map([] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1);
    });

    auto cell_areas = vertices.map([] (auto block)
    {
        auto dx = block | component(0) | nd::difference_on_axis(0) | nd::midpoint_on_axis(1);
        auto dy = block | component(1) | nd::difference_on_axis(1) | nd::midpoint_on_axis(0);
        return dx * dy;
    });

    auto primitive = cell_centers.map([&run_config] (auto block)
    {
        return block | nd::map(create_disk_profile(run_config));
    });

    auto min_dx = vertices.map([] (auto block)
    {
        return block
        | nd::map([] (auto p) { return p[0]; })
        | nd::difference_on_axis(0)
        | nd::min();
    }).min();

    auto min_dy = vertices.map([] (auto block)
    {
        return block
        | nd::map([] (auto p) { return p[1]; })
        | nd::difference_on_axis(1)
        | nd::min();
    }).min();

    auto max_velocity = std::max(mara::make_velocity(1.0), primitive.map([] (auto block)
    {
        return block
        | nd::map(std::mem_fn(&mara::iso2d::primitive_t::velocity_magnitude))
        | nd::max();
    }).max());

    auto buffer_rate_field = vertices.map([&run_config] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map([&run_config] (location_2d_t p)
        {
            auto tightness   = mara::dimensional_value_t<-1, 0, 0, double>(3.0);
            auto buffer_rate = mara::dimensional_value_t< 0, 0,-1, double>(run_config.get_double("buffer_damping_rate"));
            auto r1 = mara::make_length(run_config.get_double("domain_radius"));
            auto rc = (p[0] * p[0] + p[1] * p[1]).pow<1, 2>();
            auto y = (tightness * (rc - r1)).scalar();
            return buffer_rate * (1.0 + std::tanh(y));
        });
    });


    //=========================================================================
    auto result = solver_data_t();
    result.domain_radius         = run_config.get_double("domain_radius");
    result.mach_number           = run_config.get_double("mach_number");
    result.alpha_cutoff_radius   = run_config.get_double("alpha_cutoff_radius");
    result.alpha                 = run_config.get_double("alpha");
    result.nu                    = run_config.get_double("nu");
    result.sink_rate             = run_config.get_double("sink_rate");
    result.sink_radius           = run_config.get_double("sink_radius");
    result.softening_radius      = run_config.get_double("softening_radius");
    result.gst_suppr_radius      = run_config.get_double("source_term_softening") * std::min(min_dx, min_dy).value;
    result.plm_theta             = run_config.get_double("plm_theta");
    result.begin_live_binary     = run_config.get_double("begin_live_binary");
    result.axisymmetric_cs2      = run_config.get_int("axisymmetric_cs2");
    result.conserve_linear_p     = run_config.get_int("conserve_linear_p");
    result.fixed_dt              = run_config.get_int("fixed_dt");
    result.rk_order              = run_config.get_int("rk_order");
    result.block_size            = run_config.get_int("block_size");
    result.no_accretion_force    = run_config.get_int("no_accretion_force");
    result.density_floor         = run_config.get_double("density_floor") * run_config.get_double("disk_mass");
    result.cfl_number            = run_config.get_double("cfl_number");
    result.recommended_time_step = std::min(min_dx, min_dy) / max_velocity * run_config.get_double("cfl_number");
    result.buffer_rate_field     = buffer_rate_field.map(nd::to_shared());
    result.cell_centers          = cell_centers.map(nd::to_shared());
    result.cell_areas            = cell_areas.map(nd::to_shared());
    result.vertices              = vertices;
    result.initial_conserved_u   = create_solution(run_config).conserved_u;
    result.initial_conserved_q   = create_solution(run_config).conserved_q;
    result.riemann_solver        = riemann_solver_t::hlle;
    if      (run_config.get_string("reconstruct_method") == "pcm") result.reconstruct_method = reconstruct_method_t::pcm;
    else if (run_config.get_string("reconstruct_method") == "plm") result.reconstruct_method = reconstruct_method_t::plm;
    else throw std::invalid_argument("invalid reconstruct_method '" + run_config.get_string("reconstruct_method") + "', must be plm or pcm");

    return result;
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
