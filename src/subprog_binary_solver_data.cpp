#include "subprog_binary.hpp"
#include "core_ndarray_ops.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




//=============================================================================
binary::solver_data_t binary::create_solver_data(const mara::config_t& run_config)
{
    auto vertices = create_vertices(run_config);

    auto primitive = vertices.map([&run_config] (auto block)
    {
        return block
        | nd::midpoint_on_axis(0)
        | nd::midpoint_on_axis(1)
        | nd::map(create_disk_profile(run_config));
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
        })
        | nd::to_shared();
    });


    //=========================================================================
    auto result = solver_data_t();
    result.mach_number           = run_config.get_double("mach_number");
    result.sink_rate             = run_config.get_double("sink_rate");
    result.sink_radius           = run_config.get_double("sink_radius");
    result.softening_radius      = run_config.get_double("softening_radius");
    result.plm_theta             = run_config.get_double("plm_theta");
    result.rk_order              = run_config.get_int("rk_order");
    result.recommended_time_step = std::min(min_dx, min_dy) / max_velocity * run_config.get_double("cfl_number");
    result.binary_params         = create_binary_params(run_config);
    result.buffer_rate_field     = buffer_rate_field;
    result.vertices              = vertices;
    result.initial_conserved     = primitive
    .map(nd::map(std::mem_fn(&mara::iso2d::primitive_t::to_conserved_per_area)))
    .map(nd::to_shared());

    if      (run_config.get_string("riemann") == "hlle") result.riemann_solver = riemann_solver_t::hlle;
    // else if (run_config.get_string("riemann") == "hllc") result.riemann_solver = riemann_solver_t::hllc;
    else throw std::invalid_argument("invalid riemann solver '" + run_config.get_string("riemann") + "', must be hlle");

    if      (run_config.get_string("reconstruct_method") == "pcm") result.reconstruct_method = reconstruct_method_t::pcm;
    else if (run_config.get_string("reconstruct_method") == "plm") result.reconstruct_method = reconstruct_method_t::plm;
    else throw std::invalid_argument("invalid reconstruct_method '" + run_config.get_string("reconstruct_method") + "', must be plm or pcm");

    return result;
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
