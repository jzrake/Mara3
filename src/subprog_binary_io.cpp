#include "subprog_binary.hpp"
#include "app_serialize_tree.hpp"
#if MARA_COMPILE_SUBPROGRAM_BINARY




//=========================================================================
template<>
void mara::write<binary::solution_t>(h5::Group& group, std::string name, const binary::solution_t& solution)
{
    auto location = group.require_group(name);
    mara::write(location, "time",       solution.time);
    mara::write(location, "iteration",  solution.iteration);
    mara::write(location, "conserved",  solution.conserved);
}

template<>
void mara::write<binary::state_t>(h5::Group& group, std::string name, const binary::state_t& state)
{
    auto location = group.require_group(name);
    mara::write(location, "solution",   state.solution);
    mara::write(location, "schedule",   state.schedule);
    mara::write(location, "run_config", state.run_config);
}

template<>
void mara::write<binary::diagnostic_fields_t>(h5::Group& group, std::string name, const binary::diagnostic_fields_t& diagnostics)
{
    auto location = group.require_group(name);
    mara::write(location, "run_config",        diagnostics.run_config);
    mara::write(location, "time",              diagnostics.time);
    mara::write(location, "vertices",          diagnostics.vertices);
    mara::write(location, "sigma",             diagnostics.sigma);
    mara::write(location, "radial_velocity",   diagnostics.radial_velocity);
    mara::write(location, "phi_velocity",      diagnostics.phi_velocity);
    mara::write(location, "position_of_mass1", diagnostics.position_of_mass1);
    mara::write(location, "position_of_mass2", diagnostics.position_of_mass2);
}

template<>
void mara::read<binary::solution_t>(h5::Group& group, std::string name, binary::solution_t& solution)
{
    auto location = group.open_group(name);
    mara::read(location, "time",       solution.time);
    mara::read(location, "iteration",  solution.iteration);
    mara::read(location, "conserved",  solution.conserved);
}

template<>
void mara::read<binary::state_t>(h5::Group& group, std::string name, binary::state_t& state)
{
    auto location = group.open_group(name);
    mara::read(location, "solution",   state.solution);
    mara::read(location, "schedule",   state.schedule);
}

template<>
void mara::read<binary::diagnostic_fields_t>(h5::Group& group, std::string name, binary::diagnostic_fields_t& diagnostics)
{
    auto location = group.open_group(name);
    // mara::read(location, "run_config",        diagnostics.run_config);
    mara::read(location, "time",              diagnostics.time);
    mara::read(location, "vertices",          diagnostics.vertices);
    mara::read(location, "sigma",             diagnostics.sigma);
    mara::read(location, "radial_velocity",   diagnostics.radial_velocity);
    mara::read(location, "phi_velocity",      diagnostics.phi_velocity);
    mara::read(location, "position_of_mass1", diagnostics.position_of_mass1);
    mara::read(location, "position_of_mass2", diagnostics.position_of_mass2);
}

#endif // MARA_COMPILE_SUBPROGRAM_BINARY
