#pragma once
#include <cmath>
#include "core_geometric.hpp"




//=============================================================================
namespace mara::srhd
{
    using conserved_density_t = covariant_sequence_t<unit_mass_density<double>, 5>;
    using conserved_t         = covariant_sequence_t<unit_mass<double>, 5>;
    using flux_vector_t       = covariant_sequence_t<unit_flux<double>, 5>;
    struct primitive_t;
    struct wavespeeds_t
    {
        unit_velocity<double> m;
        unit_velocity<double> p;
    };

    inline primitive_t recover_primitive(
        const conserved_density_t& U,
        double gamma_law_index);

    inline flux_vector_t riemann_hlle(
        const primitive_t& Pl,
        const primitive_t& Pr,
        const unit_vector_t& nhat,
        double gamma_law_index);
};




//=============================================================================
struct mara::srhd::primitive_t : public mara::arithmetic_sequence_t<double, 5, primitive_t>
{



    /**
     * @brief      Retrieve const-references to the quantities by name.
     */
    const double& mass_density() const { return operator[](0); }
    const double& gamma_beta_1() const { return operator[](1); }
    const double& gamma_beta_2() const { return operator[](2); }
    const double& gamma_beta_3() const { return operator[](3); }
    const double& gas_pressure() const { return operator[](4); }



    /**
     * @brief      Return a new state with individual, named quantities replaced.
     *
     * @param[in]  v     The new value
     *
     * @return     A new primitive variable state
     */
    primitive_t with_mass_density(double v) { auto res = *this; res[0] = v; return res; }
    primitive_t with_gamma_beta_1(double v) { auto res = *this; res[1] = v; return res; }
    primitive_t with_gamma_beta_2(double v) { auto res = *this; res[2] = v; return res; }
    primitive_t with_gamma_beta_3(double v) { auto res = *this; res[3] = v; return res; }
    primitive_t with_gas_pressure(double v) { auto res = *this; res[4] = v; return res; }



    /**
     * @brief      Return the fluid specific enthalpy.
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     h
     */
    double specific_enthalpy(double gamma_law_index) const
    {
        return enthalpy_density(gamma_law_index) / mass_density();
    }



    /**
     * @brief      Return the fluid enthalpy density.
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     rho h
     */
    double enthalpy_density(double gamma_law_index) const
    {
        return mass_density() + gas_pressure() * (1.0 + 1.0 / (gamma_law_index - 1.0));
    }




    /**
     * @brief      Return the fluid specific entropy
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     log(p / rho^gamma)
     */
    double specific_entropy(double gamma_law_index) const
    {
        return std::log(gas_pressure() / std::pow(mass_density(), gamma_law_index));
    }




    /**
     * @brief      Return the square of the four-velocity magnitude.
     *
     * @return     u^2
     */
    double gamma_beta_squared() const
    {
        const auto&_ = *this;
        return _[1] * _[1] + _[2] * _[2] + _[3] * _[3];
    }




    /**
     * @brief      Return the fluid Lorentz factor.
     *
     * @return     1 + u^2
     */
    double lorentz_factor() const
    {
        return std::sqrt(1.0 + gamma_beta_squared());
    }




    /**
     * @brief      Return the kinematic three-velocity along the given unit
     *             vector.
     *
     * @param[in]  nhat  The unit vector
     *
     * @return     v / c
     */
    double beta_along(const unit_vector_t& nhat) const
    {
        const auto&_ = *this;
        return nhat.project(_[1], _[2], _[3]) / lorentz_factor();
    }




    /**
     * @brief      Return the sound-speed squared.
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     gamma p / (rho h)
     */
    double sound_speed_squared(double gamma_law_index) const
    {
        return gamma_law_index * gas_pressure() / enthalpy_density(gamma_law_index);
    }




    /**
     * @brief      Convert this state to a density of conserved mass, momentum,
     *             and energy.
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     The conserved density U
     */
    conserved_density_t to_conserved_density(double gamma_law_index) const
    {
        const auto& _ = *this;
        auto W = lorentz_factor();
        auto h = specific_enthalpy(gamma_law_index);
        auto D = mass_density() * W;
        auto p = gas_pressure();
        auto U = conserved_density_t();
        U[0].value = D;
        U[1].value = D * _[1] * h;
        U[2].value = D * _[2] * h;
        U[3].value = D * _[3] * h;
        U[4].value = D * h * W - p - D;
        return U;
    }




    /**
     * @brief      Return the flux of conserved quantities in the given
     *             direction.
     *
     * @param[in]  nhat             The unit vector
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     The flux F
     */
    flux_vector_t flux(const unit_vector_t& nhat, double gamma_law_index) const
    {
        return flux(nhat, to_conserved_density(gamma_law_index));
    }




    /**
     * @brief      Same as the above function, except uses the given
     *             pre-computed conserved variable state - which can be useful
     *             for performance reasons if you have already computed U.
     *
     * @param[in]  nhat  The direction
     * @param[in]  U     The pre-computed conserved variables
     *
     * @return     The flux F
     */
    flux_vector_t flux(const unit_vector_t& nhat, const conserved_density_t& U) const
    {
        auto v = beta_along(nhat);
        auto p = gas_pressure();
        auto F = flux_vector_t();
        F[0].value = v * U[0].value;
        F[1].value = v * U[1].value + p * nhat.get_n1();
        F[2].value = v * U[2].value + p * nhat.get_n2();
        F[3].value = v * U[3].value + p * nhat.get_n3();
        F[4].value = v * U[4].value + p * v;
        return F;
    }



    /**
     * @brief      Return the wavespeeds along a given direction
     *
     * @param[in]  nhat             The direction
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     The wavespeeds
     */
    wavespeeds_t wavespeeds(const unit_vector_t& nhat, double gamma_law_index) const
    {
        auto c2 = sound_speed_squared(gamma_law_index);
        auto vn = beta_along(nhat);
        auto uu = gamma_beta_squared();
        auto vv = uu / (1 + uu);
        auto v2 = vn * vn;
        auto k0 = std::sqrt(c2 * (1 - vv) * (1 - vv * c2 - v2 * (1 - c2)));
        return {
            make_velocity((vn * (1 - c2) - k0) / (1 - vv * c2)),
            make_velocity((vn * (1 - c2) + k0) / (1 - vv * c2)),
        };
    }



    /** 
     * @brief      Return the spherical source terms for gamma-law SRHD
     *
     * @param[in]  spherical_radius   The spherical radius
     * @param[in]  polar_angle_theta  The polar angle theta
     * @param[in]  gamma_law_index    The gamma law index
     *
     * @return     Source terms in units of mass / volume / time
     */
    auto spherical_geometry_source_terms(
        double spherical_radius,
        double polar_angle_theta,
        double gamma_law_index)
    {
        auto cotq = std::tan(M_PI_2 - polar_angle_theta);
        auto ur = gamma_beta_1();
        auto uq = gamma_beta_2();
        auto up = gamma_beta_3();
        auto pg = gas_pressure();
        auto H = enthalpy_density(gamma_law_index);
        auto r = spherical_radius;
        auto S = covariant_sequence_t<dimensional_value_t<-3, 1, -1, double>, 5>();
        S[1].value = (2.0  * pg + H * (uq * uq        + up * up)) / r;
        S[2].value = (cotq * pg + H * (up * up * cotq - ur * uq)) / r;
        S[3].value =        -up * H * (ur + uq * cotq) / r;
        return S;
    }


    /**
     * @brief      Special case of the above for 1d radial flow
     *
     * @param[in]  spherical_radius  The spherical radius
     * @param[in]  gamma_law_index   The gamma law index
     *
     * @return     Source terms in units of mass / volume / time
     */
    auto spherical_geometry_source_terms_radial(double spherical_radius, double gamma_law_index)
    {
        auto uq = gamma_beta_2();
        auto pg = gas_pressure();
        auto H = enthalpy_density(gamma_law_index);
        auto r = spherical_radius;
        auto S = covariant_sequence_t<dimensional_value_t<-3, 1, -1, double>, 5>();
        S[1].value = (2.0 * pg + H * uq * uq) / r;
        return S;
    }
};




/**
 * @brief      Attempt to recover a primitive variable state from the given
 *             vector of conserved densities.
 *
 * @param[in]  U                The conserved densities
 * @param[in]  gamma_law_index  The gamma law index
 *
 * @return     A primitive variable state, if the recovery succeeds
 */
mara::srhd::primitive_t mara::srhd::recover_primitive(
    const conserved_density_t& U,
    double gamma_law_index)
{
    constexpr bool allowNegativePressure = false;
    constexpr int newtonIterMax          = 50;
    constexpr double errorTolerance      = 1e-10;

    // indexes to conserved quanitites
    enum {
        DDD = 0,
        S11 = 1,
        S22 = 2,
        S33 = 3,
        TAU = 4,
    };

    const double gm  = gamma_law_index;
    const double D   = U[DDD].value;
    const double tau = U[TAU].value;
    const double SS  =
    U[S11].value * U[S11].value +
    U[S22].value * U[S22].value +
    U[S33].value * U[S33].value;

    bool solution_found = 0;
    int iteration = 0;
    double W0 = 1.0;
    double p = 0.0; // guess pressure

    while (iteration < newtonIterMax)
    {
        auto v2  = std::min(SS / std::pow(tau + D + p, 2), 1.0 - 1e-10);
        auto W2  = 1.0 / (1.0 - v2);
        auto W   = std::sqrt(W2);
        auto e   = (tau + D * (1.0 - W) + p * (1.0 - W2)) / (D * W);
        auto d   = D / W;
        auto h   = 1.0 + e + p / d;
        auto cs2 = gm * p / (d * h);

        auto f = d * e * (gm - 1.0) - p;
        auto g = v2 * cs2 - 1.0;
        p -= f / g;

        if (std::fabs(f) < errorTolerance)
        {
            W0 = W;
            solution_found = true;
            break;
        }
        ++iteration;
    }

    auto P = primitive_t();

    P[0] = D / W0;
    P[1] = W0 * U[S11].value / (tau + D + p);
    P[2] = W0 * U[S22].value / (tau + D + p);
    P[3] = W0 * U[S33].value / (tau + D + p);
    P[4] = p;

    if (! solution_found)
    {
        throw std::invalid_argument("mara::srhd::recover_primitive failure: "
            "root finder not converging U=" + to_string(U));
    }
    if (P.gas_pressure() < 0.0 && ! allowNegativePressure)
    {
        throw std::invalid_argument("mara::srhd::recover_primitive failure: "
            "negative pressure U=" + mara::to_string(U));
    }
    if (P.mass_density() < 0.0)
    {
        throw std::invalid_argument("mara::srhd::recover_primitive failure: "
            "negative density U=" + mara::to_string(U));
    }
    if (std::isnan(W0))
    {
        throw std::invalid_argument("mara::srhd::recover_primitive failure: "
            "nan W U=" + mara::to_string(U));
    }
    return P;
}




/**
 * @brief      Return the HLLE flux for the given pair of states
 *
 * @param[in]  Pl               The state to the left of the interface
 * @param[in]  Pr               The state to the right
 * @param[in]  nhat             The normal vector to the interface
 * @param[in]  gamma_law_index  The gamma law index
 *
 * @return     A vector of fluxes
 */
mara::srhd::flux_vector_t mara::srhd::riemann_hlle(
    const primitive_t& Pl,
    const primitive_t& Pr,
    const unit_vector_t& nhat,
    double gamma_law_index)
{
    auto Ul = Pl.to_conserved_density(gamma_law_index);
    auto Ur = Pr.to_conserved_density(gamma_law_index);
    auto Al = Pl.wavespeeds(nhat, gamma_law_index);
    auto Ar = Pr.wavespeeds(nhat, gamma_law_index);
    auto Fl = Pl.flux(nhat, Ul);
    auto Fr = Pr.flux(nhat, Ur);

    auto ap = std::max(make_velocity(0.0), std::max(Al.p, Ar.p));
    auto am = std::min(make_velocity(0.0), std::min(Al.m, Ar.m));

    return (Fl * ap - Fr * am - (Ul - Ur) * ap * am) / (ap - am);
}
