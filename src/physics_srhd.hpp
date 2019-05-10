#pragma once
#include <cmath>
#include "core_datatypes.hpp"




//=============================================================================
namespace mara::srhd
{
    using conserved_density_t = dimensional_sequence_t<intrinsic_t, 5>;
    using conserved_t         = dimensional_sequence_t<extrinsic_t, 5>;
    using flux_vector_t       = dimensional_sequence_t<flux_t, 5>;
    using flow_vector_t       = dimensional_sequence_t<flow_rate_t, 5>;

    struct primitive_t;
    struct wavespeeds_t { double p; double m; };

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
    const double& mass_density() const { return operator[](0); }
    const double& gamma_beta_1() const { return operator[](1); }
    const double& gamma_beta_2() const { return operator[](2); }
    const double& gamma_beta_3() const { return operator[](3); }
    const double& gas_pressure() const { return operator[](4); }

    primitive_t mass_density(double v) { auto res = *this; res[0] = v; return res; }
    primitive_t gamma_beta_1(double v) { auto res = *this; res[1] = v; return res; }
    primitive_t gamma_beta_2(double v) { auto res = *this; res[2] = v; return res; }
    primitive_t gamma_beta_3(double v) { auto res = *this; res[3] = v; return res; }
    primitive_t gas_pressure(double v) { auto res = *this; res[4] = v; return res; }

    double specific_enthalpy(double gamma_law_index) const
    {
        const double e = gas_pressure() / (mass_density() * (gamma_law_index - 1.0));
        const double h = 1.0 + e + gas_pressure() / mass_density();
        return h;
    }

    double gamma_beta_squared() const
    {
        const auto&_ = *this;
        return _[1] * _[1] + _[2] * _[2] + _[3] * _[3];
    }

    double lorentz_factor() const
    {
        return std::sqrt(1.0 + gamma_beta_squared());
    }

    area_t beta_along(const area_element_t& dA) const
    {
        const auto&_ = *this;
        return (dA[0] * _[1] + dA[1] * _[2] + dA[2] * _[3]) / lorentz_factor();
    }

    double beta_along(const unit_vector_t& nhat) const
    {
        const auto&_ = *this;
        return nhat.project(_[1], _[2], _[3]) / lorentz_factor();
    }

    double sound_speed_squared(double gamma_law_index) const
    {
        return gamma_law_index * gas_pressure() / (mass_density() * specific_enthalpy(gamma_law_index));
    }

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

    flux_vector_t flux(const unit_vector_t& nhat, double gamma_law_index) const
    {
        return flux(nhat, to_conserved_density(gamma_law_index));
    }

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

    wavespeeds_t wavespeeds(const unit_vector_t& nhat, double gamma_law_index) const
    {
        auto c2 = sound_speed_squared(gamma_law_index);
        auto vn = beta_along(nhat);
        auto uu = gamma_beta_squared();
        auto vv = uu / (1 + uu);
        auto v2 = vn * vn;
        auto k0 = std::sqrt(c2 * (1 - vv) * (1 - vv * c2 - v2 * (1 - c2)));
        return {
            (vn * (1 - c2) - k0) / (1 - vv * c2),
            (vn * (1 - c2) + k0) / (1 - vv * c2),
        };
    }
};




//=============================================================================
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
        LAR = 5,
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
    double f;
    double g;
    double W0 = 1.0;
    double p = 0.0; // guess pressure

    while (iteration < newtonIterMax)
    {
        double v2  = std::min(SS / std::pow(tau + D + p, 2), 1.0 - 1e-10);
        double W2  = 1.0 / (1.0 - v2);
        double W   = std::sqrt(W2);
        double e   = (tau + D * (1.0 - W) + p * (1.0 - W2)) / (D * W);
        double d   = D / W;
        double h   = 1.0 + e + p / d;
        double cs2 = gm * p / (d * h);

        f = d * e * (gm - 1.0) - p;
        g = v2 * cs2 - 1.0;
        p -= f / g;

        if (std::fabs(f) < errorTolerance)
        {
            W0 = W;
            solution_found = true;
        }
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




//=============================================================================
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

    auto epl = std::max(Al.m, Al.p);
    auto eml = std::min(Al.m, Al.p);
    auto epr = std::max(Ar.m, Ar.p);
    auto emr = std::min(Ar.m, Ar.p);
    auto ap  = make_velocity(std::max(epl, epr));
    auto am  = make_velocity(std::min(eml, emr));
    auto da  = ap - am;

    return Fl * (ap / da) - Fr / (am / da) - (Ul - Ur) * ap * (am / da);
}
