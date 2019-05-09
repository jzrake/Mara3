#include <cmath>
#include <algorithm>
#include <array>
#include "datatypes.hpp"




//=============================================================================
template<typename DerivedType>
struct dimensional_t
{
    DerivedType operator+(dimensional_t v) const { return {{ value + v.value }}; }
    DerivedType operator-(dimensional_t v) const { return {{ value - v.value }}; }
    DerivedType operator*(double s) const { return {{ value * s }}; }
    DerivedType operator/(double s) const { return {{ value / s }}; }

    double value;
};



//=============================================================================
namespace mara::srhd
{
    struct time_delta_t : dimensional_t<time_delta_t> {};
    struct area_t       : dimensional_t<area_t> {};
    struct volume_t     : dimensional_t<volume_t> {};
    struct intrinsic_t  : dimensional_t<intrinsic_t> {}; // e.g. energy / volume
    struct extrinsic_t  : dimensional_t<extrinsic_t> {}; // intrinsic * volume
    struct flow_rate_t  : dimensional_t<flow_rate_t> {}; // extrinsic / time
    struct flux_t       : dimensional_t<flux_t> {};      // flow_rate_t / area

    extrinsic_t operator*(intrinsic_t i, volume_t v) { return {{ i.value * v.value }}; }
    intrinsic_t operator/(extrinsic_t e, volume_t v) { return {{ e.value / v.value }}; }

    extrinsic_t operator*(flow_rate_t e, time_delta_t t) { return {{ e.value * t.value }}; }
    flow_rate_t operator/(extrinsic_t e, time_delta_t t) { return {{ e.value / t.value }}; }

    flow_rate_t operator*(flux_t f, area_t a) { return {{ f.value * a.value }}; }
    flux_t operator/(flow_rate_t r, area_t a) { return {{ r.value / a.value }}; }


    struct primitive_t;

    using conserved_density_t = dimensional_sequence_t<5, intrinsic_t>;
    using conserved_t         = dimensional_sequence_t<5, extrinsic_t>;
    using flux_vector_t       = dimensional_sequence_t<5, flux_t>;
    using area_element_t      = dimensional_sequence_t<3, area_t>;
};




//=============================================================================
struct mara::srhd::primitive_t : public mara::arithmetic_sequence_t<5, double, primitive_t>
{
    const double& mass_density() const { return operator[](0); }
    const double& gamma_beta_1() const { return operator[](1); }
    const double& gamma_beta_2() const { return operator[](2); }
    const double& gamma_beta_3() const { return operator[](3); }
    const double& gas_pressure() const { return operator[](4); }

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

    double beta_along(const area_element_t& dA) const
    {
        const auto&_ = *this;
        return (dA[0] * _[1] + dA[1] * _[2] + dA[2] * _[3]).value / lorentz_factor();
    }

    double sound_speed_squared(double gamma_law_index) const
    {
        return gamma_law_index * gas_pressure() / (mass_density() * specific_enthalpy(gamma_law_index));
    }

    conserved_t to_conserved(double gamma_law_index) const
    {
        const auto& _ = *this;
        auto W = lorentz_factor();
        auto h = specific_enthalpy(gamma_law_index);
        auto D = mass_density() * W;
        auto p = gas_pressure();
        auto U = conserved_t();
        U[0].value = D;
        U[1].value = D * _[1] * h;
        U[2].value = D * _[2] * h;
        U[3].value = D * _[3] * h;
        U[4].value = D * h * W - p - D;
        return U;
    }

    flux_vector_t flux(const area_element_t& dA, double gamma_law_index) const
    {
        auto v = beta_along(dA);
        auto p = gas_pressure();
        auto U = to_conserved(gamma_law_index);
        auto F = flux_vector_t();
        F[0].value = v * U[0].value;
        F[1].value = v * U[1].value + p * dA[0].value;
        F[2].value = v * U[2].value + p * dA[1].value;
        F[3].value = v * U[3].value + p * dA[2].value;
        F[4].value = v * U[4].value + p * v;
        return F;
    }
};
