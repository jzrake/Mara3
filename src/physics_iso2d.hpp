/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 
 ==============================================================================
*/




#pragma once
#include <cmath>
#include "core_geometric.hpp"
#include "core_matrix.hpp"




//=============================================================================
namespace mara { struct iso2d; }




//=============================================================================
struct mara::iso2d
{
    using unit_conserved_per_area = dimensional_value_t<-2, 1, 0, double>;
    using unit_conserved          = dimensional_value_t< 0, 1, 0, double>;
    using unit_flux               = dimensional_value_t<-1, 1,-1, double>;

    using conserved_per_area_t    = covariant_sequence_t<unit_conserved_per_area, 3>;
    using conserved_t             = covariant_sequence_t<unit_conserved, 3>;
    using flux_t           = covariant_sequence_t<unit_flux, 3>;

    struct primitive_t;
    struct wavespeeds_t
    {
        unit_velocity<double> m;
        unit_velocity<double> p;
    };

    static inline primitive_t recover_primitive(const conserved_per_area_t& U);

    static inline primitive_t roe_average(
        const primitive_t& Pl,
        const primitive_t& Pr);

    static inline flux_t riemann_hlle(
        const primitive_t& Pl,
        const primitive_t& Pr,
        const unit_vector_t& nhat,
        double sound_speed_squared);
};




//=============================================================================
struct mara::iso2d::primitive_t : public mara::arithmetic_sequence_t<double, 3, primitive_t>
{




    /**
     * @brief      Retrieve const-references to the quantities by name.
     */
    const double& sigma() const { return operator[](0); }
    const double& velocity_x() const { return operator[](1); }
    const double& velocity_y() const { return operator[](2); }




    /**
     * @brief      Return a new state with individual, named quantities replaced.
     *
     * @param[in]  v     The new value
     *
     * @return     A new primitive variable state
     */
    primitive_t with_sigma(double v) const { auto res = *this; res[0] = v; return res; }
    primitive_t with_velocity_x(double v) const { auto res = *this; res[1] = v; return res; }
    primitive_t with_velocity_y(double v) const { auto res = *this; res[2] = v; return res; }




    /**
     * @brief      Return the square of the four-velocity magnitude.
     *
     * @return     u^2
     */
    double velocity_squared() const
    {
        const auto&_ = *this;
        return _[1] * _[1] + _[2] * _[2];
    }




    /**
     * @brief      Return the kinematic three-velocity along the given unit
     *             vector.
     *
     * @param[in]  nhat  The unit vector
     *
     * @return     v
     */
    double velocity_along(const unit_vector_t& nhat) const
    {
        const auto&_ = *this;
        return nhat.project(_[1], _[2], 0.0);
    }




    /**
     * @brief      Convert this state to conserved mass and momentum, per unit
     *             area.
     *
     * @return     The conserved quantities per unit area.
     */
    conserved_per_area_t to_conserved_per_area() const
    {
        auto U = conserved_per_area_t();
        U[0].value = sigma();
        U[1].value = sigma() * velocity_x();
        U[2].value = sigma() * velocity_y();
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
    flux_t flux(const unit_vector_t& nhat, double sound_speed_squared) const
    {
        auto v = velocity_along(nhat);
        auto p = sigma() * sound_speed_squared;
        auto F = flux_t();
        F[0] = v * sigma();
        F[1] = v * sigma() * velocity_x() + p * nhat.get_n1();
        F[2] = v * sigma() * velocity_y() + p * nhat.get_n2();
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
    wavespeeds_t wavespeeds(const unit_vector_t& nhat, double sound_speed_squared) const
    {
        auto cs = std::sqrt(sound_speed_squared);
        auto vn = velocity_along(nhat);
        return {
            make_velocity(vn - cs),
            make_velocity(vn + cs),
        };
    }
};




/**
 * @brief      Attempt to recover a primitive variable state from the given
 *             vector of conserved densities.
 *
 * @param[in]  U     The conserved densities
 *
 * @return     A primitive variable state, if the recovery succeeds
 */
mara::iso2d::primitive_t mara::iso2d::recover_primitive(const conserved_per_area_t& U)
{
    auto P = primitive_t();
    P[0] = U[0].value;
    P[1] = U[1].value / U[0].value;
    P[2] = U[2].value / U[0].value;
    return P;
}




/**
 * @brief      Compute a sensible Roe-average state Q = Roe(Pr, Pl), defined
 *             here as the primitives weighted by square root of the mass
 *             density. This averaged state is symmetric in (Pr, Pl), and has
 *             the property that A(Q) * (Ur - Ul) = F(Ur) - F(Ul), where A is
 *             the flux Jacobian.
 *
 * @param[in]  Pr    The other state
 * @param[in]  Pl    The first state
 *
 * @return     The Roe-averaged primitive state
 */
mara::iso2d::primitive_t mara::iso2d::roe_average(
    const primitive_t& Pr,
    const primitive_t& Pl)
{
    auto kr = std::sqrt(Pr.sigma());
    auto kl = std::sqrt(Pl.sigma());
    return (Pr * kr + Pl * kl) / (kr + kl);
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
mara::iso2d::flux_t mara::iso2d::riemann_hlle(
    const primitive_t& Pl,
    const primitive_t& Pr,
    const unit_vector_t& nhat,
    double sound_speed_squared)
{
    auto Ul = Pl.to_conserved_per_area();
    auto Ur = Pr.to_conserved_per_area();
    auto Al = Pl.wavespeeds(nhat, sound_speed_squared);
    auto Ar = Pr.wavespeeds(nhat, sound_speed_squared);
    auto Fl = Pl.flux(nhat, sound_speed_squared);
    auto Fr = Pr.flux(nhat, sound_speed_squared);

    auto ap = std::max(make_velocity(0.0), std::max(Al.p, Ar.p));
    auto am = std::min(make_velocity(0.0), std::min(Al.m, Ar.m));

    return (Fl * ap - Fr * am - (Ul - Ur) * ap * am) / (ap - am);
}
