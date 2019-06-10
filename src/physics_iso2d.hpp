/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake and Andrew MacFadyen

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
    using unit_flow               = dimensional_value_t< 0, 1,-1, double>;
    using unit_flux               = dimensional_value_t<-1, 1,-1, double>;

    using conserved_per_area_t    = covariant_sequence_t<unit_conserved_per_area, 3>;
    using conserved_t             = covariant_sequence_t<unit_conserved, 3>;
    using flow_t                  = covariant_sequence_t<unit_flow, 3>;
    using flux_t                  = covariant_sequence_t<unit_flux, 3>;

    struct primitive_t;
    struct wavespeeds_t
    {
        unit_velocity<double> m;
        unit_velocity<double> p;
    };
    struct riemann_hllc_variables_t;

    static inline primitive_t recover_primitive(
        const conserved_per_area_t& U,
        double density_floor);

    static inline primitive_t roe_average(
        const primitive_t& Pl,
        const primitive_t& Pr);

    static inline flux_t riemann_hlle(
        const primitive_t& Pl,
        const primitive_t& Pr,
        double sound_speed_squared_l,
        double sound_speed_squared_r,
        const unit_vector_t& nhat);

    static inline flux_t riemann_hllc(
        const primitive_t& Pl,
        const primitive_t& Pr,
        double sound_speed_squared_l,
        double sound_speed_squared_r,
        const unit_vector_t& nhat);

    static inline riemann_hllc_variables_t compute_hllc_variables(
        const primitive_t& Pl,
        const primitive_t& Pr,
        double sound_speed_squared_l,
        double sound_speed_squared_r,
        const unit_vector_t& nhat);
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

    unit_velocity<double> velocity_magnitude() const
    {
        return std::sqrt(velocity_squared());
    }

    auto velocity() const
    {
        return covariant_sequence_t<double, 3>{velocity_x(), velocity_y(), 0.0};
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
     * @brief      Return the (vertically integrated) gas pressure, given the
     *             sound speed squared.
     *
     * @param[in]  sound_speed_squared  The sound speed squared
     *
     * @return     The vertically integrated gas pressure
     */
    double gas_pressure(double sound_speed_squared) const
    {
        return sigma() * sound_speed_squared;
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
        auto p = gas_pressure(sound_speed_squared);
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
mara::iso2d::primitive_t mara::iso2d::recover_primitive(const conserved_per_area_t& U, double density_floor)
{
    if (U[0] < 0.0)
    {
        if (density_floor == 0.0)
        {
            throw std::invalid_argument("mara::iso2d::recover_primitive (negative density)");
        }
        else
        {
            // This is a pretty extreme measure. If it happens too often you
            // should expect other things to go wrong.
            return primitive_t()
            .with_sigma(density_floor)
            .with_velocity_x(0.0)
            .with_velocity_y(0.0);
        }
    }
    auto P = primitive_t();
    P[0] = U[0].value;
    P[1] = U[1].value / P[0];
    P[2] = U[2].value / P[0];
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
 * @param[in]  Pl                     The state to the left of the interface
 * @param[in]  Pr                     The state to the right
 * @param[in]  sound_speed_squared_l  The sound speed squared to the left of the interface
 * @param[in]  sound_speed_squared_r  The sound speed squared to the right
 * @param[in]  nhat                   The normal vector to the interface
 * @param[in]  gamma_law_index  The gamma law index
 *
 * @return     A vector of fluxes
 */
mara::iso2d::flux_t mara::iso2d::riemann_hlle(
    const primitive_t& Pl,
    const primitive_t& Pr,
    double sound_speed_squared_l,
    double sound_speed_squared_r,
    const unit_vector_t& nhat)
{
    auto Ul = Pl.to_conserved_per_area();
    auto Ur = Pr.to_conserved_per_area();
    auto Al = Pl.wavespeeds(nhat, sound_speed_squared_l);
    auto Ar = Pr.wavespeeds(nhat, sound_speed_squared_r);
    auto Fl = Pl.flux(nhat, sound_speed_squared_l);
    auto Fr = Pr.flux(nhat, sound_speed_squared_r);

    auto ap = std::max(make_velocity(0.0), std::max(Al.p, Ar.p));
    auto am = std::min(make_velocity(0.0), std::min(Al.m, Ar.m));

    return (Fl * ap - Fr * am - (Ul - Ur) * ap * am) / (ap - am);
}




/**
 * @brief      Data structure containing variables used in an HLLC calculation.
 *             It's nice to have this, as opposed to defining all the variables
 *             in the HLLC flux function, because we can use it in unit tests to
 *             check the intermediate results. It might also be used to derive
 *             other quantities, like the flux through a moving face, or the
 *             star-state itself, in case that's needed in addition to the flux.
 *
 * @note       Use compute_hllc_variables below to create an instance of this
 *             struct from left and right states.
 */
struct mara::iso2d::riemann_hllc_variables_t
{
    mara::unit_vector_t nhat;
    mara::iso2d::primitive_t Pl;
    mara::iso2d::primitive_t Pr;
    mara::covariant_sequence_t<double, 3> v_para_l;
    mara::covariant_sequence_t<double, 3> v_para_r;
    mara::covariant_sequence_t<double, 3> v_perp_l;
    mara::covariant_sequence_t<double, 3> v_perp_r;

    double ul;
    double ur;
    double sigma_l;
    double sigma_r;
    double sigma_bar;
    double al;
    double ar;
    double a_bar;
    double press_l;
    double press_r;
    double ppvrs;
    double pstar;
    double ql;
    double qr;
    double sl;
    double sr;
    double sstar;

    auto contact_speed() const { return sstar; }
    auto Ul() const { return Pl.to_conserved_per_area(); }
    auto Ur() const { return Pr.to_conserved_per_area(); }
    auto Fl() const { return Pl.flux(nhat, al * al); }
    auto Fr() const { return Pr.flux(nhat, ar * ar); }

    auto Ul_star() const
    {
        return conserved_per_area_t
        {
            sigma_l * (sl - ul) / (sl - sstar),
            sigma_l * (sl - ul) / (sl - sstar) * (sstar * nhat[0] + v_perp_l[0]),
            sigma_l * (sl - ul) / (sl - sstar) * (sstar * nhat[1] + v_perp_l[1]),
        };
    }

    auto Ur_star() const
    {
        return conserved_per_area_t
        {
            sigma_r * (sr - ur) / (sr - sstar),
            sigma_r * (sr - ur) / (sr - sstar) * (sstar * nhat[0] + v_perp_r[0]),
            sigma_r * (sr - ur) / (sr - sstar) * (sstar * nhat[1] + v_perp_r[1]),
        };
    }

    auto interface_flux() const
    {
        if      (0.0   <= sl                 ) return Fl();
        else if (sl    <= 0.0 && 0.0 <= sstar) return Fl() + (Ul_star() - Ul()) * make_velocity(sl);
        else if (sstar <= 0.0 && 0.0 <= sr   ) return Fr() + (Ur_star() - Ur()) * make_velocity(sr);
        else if (sr    <= 0.0                ) return Fr();
        throw std::invalid_argument("riemann_hllc_variables_t::interface_flux");
    }

    auto interface_conserved_state() const
    {
        if      (0.0   <= sl                 ) return Ul();
        else if (sl    <= 0.0 && 0.0 <= sstar) return Ul_star();
        else if (sstar <= 0.0 && 0.0 <= sr   ) return Ur_star();
        else if (sr    <= 0.0                ) return Ur();
        throw std::invalid_argument("riemann_hllc_variables_t::interface_conserved_state");
    }
};




/**
 * @brief      Return the HLLC variables for the given pair of states, following
 *             Toro 3rd ed. Sec 10.6.
 *
 * @param[in]  Pl                     The state to the left of the interface
 * @param[in]  Pr                     The state to the right
 * @param[in]  sound_speed_squared_l  The Sound speed squared to the left
 * @param[in]  sound_speed_squared_r  The Sound speed squared to the right
 * @param[in]  nhat                   The normal vector to the interface
 *
 * @return     A vector of fluxes
 */
inline mara::iso2d::riemann_hllc_variables_t mara::iso2d::compute_hllc_variables(
    const mara::iso2d::primitive_t& Pl,
    const mara::iso2d::primitive_t& Pr,
    double sound_speed_squared_l,
    double sound_speed_squared_r,
    const mara::unit_vector_t& nhat)
{
    // left and right parallel velocity magnitudes
    auto ul = Pl.velocity_along(nhat);
    auto ur = Pr.velocity_along(nhat);

    // left and right parallel and perpendicular velocity vectors
    auto v_para_l = nhat * ul;
    auto v_para_r = nhat * ur;
    auto v_perp_l = Pl.velocity() - v_para_l;
    auto v_perp_r = Pr.velocity() - v_para_r;

    // left, right, and average sigma's
    auto sigma_l   = Pl.sigma();
    auto sigma_r   = Pr.sigma();
    auto sigma_bar = 0.5 * (sigma_l + sigma_r);

    // left, right, and average sound speeds
    auto al        = std::sqrt(sound_speed_squared_l);
    auto ar        = std::sqrt(sound_speed_squared_r);
    auto a_bar     = 0.5 * (al + ar);

    // left, right, and star-state pressures (Equation 10.61)
    auto press_l   = sigma_l * sound_speed_squared_l;
    auto press_r   = sigma_r * sound_speed_squared_r;
    auto ppvrs     = 0.5 * (press_l + press_r) - 0.5 * (ur - ul) * sigma_bar * a_bar; 
    auto pstar     = std::max(0.0, ppvrs);

    // Equation 10.69 for the isothermal case, gamma = 1
    auto ql = std::max(1.0, std::sqrt(pstar / press_l));
    auto qr = std::max(1.0, std::sqrt(pstar / press_r));

    // Equation 10.68 for the wavespeeds
    auto sl = ul - al * ql;
    auto sr = ur + ar * qr;

    // Equation 10.70 for the contact speed
    auto den = sigma_l * (sl - ul) - sigma_r * (sr - ur);
    auto sstar = (press_r - press_l + ul * sigma_l * (sl - ul) - ur * sigma_r * (sr - ur)) / den;

    if (std::isnan(sstar)) //den == 0.0)
    {
        throw std::invalid_argument("mara::iso2d::compute_hllc_variables (probably hitting too many density floors)");
    }

    //=========================================================================
    auto r = riemann_hllc_variables_t();
    r.nhat = nhat;
    r.Pl = Pl;
    r.Pr = Pr;
    r.v_para_l = v_para_l;
    r.v_para_r = v_para_r;
    r.v_perp_l = v_perp_l;
    r.v_perp_r = v_perp_r;
    r.ul = ul;
    r.ur = ur;
    r.sigma_l = sigma_l;
    r.sigma_r = sigma_r;
    r.sigma_bar = sigma_bar;
    r.al = al;
    r.ar = ar;
    r.a_bar = a_bar;
    r.press_l = press_l;
    r.press_r = press_r;
    r.ppvrs = ppvrs;
    r.pstar = pstar;
    r.ql = ql;
    r.qr = qr;
    r.sl = sl;
    r.sr = sr;
    r.sstar = sstar;
    return r;
}




/**
 * @brief      Return the HLLC flux for the given pair of states, following Toro
 *             3rd ed. Sec 10.6.
 *
 * @param[in]  Pl                     The state to the left of the interface
 * @param[in]  Pr                     The state to the right
 * @param[in]  sound_speed_squared_l  The Sound speed squared to the left
 * @param[in]  sound_speed_squared_r  The Sound speed squared to the right
 * @param[in]  nhat                   The normal vector to the interface
 *
 * @return     A vector of fluxes
 */
mara::iso2d::flux_t mara::iso2d::riemann_hllc(
    const primitive_t& Pl,
    const primitive_t& Pr,
    double sound_speed_squared_l,
    double sound_speed_squared_r,
    const unit_vector_t& nhat)
{
    return compute_hllc_variables(Pl, Pr, sound_speed_squared_l, sound_speed_squared_r, nhat).interface_flux();
}
