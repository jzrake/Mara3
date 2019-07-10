/**
 ==============================================================================
 Copyright 2019, Christopher Tiede

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
namespace mara { struct mhd; }




//=============================================================================
struct mara::mhd
{
    using unit_conserved_density  = dimensional_value_t<-3, 1, 0, double>;
    using unit_conserved          = dimensional_value_t< 0, 1, 0, double>;
    using unit_flow               = dimensional_value_t< 0, 1,-1, double>;
    using unit_flux               = dimensional_value_t< -2, 1,-1, double>; //flux through faces

    using conserved_density_t     = arithmetic_sequence_t<unit_conserved_density, 8>; //8 since B_x != const?
    using conserved_t             = arithmetic_sequence_t<unit_conserved        , 8>;
    using flux_vector_t           = arithmetic_sequence_t<unit_flux             , 8>;
    
    struct primitive_t;

    struct wavespeeds_t
    {
        //TODO: NEED TO ADD EXTRA WAVE SPEEDS HERE -- which to carry around?
        unit_velocity<double> m;
        unit_velocity<double> p;
    };

    static inline primitive_t recover_primitive(
        const conserved_density_t& U,
        double gamma_law_index,
        double temperature_floor);

    static inline primitive_t roe_average(
        const primitive_t& Pl,
        const primitive_t& Pr);

    static inline flux_vector_t riemann_hlle(
        const primitive_t& Pl,
        const primitive_t& Pr,
        const unit_vector_t& nhat,
        double gamma_law_index);
};




//=============================================================================
struct mara::mhd::primitive_t : public mara::derivable_sequence_t<double, 8, primitive_t>
{




    /**
     * @brief      Retrieve const-references to the quantities by name.
     */
    const double& mass_density() const { return operator[](0); }
    const double& velocity_1()   const { return operator[](1); }
    const double& velocity_2()   const { return operator[](2); }
    const double& velocity_3()   const { return operator[](3); }
    const double& gas_pressure() const { return operator[](4); }
    
    const double& bfield_1()     const { return operator[](5); }
    const double& bfield_2()     const { return operator[](6); }
    const double& bfield_3()     const { return operator[](7); }




    /**
     * @brief      Return a new state with individual, named quantities replaced.
     *
     * @param[in]  v     The new value
     *
     * @return     A new primitive variable state
     */
    primitive_t with_mass_density(double v) const { auto res = *this; res[0] = v; return res; }
    primitive_t with_velocity_1(double v)   const { auto res = *this; res[1] = v; return res; }
    primitive_t with_velocity_2(double v)   const { auto res = *this; res[2] = v; return res; }
    primitive_t with_velocity_3(double v)   const { auto res = *this; res[3] = v; return res; }
    primitive_t with_gas_pressure(double v) const { auto res = *this; res[4] = v; return res; }
    
    primitive_t with_bfield_1(double v)     const { auto res = *this; res[5] = v; return res; }
    primitive_t with_bfield_2(double v)     const { auto res = *this; res[6] = v; return res; }
    primitive_t with_bfield_3(double v)     const { auto res = *this; res[7] = v; return res; }



    /**
     * @brief      Return the fluid specific enthalpy.
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     h = (u + p) / rho
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
     * @return     H = u + p
     */
    double enthalpy_density(double gamma_law_index) const
    {
        return gas_pressure() * (1.0 + 1.0 / (gamma_law_index - 1.0));
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
    double velocity_squared() const
    {
        const auto&_ = *this;
        return _[1] * _[1] + _[2] * _[2] + _[3] * _[3];
    }

    /**
     * @brief      Return the kinematic three-velocity along the given unit
     *             vector.
     *
     * @param[in]  nhat  The unit vector
     *
     * @return     v / c
     */
    double velocity_along(const unit_vector_t& nhat) const
    {
        const auto&_ = *this;
        return nhat.project(_[1], _[2], _[3]);
    }



     /**
     * @brief      Return the square of the magnetic field magnitude.
     *
     * @return     u^2
     */
    double bfield_squared() const
    {
        const auto &_ = *this;
        return _[5] * _[5] + _[6] * _[6] + _[7] * _[7];
    }

    /**
     * @brief      Return the magnetic field along the given unit
     *             vector.
     *
     * @param[in]  nhat  The unit vector
     *
     * @return     v / c
     */
    double bfield_along(const unit_vector_t& nhat) const
    {
        const auto &_ = *this;
        return nhat.project(_[5], _[6], _[7]);
    }

    double bfield_dot_velocity() const
    {
        //Vector correspondence between 1 and 5; 2 and 6; 3 and 7 (e.g. vx and Bx)
        const auto &_ = *this;
        return _[1] * _[5] + _[2] * _[6] + _[3] * _[7];
    }



   /**
     * @brief      Return the sound-speed squared.
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     gamma p / rho
     */
    double sound_speed_squared(double gamma_law_index) const
    {
        return gamma_law_index * gas_pressure() / mass_density();
    }

   /**
     * @brief      Return Alfven-speed squared
     *
     * @return     B^2 / (4*pi*rho)
     */
    double alfven_speed_squared() const
    {
        return bfield_squared() / (4 * M_PI * mass_density());
    }

   /**
     * @brief      Return the squared-Alfven-speed along some direction
     *
     * @param[in]  nhat  The unit vector
     *
     * @return     (B_nhat)^2 / (4*pi*rho)
     */
    double alfven_speed_squared_along(const unit_vector_t& nhat) const
    {
        const auto b_along = bfield_along(nhat);  
        return b_along*b_along / (4 * M_PI * mass_density());
    }

   /**
     * @brief      Return the fast/slow magnetosonic wavespeed squared along nhat
     *
     * @param[in]  fast_slow  The delimiter for the desired wave
     *
     * @param[in]  nhat       The unit vector
     *
     * @return     0.5 * ( c_s^2 + c_a^2 ) \pm 0.5*\sqrt{ (c_s^2+c_a^2)^2 - 4*c_s^2*c_{a,i}^2 }
     */
    // **** Not sure I need this. And the fast_slow thing might be a bad idea.... lol
    double magneto_sonic_speed_square(std::string fast_slow, unit_vector_t& nhat) const
    {
        const auto cs2 = sound_speed_squared();
        const auto one = cs2 + alfven_speed_squared();
        const auto two = 0.5 * std::sqrt(one * one - 4 * cs2 * alfven_speed_squared_along(nhat));

        double result = 0.0;
        switch(fast_slow)
        {
            case "fast": result = 0.5 * one + two; break;
            case "slow": result = 0.5 * one - two; break;
            default: 
                throw std::invalid_argument("invalid call to magneto_sonic_speed_square; must specify with strings 'fast' or 'slow' ");
        }
        return result;
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
        auto d = mass_density();
        auto p = gas_pressure();
        auto U = conserved_density_t();
        U[0].value = d;
        U[1].value = d * _[1];
        U[2].value = d * _[2];
        U[3].value = d * _[3];
        U[4].value = 0.5 * d * velocity_squared() + p / (gamma_law_index - 1);

        //Bfields
        U[5].value = _[5];
        U[6].value = _[6];
        U[7].value = _[7];
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
        auto v  = velocity_along(nhat);
        auto b  = bfield_along(nhat);
        auto p  = gas_pressure();
        auto bv = bfield_dot_velocity();
        auto F = flux_vector_t();
        F[0].value = v * U[0].value;
        F[1].value = v * U[1].value + p * nhat.get_n1() /*mhd*/ - U[5].value * b / (4*M_PI); //Do I need all these 4pi's?...
        F[2].value = v * U[2].value + p * nhat.get_n2() /*mhd*/ - U[6].value * b / (4*M_PI);
        F[3].value = v * U[3].value + p * nhat.get_n3() /*mhd*/ - U[7].value * b / (4*M_PI); 
        F[4].value = v * U[4].value + p * v             /*mhd*/ - b * bv   / (4*M_PI);

        // mhd: B \cross v
        // when flux is in direction of nhat it should become zero...
        //      should work if no error associated with _along() functions...
        F[5].value = v * U[5].value - U[1].value * b;
        F[6].value = v * U[6].value - U[2].value * b;
        F[7].value = v * U[7].value - U[3].value * b;

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
        auto cs = std::sqrt(sound_speed_squared(gamma_law_index));
        auto vn = velocity_along(nhat);
        return {
            make_velocity(vn - cs),
            make_velocity(vn + cs),
        };
    }

    /**
     * @brief      Above function generalized to other wave speeds
     *
     * @param[in]  nhat             The direction
     * @param[in]  wave_function    The function returning the proper wavespeed
     *
     * @return     The wavespeeds
     */
`   //
    // *** I'm not sure I'm allowed to overload a function like this
    // ***     or even that passing a function like that even works haha
    // 
    wavespeeds_t wavespeeds(const unit_vector_t& nhat, auto wave_fuction) const
    {
        auto c0 = std::sqrt(wave_function);
        auto vn = velocity_along(nhat);
        return {
            make_velocity(vn - c0),
            make_velocity(vn + c0)
        };
    }




    /**
     * @brief      Return the spherical source terms for gamma-law mhd
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
        auto vr = velocity_1();
        auto vq = velocity_2();
        auto vp = velocity_3();
        auto pg = gas_pressure();
        auto d = mass_density();
        auto r = spherical_radius;
        auto S = arithmetic_sequence_t<dimensional_value_t<-3, 1, -1, double>, 5>();
        S[1].value = (2.0  * pg + d * (vq * vq        + vp * vp)) / r;
        S[2].value = (cotq * pg + d * (vp * vp * cotq - vr * vq)) / r;
        S[3].value =        -vp * d * (vr + vq * cotq) / r;
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
        auto vq = velocity_2();
        auto pg = gas_pressure();
        auto d = mass_density();
        auto r = spherical_radius;
        auto S = arithmetic_sequence_t<dimensional_value_t<-3, 1, -1, double>, 5>();
        S[1].value = (2.0 * pg + d * vq * vq) / r;
        return S;
    }




    /**
     * @brief      Data structure that computes the left and right eigenvectors
     *             of the flux Jacobian, and caches the variables involved to
     *             reduce redundant calculations.
     */
    struct eigensystem_formulas_t
    {


        //=====================================================================
        eigensystem_formulas_t(const primitive_t& p, double gamma_law_index)
        {
            g = gamma_law_index;
            m = gamma_law_index - 1;
            u = p.velocity_1();
            v = p.velocity_2();
            w = p.velocity_3();
            u2 = u * u;
            v2 = v * v;
            w2 = w * w;
            V2 = u2 + v2 + w2;
            a2 = p.sound_speed_squared(gamma_law_index);
            a = std::sqrt(a2);
            H = 0.5 * V2 + a2 / m;
        }




        /**
         * @brief      Return the Jacobian matrix dF / dU (Toro eqn. 3.79)
         *
         * @return     A 5x5 matrix
         *
         * @note       There is a typo in Toro (3rd edition) eqn. 3.79 in row 5,
         *             column 1 of A. The expression written is 0.5 * u * ((g -
         *             3) * H - a2), but it should read u * (0.5 * m * V2 - H)
         *             (where m = g - 1) as it does in the 2d version in eqn.
         *             3.70.
         */
        auto flux_jacobian() const
        {
            return matrix_t<unit_velocity<double>, 5, 5> {
                {{0,                      1,             0,          0,         0},
                 {m * H - u2 - a2,        (3 - g) * u,  -m * v,     -m * w,     m},
                 {-u * v,                 v,             u,          0,         0},
                 {-u * w,                 w,             0,          u,         0},
                 {u * (0.5 * m * V2 - H), H - m * u2,   -m * u * v, -m * u * w, g * u}}
            };
        }




        /**
         * @brief      Return the eigenvalues of the flux Jacobian
         *
         * @return     The eigenvalues
         */
        auto eigenvalues() const
        {
            return mara::diagonal_matrix<unit_velocity<double>>(u - a, u, u, u, u + a);
        }




        /**
         * @brief      Return the right eigenvectors of the Jacobian matrix
         *             (Toro eqn. 3.82)
         *
         * @return     The eigenvectors as a 5x5 matrix
         */
        auto right_eigenvectors() const
        {
            return matrix_t<unit_scalar<double>, 5, 5> {
                {{1,         1,        0,  0, 1},
                 {u - a,     u,        0,  0, u + a},
                 {v,         v,        1,  0, v},
                 {w,         w,        0,  1, w},
                 {H - u * a, 0.5 * V2, v,  w, H + u * a}}
            };
        }




        /**
         * @brief      Return the left eigenvectors of the Jacobian matrix (Toro
         *             eqn. 3.83)
         *
         * @return     The eigenvectors as a 5x5 matrix
         */
        auto left_eigenvectors() const
        {
            return matrix_t<unit_scalar<double>, 5, 5> {
                {{      H + (a / m) * (u - a), -(u + a / m), -v,         -w,           1, },
                 { -2 * H + (4 / m) * a2,             2 * u,  2 * v,      2 * w,      -2, },
                 {          -2 * v  * a2 / m,             0,  2 * a2 / m, 0,           0, },
                 {          -2 * w  * a2 / m,             0,  0,          2 * a2 / m,  0, },
                 {      H - (a / m) * (u + a), -(u - a / m), -v,         -w,           1, }},
            } * (m / 2 / a2);
        }




        //=====================================================================
        double g;
        double m;
        double u;
        double v;
        double w;
        double u2;
        double v2;
        double w2;
        double V2;
        double a2;
        double a;
        double H;
    };




    /**
     * @brief      Compute the flux Jacobian dF / dU
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     A 5x5 matrix
     */
    auto flux_jacobian(double gamma_law_index) const
    {
        return eigensystem_formulas_t(*this, gamma_law_index).flux_jacobian();
    }




    /**
     * @brief      Compute the eigenvalues of the flux Jacobian
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     The eigenvalues
     */
    auto eigenvalues(double gamma_law_index) const
    {
        return eigensystem_formulas_t(*this, gamma_law_index).eigenvalues();
    }




    /**
     * @brief      Compute the right eigenvectors of the flux Jacobian dF / dU
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     A 5x5 matrix
     */
    auto right_eigenvectors(double gamma_law_index) const
    {
        return eigensystem_formulas_t(*this, gamma_law_index).right_eigenvectors();
    }




    /**
     * @brief      Compute the left eigenvectors of the flux Jacobian dF / dU
     *
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     A 5x5 matrix
     */
    auto left_eigenvectors(double gamma_law_index) const
    {
        return eigensystem_formulas_t(*this, gamma_law_index).left_eigenvectors();
    }




    /**
     * @brief      Compute eigenvalues and left and right eigenvectors all at
     *             once (more efficient than calling the above functions
     *             on-by-one).
     *
     * @return     A tuple (eigenvalues, right, left)
     */
    auto eigensystem(double gamma_law_index) const
    {
        auto sys = eigensystem_formulas_t(*this, gamma_law_index);

        return std::make_tuple(sys.eigenvalues(), sys.right_eigenvectors(), sys.left_eigenvectors());
    }
};




/**
 * @brief      Attempt to recover a primitive variable state from the given
 *             vector of conserved densities.
 *
 * @param[in]  U                  The conserved densities
 * @param[in]  gamma_law_index    The gamma law index
 * @param[in]  temperature_floor  If greater than 0.0, sets p = max(p, T * rho)
 *
 * @return     A primitive variable state, if the recovery succeeds
 */
mara::mhd::primitive_t mara::mhd::recover_primitive(
    const conserved_density_t& U,
    double gamma_law_index,
    double temperature_floor)
{
    auto p_squared = (U[1] * U[1] + U[2] * U[2] + U[3] * U[3]).value;
    auto d = U[0].value;
    auto P = primitive_t();

    P[0] =  d;
    P[1] =  U[1].value / d;
    P[2] =  U[2].value / d;
    P[3] =  U[3].value / d;
    P[4] = (U[4].value - 0.5 * p_squared / d) * (gamma_law_index - 1.0);

    if (P[4] < 0.0 && temperature_floor > 0.0)
    {
        P[4] = temperature_floor * d;
    }
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
mara::mhd::primitive_t mara::mhd::roe_average(
    const primitive_t& Pr,
    const primitive_t& Pl)
{
    auto kr = std::sqrt(Pr.mass_density());
    auto kl = std::sqrt(Pl.mass_density());
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
mara::mhd::flux_vector_t mara::mhd::riemann_hlle(
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
