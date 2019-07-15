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
    using unit_conserved_density  = dimensional_value_t< -3, 1, 0, double>;
    using unit_conserved          = dimensional_value_t<  0, 1, 0, double>;
    using unit_flow               = dimensional_value_t<  0, 1,-1, double>;
    using unit_flux               = dimensional_value_t< -2, 1,-1, double>; //flux through faces

    using conserved_density_t     = arithmetic_sequence_t<unit_conserved_density, 8>; 
    using conserved_t             = arithmetic_sequence_t<unit_conserved        , 8>;
    using flux_vector_t           = arithmetic_sequence_t<unit_flux             , 8>;
    
    struct primitive_t;

    struct wavespeeds_t
    {
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
     * @brief      Return the magnitude of the velocity.
     *
     * @return     |u|
     */
    double velocity_magnitude() const
    {
        return std::sqrt( velocity_squared() );
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
     * @return     B^2
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
     * @return     B /e ?   ------> need to think about and fix these units everywhere
     */
    double bfield_along(const unit_vector_t& nhat) const
    {
        const auto &_ = *this;
        return nhat.project(_[5], _[6], _[7]);
    }

     /**
     * @brief      Return the dot product of the bfield and velocity vectors
     *
     * @return     B \dot v
     */
    double bfield_dot_velocity() const
    {
        //Coordinate correspondence between 1 and 5; 2 and 6; 3 and 7 (e.g. vx and Bx)
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
     * @return     B^2 / rho
     */
    double alfven_speed_squared() const
    {
        return bfield_squared() /  mass_density();
    }

   /**
     * @brief      Return the squared-Alfven-speed along some direction
     *
     * @param[in]  nhat  The unit vector
     *
     * @return     (B_nhat)^2 / rho
     */
    double alfven_speed_squared_along(const unit_vector_t& nhat) const
    {
        const auto b_along = bfield_along(nhat);  
        return b_along*b_along / mass_density();
    }

   /**
     * @brief      Return the fast magnetosonic wavespeed squared along nhat
     *
     * @param[in]  nhat             The unit vector
     * 
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     0.5 * ( c_s^2 + c_a^2 ) + 0.5*\sqrt{ (c_s^2+c_a^2)^2 - 4*c_s^2*c_{a,i}^2 }
     */
    double magnetosonic_speed_squared_fast(const unit_vector_t& nhat, double gamma_law_index) const
    {
        const auto cs2 = sound_speed_squared(gamma_law_index);
        const auto one = cs2 + alfven_speed_squared();
        const auto two = 0.5 * std::sqrt(one * one - 4 * cs2 * alfven_speed_squared_along(nhat));

        return 0.5 * one + two;
    }


   /**
     * @brief      Return the slow magnetosonic wavespeed squared along nhat
     *
     * @param[in]  nhat             The unit vector
     * 
     * @param[in]  gamma_law_index  The gamma law index
     *
     * @return     0.5 * ( c_s^2 + c_a^2 ) - 0.5*\sqrt{ (c_s^2+c_a^2)^2 - 4*c_s^2*c_{a,i}^2 }
     */
    double magnetosonic_speed_squared_slow(const unit_vector_t& nhat, double gamma_law_index) const
    {
        const auto cs2 = sound_speed_squared(gamma_law_index);
        const auto one = cs2 + alfven_speed_squared();
        const auto two = 0.5 * std::sqrt(one * one - 4 * cs2 * alfven_speed_squared_along(nhat));

        return 0.5 * one - two;
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
        U[4].value = 0.5 * d * velocity_squared() + p / (gamma_law_index - 1) /*mhd*/ + bfield_squared() / 2.0;

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
        auto d  = mass_density();
        auto v  = velocity_along(nhat);
        auto b  = bfield_along(nhat);
        auto p  = gas_pressure();
        auto p_tot = p + bfield_squared() / 2.0;
        auto bv = bfield_dot_velocity();
        auto F  = flux_vector_t();
        F[0].value = v * U[0].value;
        F[1].value = v * U[1].value + p_tot * nhat.get_n1() /*mhd*/ - U[5].value * b;
        F[2].value = v * U[2].value + p_tot * nhat.get_n2() /*mhd*/ - U[6].value * b;
        F[3].value = v * U[3].value + p_tot * nhat.get_n3() /*mhd*/ - U[7].value * b; 
        F[4].value = v * U[4].value + p_tot * v             /*mhd*/ - bv         * b;

        // mhd: B \cross v
        F[5].value = v * U[5].value - U[1].value / d * b;
        F[6].value = v * U[6].value - U[2].value / d * b;  
        F[7].value = v * U[7].value - U[3].value / d * b;

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
    wavespeeds_t sound_wave_speeds(const unit_vector_t& nhat, double gamma_law_index) const
    {
        auto cs = std::sqrt(sound_speed_squared(gamma_law_index));
        auto vn = velocity_along(nhat);
        return {
            make_velocity(vn - cs),
            make_velocity(vn + cs),
        };
    }

    wavespeeds_t alfven_wave_speeds(const unit_vector_t& nhat, double gamma_law_index) const
    {
        auto ca = std::sqrt( alfven_speed_squared_along(nhat) );
        auto vn = velocity_along(nhat);
        return{
            make_velocity(vn - ca),
            make_velocity(vn + ca)
        };
    }

    wavespeeds_t fast_wave_speeds(const unit_vector_t& nhat, double gamma_law_index) const
    {
        auto cf = std::sqrt( magnetosonic_speed_squared_fast(nhat, gamma_law_index) );
        auto vn = velocity_along(nhat);
        return{
            make_velocity(vn - cf),
            make_velocity(vn + cf)
        };
    }

    wavespeeds_t slow_wave_speeds(const unit_vector_t& nhat, double gamma_law_index) const
    {
        auto cs = std::sqrt( magnetosonic_speed_squared_slow(nhat, gamma_law_index) );
        auto vn = velocity_along(nhat);
        return{
            make_velocity(vn - cs),
            make_velocity(vn + cs)
        };
    }

    /**
     * @brief      Above function generalized to other wave speeds
     *
     * @param[in]  nhat         The direction
     * @param[in]  wavespeed    The wavespeed of the wave of interest
     *
     * @return     The wavespeeds
     */
    wavespeeds_t wavespeeds_general( const unit_vector_t& nhat, double wavespeed ) const
    {
        auto c0 = std::sqrt(wavespeed);
        auto vn = velocity_along(nhat);
        return {
            make_velocity(vn - c0),
            make_velocity(vn + c0)
        };
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
    auto b_squared = (U[5] * U[5] + U[6] * U[6] + U[7] * U[7]).value;
    auto d = U[0].value;
    auto P = primitive_t();

    P[0] =  d;
    P[1] =  U[1].value / d;
    P[2] =  U[2].value / d;
    P[3] =  U[3].value / d;
    P[4] = (U[4].value - 0.5 * p_squared / d - 0.5 * b_squared) * (gamma_law_index - 1.0);
    P[5] =  U[5].value;
    P[6] =  U[6].value;
    P[7] =  U[7].value;

    if (P[4] < 0.0 && temperature_floor > 0.0)
    {
        P[4] = temperature_floor * d;
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
mara::mhd::flux_vector_t mara::mhd::riemann_hlle(
    const primitive_t& Pl,
    const primitive_t& Pr,
    const unit_vector_t& nhat,
    double gamma_law_index)
{
    auto Ul = Pl.to_conserved_density(gamma_law_index);
    auto Ur = Pr.to_conserved_density(gamma_law_index);
    auto Al = Pl.fast_wave_speeds(nhat, gamma_law_index ); 
    auto Ar = Pr.fast_wave_speeds(nhat, gamma_law_index );
    auto Fl = Pl.flux(nhat, Ul);
    auto Fr = Pr.flux(nhat, Ur);

    auto ap = std::max(make_velocity(0.0), std::max(Al.p, Ar.p));
    auto am = std::min(make_velocity(0.0), std::min(Al.m, Ar.m));

    return (Fl * ap - Fr * am - (Ul - Ur) * ap * am) / (ap - am);
}
