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




//=============================================================================
namespace mara
{
    struct jet_nozzle_model;
}




/**
 * @brief      Model of an ultra-relativistic, cold jet inflow with a
 *             Gaussian-like angular structure,
 *
 *             L(q, t) = dj G0^2 r0^2 c^3 exp(-(q / qj)^as) exp(-t / tj)
 *
 *             where L(q, t) is the luminosity per steradian at polar angle
 *             q and time t. Here, dj is the comoving mass density at the
 *             jet base, determined from the total energy and base Lorentz
 *             factor G0.
 */
struct mara::jet_nozzle_model
{
    static constexpr double light_speed_cgs = 3e10;

    jet_nozzle_model with_total_energy      (double new_Ej) const { return {new_Ej, G0, tj, qj, as, r0}; }
    jet_nozzle_model with_lorentz_factor    (double new_G0) const { return {Ej, new_G0, tj, qj, as, r0}; }
    jet_nozzle_model with_jet_duration      (double new_tj) const { return {Ej, G0, new_tj, qj, as, r0}; }
    jet_nozzle_model with_opening_angle     (double new_qj) const { return {Ej, G0, tj, new_qj, as, r0}; }
    jet_nozzle_model with_structure_exponent(double new_as) const { return {Ej, G0, tj, qj, new_as, r0}; }
    jet_nozzle_model with_inner_radius      (double new_r0) const { return {Ej, G0, tj, qj, as, new_r0}; }


    /**
     * @brief      Return the luminosity per steradian of each jet.
     *
     * @param[in]  q     The polar angle
     * @param[in]  t     The time
     *
     * @return     The luminosity (erg / s / Sr)
     */
    double luminosity_per_steradian(double q, double t) const
    {
        return density_at_base() *
        std::pow(G0, 2) *
        std::pow(r0, 2) *
        std::pow(light_speed_cgs, 3) *
        std::exp(-std::pow(q / qj, as)) * std::exp(-t / tj);
    }


    /**
     * @brief      Return the jet gamma-beta (also the Lorentz factor given
     *             the ultra-relativistic assumption) at the jet base at the
     *             given polar angle q and time t. If including the
     *             counter-jet as well, this should be called as
     *             lorentz_factor(q) + lorentz_factor(M_PI - q).
     *
     * @param[in]  q     The polar angle
     * @param[in]  t     The time
     *
     * @return     The Lorentz factor
     */
    double gamma_beta(double q, double t) const
    {
        return G0 *
        std::exp(-0.5 * std::pow(q / qj, as)) *
        std::exp(-0.5 * t / tj);
    }


    /**
     * @brief      Estimate the comoving mass density at the jet base (r0)
     *             necessary for the jet (plus counter-jet) to have the
     *             total energy.
     *
     * @return     The density (g / cm^3)
     *
     * @note       This is estimate is accurate when the jet is cold (h - 1
     *             << 1) and ultra-relativistic (G0 >> 1), and when the
     *             structure exponent (as) is 2. Expect errors at the ~10%
     *             level for different values of as.
     */
    double density_at_base() const
    {
        return Ej / (2 * M_PI * std::pow(G0 * r0 * qj, 2) * tj * std::pow(light_speed_cgs, 3));
    }

    double Ej = 1.0; /** total explosion energy (erg) */
    double G0 = 2.0; /** Lorentz factor on-axis and at t=0 */
    double tj = 1.0; /** engine duration (s) */
    double qj = 0.1; /** engine opening angle (radian) */
    double as = 2.0; /** structure exponent */
    double r0 = 1.0; /** inner radius */
};
