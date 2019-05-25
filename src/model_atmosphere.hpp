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
    struct power_law_atmosphere_model;
}




/**
 * @brief      Model of a density profile that is a broken power-law in the
 *             spherical radius:
 *             
 *                      | f0 * (r / r0)^(-n1)           if r < rc
 *             rho(r) = |
 *                      | rho(rc) * (r / rc)^(-n2)      otherwise
 */
struct mara::power_law_atmosphere_model
{
    power_law_atmosphere_model with_coefficient  (double new_f0) const { return {new_f0, r0, rc, n1, n2}; }
    power_law_atmosphere_model with_inner_radius (double new_r0) const { return {f0, new_r0, rc, n1, n2}; }
    power_law_atmosphere_model with_cutoff_radius(double new_rc) const { return {f0, r0, new_rc, n1, n2}; }
    power_law_atmosphere_model with_inner_index  (double new_n1) const { return {f0, r0, rc, new_n1, n2}; }
    power_law_atmosphere_model with_outer_index  (double new_n2) const { return {f0, r0, rc, n1, new_n2}; }
    power_law_atmosphere_model with_total_mass(double new_total_mass) const
    {
        return with_coefficient(new_total_mass / total_mass());
    }

    double mass_within_cutoff() const
    {
        return n1 == 3.0
        ? 4 * M_PI * (density_at(rc) * std::pow(rc, 3) * std::log(rc / r0))
        : 4 * M_PI * (density_at(rc) * std::pow(rc, 3) - density_at(r0) * std::pow(r0, 3)) / (3 - n1);
    }

    double mass_beyond_cutoff() const
    {
        if (n2 <= 3.0)
        {
            throw std::invalid_argument("power_law_atmosphere: outer index (n2) must be greater than 3");
        }
        return 4 * M_PI * density_at(rc) * std::pow(rc, 3) / (n2 - 3);
    }

    double total_mass() const
    {
        return mass_within_cutoff() + mass_beyond_cutoff();
    }

    double density_at(double r) const
    {
        return r <= rc ? f0 * std::pow(r / r0, -n1) : density_at(rc) * std::pow(r / rc, -n2);
    }

    double f0 = 1.0; /** coefficient (g / cm^3) */
    double r0 = 1.0; /** inner radius (cm) */
    double rc = 1e2; /** cutoff radius (cm) where index switches from n1 to n2 */
    double n1 = 2.0; /** power-law index where r < rc */
    double n2 = 6.0; /** power-law index where r > rc */
};
