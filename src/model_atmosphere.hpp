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
#include <stdexcept>




//=============================================================================
namespace mara
{
    struct power_law_atmosphere_model;
    struct cloud_and_envelop_model;

    namespace detail
    {
        template<typename Function>
        double solve_secant(Function f,
            double starting_guess1,
            double starting_guess2,
            double tolerance);
    };
}




//=============================================================================
template<typename Function>
double mara::detail::solve_secant(Function f,
    double starting_guess1,
    double starting_guess2,
    double tolerance)
{
    double x1 = starting_guess1;
    double x2 = starting_guess2;
    double y1 = f(x1);
    double y2 = f(x2);

    while (std::abs(y2) > tolerance)
    {
        double x_next = x2 - y2 * (x2 - x1) / (y2 - y1);
        double y_next = f(x_next);

        x1 = x2;
        y1 = y2;
        x2 = x_next;
        y2 = y_next;
    }
    return x2;
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

    //=============================================================================
    double f0 = 1.0; /** coefficient (g / cm^3) */
    double r0 = 1.0; /** inner radius (cm) */
    double rc = 1e2; /** cutoff radius (cm) where index switches from n1 to n2 */
    double n1 = 2.0; /** power-law index where r < rc */
    double n2 = 6.0; /** power-law index where r > rc */
};




/**
 * @brief      A model describing an expanding gas cloud, with power-law density
 *             profile, surrounded by a balistic envelop. The envelop is
 *             described as an ensemble of balistic shells ejected
 *             simultaneously from the origin at t=0. The shell with mass dm
 *             contains gas with four-velocities between u and u + du. Thus the
 *             envelop profile at time t is given by the distribution du / dm or
 *             equivalently by u(m). Note that u(m) is a monotonically
 *             decreasing function of mass, so the slowest-moving shell is
 *             determined by the total envelop mass, nominally ~ 0.5% of a solar
 *             mass; u_cloud = u(m_envelop). The boundary between the cloud and
 *             the envelop lies at r_cloud = v_cloud * t.
 */
struct mara::cloud_and_envelop_model
{
    cloud_and_envelop_model with_inner_radius(double r0) const
    {
        auto result = *this; result.inner_radius = r0; return result;
    }

    cloud_and_envelop_model with_cloud_index(double n1) const
    {
        auto result = *this; result.cloud_index = n1; return result;
    }

    double gamma_beta(double m) const
    {
        return u1 * std::pow(m / m1, -psi);
    }

    double velocity(double m) const
    {
        double u = gamma_beta(m);
        return u / std::sqrt(1.0 + u * u) * light_speed;
    }

    double dudm(double m) const
    {
        return -psi / m * gamma_beta(m);
    }

    double radius(double m, double t) const
    {
        return velocity(m) * t;
    }

    double density(double m, double t) const
    {
        double gamma_squared = 1.0 + std::pow(gamma_beta(m), 2);
        double beta = velocity(m) / light_speed;
        return gamma_squared * beta / (4 * M_PI * std::pow(radius(m, t), 3)) / std::abs(dudm(m));
    }

    double cloud_gamma_beta() const
    {
        double beta = velocity(envelop_mass) / light_speed;
        return beta / std::sqrt(1.0 - beta * beta);
    }

    double cloud_velocity() const
    {
        return velocity(envelop_mass);
    }

    double cloud_outer_boundary(double t) const
    {
        return cloud_velocity() * t;
    }

    double envelop_outer_boundary(double t) const
    {
        return radius(m1, t);
    }

    double mass_coordinate(double r, double t) const
    {
        auto f = [this, r, t] (double m) { return std::log10(r) - std::log10(radius(m, t)); };
        return detail::solve_secant(f, m1, m1 * 2, 1e-10);
    }

    double power_law_cloud(double r, double t) const
    {
        double r_outer = cloud_outer_boundary(t);
        double d_outer = density_at(r_outer, t);
        return d_outer * std::pow(r / r_outer, -cloud_index);
    }

    double density_at(double r, double t) const
    {
        double r1 = envelop_outer_boundary(t);

        if (r < cloud_outer_boundary(t))
            return power_law_cloud(r, t);
        if (r > r1)
            return density_at(r1, t) * std::pow(r / r1, -2.0);
        return density(mass_coordinate(r, t), t);
    }

    double gamma_beta_at(double r, double t) const
    {
        double r1 = envelop_outer_boundary(t);

        if (r < cloud_outer_boundary(t))
            return cloud_gamma_beta();
        if (r > r1)
            return gamma_beta(mass_coordinate(r1, t));
        return gamma_beta(mass_coordinate(r, t));
    }

    double velocity_at(double r, double t) const
    {
        double u = gamma_beta_at(r, t);
        return u / std::sqrt(1.0 + u * u) * light_speed;
    }

    double cloud_mass(double t) const
    {
        double n1 = cloud_index;
        double r0 = inner_radius;
        double rc = cloud_outer_boundary(t);
        return n1 == 3.0
        ? 4 * M_PI * (density_at(rc, t) * std::pow(rc, 3) * std::log(rc / r0))
        : 4 * M_PI * (density_at(rc, t) * std::pow(rc, 3) - density_at(r0, t) * std::pow(r0, 3)) / (3 - n1);
    }

    double total_mass(double t) const
    {
        return cloud_mass(t) + envelop_mass;
    }

    //=========================================================================
    static constexpr double solar_mass  = 1.989e33; // g
    static constexpr double light_speed = 2.998e10; // cm / s
    double inner_radius = 3e8; // cm
    double envelop_mass = 0.005 * solar_mass;
    double u1 = 4.0;
    double m1 = 1e26;
    double psi = 0.25;
    double cloud_index = 2.0;
};
