/**
 ==============================================================================
 Copyright 2019, Andrew MacFadyen and Jonathan Zrake

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
#include <stdexcept>
#include <algorithm>
#include <cmath>




//=============================================================================
namespace mara
{


    //=========================================================================
    struct orbital_elements_t
    {
        double separation     = 1.0;
        double total_mass     = 1.0;
        double mass_ratio     = 1.0;
        double eccentricity   = 0.0;
    };


    //=========================================================================
    struct full_orbital_elements_t
    {
        double pomega = 0.0;
        double phi = 0.0; // true anomoly + pi
        double cm_position_x = 0.0;
        double cm_position_y = 0.0;
        double cm_velocity_x = 0.0;
        double cm_velocity_y = 0.0;
        orbital_elements_t elements;

        inline full_orbital_elements_t operator+(const full_orbital_elements_t&) const;
        inline full_orbital_elements_t operator-(const full_orbital_elements_t&) const;
        inline full_orbital_elements_t operator*(double scale) const;
    };


    //=========================================================================
    struct point_mass_t
    {
        double mass       = 1.0;
        double position_x = 0.0;
        double position_y = 0.0;
        double velocity_x = 0.0;
        double velocity_y = 0.0;
    };


    //=========================================================================
    struct two_body_state_t
    {
        point_mass_t body1;
        point_mass_t body2;
    };


    //=========================================================================
    inline full_orbital_elements_t make_full_orbital_elements_with_zeros();
    inline two_body_state_t compute_two_body_state(const orbital_elements_t& params, double t);
    inline full_orbital_elements_t compute_orbital_elements(const two_body_state_t& two_body);
    inline double orbital_energy(orbital_elements_t elements);
    inline double orbital_angular_momentum(orbital_elements_t elements);
}




//=============================================================================
namespace mara::two_body::detail
{
    template<typename Function, typename Derivative>
    double solve_newton_rapheson(
        Function f,
        Derivative g,
        double starting_guess,
        double tolerance=1e-8);

    inline double wrap(double x0, double x1, double x);
    inline double clamp(double x0, double x1, double x);
}




//=============================================================================
template<typename Function, typename Derivative>
double mara::two_body::detail::solve_newton_rapheson(Function f,
    Derivative g,
    double starting_guess,
    double tolerance)
{
    double x = starting_guess;
    double y = f(x);

    while (std::abs(y) > tolerance)
    {
        x -= y / g(x);
        y = f(x);
    }
    return x;
}




double mara::two_body::detail::wrap(double x0, double x1, double x)
{
    if (x < x0)
        return wrap(x0, x1, x + (x1 - x0));
    if (x > x1)
        return wrap(x0, x1, x - (x1 - x0));
    return x;
}

double mara::two_body::detail::clamp(double x0, double x1, double x)
{
    return std::min(std::max(x, x0), x1);
}




//=============================================================================
mara::two_body_state_t mara::compute_two_body_state(const orbital_elements_t& params, double t)
{
    double e = params.eccentricity;
    double q = params.mass_ratio;
    double a = params.separation;
    double omega = a == 0.0 ? 0.0 : std::sqrt(params.total_mass / a / a / a);
    double mu = q / (1.0 + q);

    auto make_result_from_E_and_a = [&] (double E, double a)
    {
        // E is the eccentric anomoly (omega * t for circular orbits)
        // a is the separation
        auto result = two_body_state_t();
        result.body1.mass = params.total_mass * (1 - mu);
        result.body2.mass = params.total_mass * mu;
        result.body1.position_x = -a * mu * (e - std::cos(E));
        result.body1.position_y = +a * mu * (0 + std::sin(E)) * std::sqrt(1 - e * e);
        result.body2.position_x = -result.body1.position_x / q;
        result.body2.position_y = -result.body1.position_y / q;
        result.body1.velocity_x = -a * mu * omega / (1 - e * std::cos(E)) * std::sin(E);
        result.body1.velocity_y = +a * mu * omega / (1 - e * std::cos(E)) * std::cos(E) * std::sqrt(1 - e * e);
        result.body2.velocity_x = -result.body1.velocity_x / q;
        result.body2.velocity_y = -result.body1.velocity_y / q;
        return result;
    };

    if (params.eccentricity > 0.0)
    {
        auto M = omega * t; // mean anomoly
        auto anomoly_equation   = [e, M] (double E) { return E - e * std::sin(E) - M; };
        auto anomoly_derivative = [e]    (double E) { return 1 - e * std::cos(E); };

        double E = two_body::detail::solve_newton_rapheson(anomoly_equation, anomoly_derivative, M);
        return make_result_from_E_and_a(E, a);
    }
    else
    {
        double E = omega * t;
        double a = params.separation;
        return make_result_from_E_and_a(E, a);
    }
}




//=============================================================================
mara::full_orbital_elements_t mara::make_full_orbital_elements_with_zeros()
{
    auto result = full_orbital_elements_t{};
    result.elements.mass_ratio = 0.0;
    result.elements.separation = 0.0;
    result.elements.total_mass = 0.0;
    return result;
}




//=============================================================================
mara::full_orbital_elements_t mara::compute_orbital_elements(const two_body_state_t& two_body)
{
    auto c1 = two_body.body1;
    auto c2 = two_body.body2;


    double M1 = c1.mass;
    double M2 = c2.mass;
    double M = M1 + M2;
    double q = M2 / M1;


    // The position and velocity of the CM frame
    double x_cm  = (c1.position_x * c1.mass + c2.position_x * c2.mass) / M;
    double y_cm  = (c1.position_y * c1.mass + c2.position_y * c2.mass) / M;
    double vx_cm = (c1.velocity_x * c1.mass + c2.velocity_x * c2.mass) / M;
    double vy_cm = (c1.velocity_y * c1.mass + c2.velocity_y * c2.mass) / M;


    // The positions and velocities of the components in the CM frame
    double x1 = c1.position_x - x_cm;
    double y1 = c1.position_y - y_cm;
    double x2 = c2.position_x - x_cm;
    double y2 = c2.position_y - y_cm;
    double r1 = std::sqrt(x1 * x1 + y1 * y1);
    double r2 = std::sqrt(x2 * x2 + y2 * y2);
    double vx1 = c1.velocity_x - vx_cm;
    double vy1 = c1.velocity_y - vy_cm;
    double vx2 = c2.velocity_x - vx_cm;
    double vy2 = c2.velocity_y - vy_cm;
    double vf1 = -vx1 * y1 / r1 + vy1 * x1 / r1;
    double vf2 = -vx2 * y2 / r2 + vy2 * x2 / r2;


    // Energy and angular momentum variables
    double E1 = 0.5 * M1 * (vx1 * vx1 + vy1 * vy1);
    double E2 = 0.5 * M2 * (vx2 * vx2 + vy2 * vy2);
    double L1 = M1 * r1 * vf1;
    double L2 = M2 * r2 * vf2;
    double R = r1 + r2;
    double L = L1 + L2;
    double E = E1 + E2 - M1 * M2 / R;


    // Semi-major, semi-minor axes; eccentricity, apsides
    double a = -0.5 * M1 * M2 / E;
    double b = std::sqrt(-0.5 * L * L / E * (M1 + M2) / (M1 * M2));
    double e = std::sqrt(1.0 - std::min(1.0, b * b / a / a));
    // double rmin = a - std::sqrt(a * a - b * b);
    // double rmax = a + std::sqrt(a * a - b * b);


    // Argument of periapsis (pomega)
    double a2 = a / (1.0 + q);
    double b2 = b / (1.0 + q);
    double cf = e == 0.0 ? -1.0 : -(a2 - r2) / (a2 * e);
    double delta = std::atan2(b2 * std::sqrt(std::max(0.0, 1.0 - cf * cf)), a2 * (e + cf));
    double pomega = std::atan2(y2, x2) - delta;


    // Store results
    auto P = full_orbital_elements_t();
    P.phi = std::acos(two_body::detail::clamp(-1.0, 1.0, cf));
    P.pomega = two_body::detail::wrap(0.0, 2 * M_PI, pomega);
    P.cm_position_x = x_cm;
    P.cm_position_y = y_cm;
    P.cm_velocity_x = vx_cm;
    P.cm_velocity_y = vy_cm;
    P.elements.separation = a;
    P.elements.total_mass = M;
    P.elements.mass_ratio = q;
    P.elements.eccentricity = e;


    // Check that it was a valid orbit
    if (E >= 0.0)
    {
        throw std::invalid_argument("mara::compute_orbital_elements "
            "(two_body_state does not correspond to a bound orbit)");
    }

    return P;
}

double mara::orbital_energy(orbital_elements_t elements)
{
    double a = elements.separation;
    double q = elements.mass_ratio;
    double M = elements.total_mass;
    double M1 = M / (1 + q);
    double M2 = M - M1;
    double E = -0.5 * M1 * M2 / a;
    return E;
}

double mara::orbital_angular_momentum(orbital_elements_t elements)
{
    double a = elements.separation;
    double q = elements.mass_ratio;
    double e = elements.eccentricity;
    double M = elements.total_mass;
    double M1 = M / (1 + q);
    double M2 = M - M1;
    double mu = M1 * M2 / M;
    double b2 = a * a * (1.0 - e * e);
    double L2 = -2.0 * orbital_energy(elements) * b2 * mu;
    return std::sqrt(L2);
}

mara::full_orbital_elements_t mara::full_orbital_elements_t::operator+(const full_orbital_elements_t& other) const
{
    return {
        pomega + other.pomega,
        phi + other.phi,
        cm_position_x + other.cm_position_x,
        cm_position_y + other.cm_position_y,
        cm_velocity_x + other.cm_velocity_x,
        cm_velocity_y + other.cm_velocity_y,
        {
            elements.separation + other.elements.separation,
            elements.total_mass + other.elements.total_mass,
            elements.mass_ratio + other.elements.mass_ratio,
            elements.eccentricity + other.elements.eccentricity,
        },
    };
}

mara::full_orbital_elements_t mara::full_orbital_elements_t::operator-(const full_orbital_elements_t& other) const
{
    return {
        pomega - other.pomega,
        phi - other.phi,
        cm_position_x - other.cm_position_x,
        cm_position_y - other.cm_position_y,
        cm_velocity_x - other.cm_velocity_x,
        cm_velocity_y - other.cm_velocity_y,
        {
            elements.separation - other.elements.separation,
            elements.total_mass - other.elements.total_mass,
            elements.mass_ratio - other.elements.mass_ratio,
            elements.eccentricity - other.elements.eccentricity,
        },
    };
}

mara::full_orbital_elements_t mara::full_orbital_elements_t::operator*(double scale) const
{
    return {
        pomega * scale,
        phi * scale,
        cm_position_x * scale,
        cm_position_y * scale,
        cm_velocity_x * scale,
        cm_velocity_y * scale,
        {
            elements.separation * scale,
            elements.total_mass * scale,
            elements.mass_ratio * scale,
            elements.eccentricity * scale,
        },
    };
}
