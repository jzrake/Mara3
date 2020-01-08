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
        double pomega        = 0.0; // argument of periapse
        double tau           = 0.0; // time of last periapse
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
    inline full_orbital_elements_t make_full_orbital_elements(orbital_elements_t);
    inline two_body_state_t compute_two_body_state(const orbital_elements_t& params, double t);

    inline double total_energy(two_body_state_t s);
    inline double total_mass(two_body_state_t s);
    inline double separation(two_body_state_t s);
    inline double delta_a_over_a(two_body_state_t s2, two_body_state_t s1);
    inline double mean_anomaly(const full_orbital_elements_t& params, double t);

    inline two_body_state_t compute_two_body_state(const full_orbital_elements_t& params, double t);
    inline full_orbital_elements_t compute_orbital_elements(const two_body_state_t& two_body, double t);

    inline double orbital_energy(orbital_elements_t elements);
    inline double orbital_period(orbital_elements_t elements);
    inline double orbital_angular_momentum(orbital_elements_t elements);
    inline full_orbital_elements_t diff(const full_orbital_elements_t& a, const full_orbital_elements_t& b);
    inline full_orbital_elements_t diff_cm(const full_orbital_elements_t& a, double dt);
}




//=============================================================================
namespace mara::two_body::detail
{
    template<typename Function, typename Derivative>
    double solve_newton_rapheson(
        Function f,
        Derivative g,
        double starting_guess,
        double tolerance=1e-10);

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

mara::two_body_state_t mara::compute_two_body_state(const full_orbital_elements_t& params, double t)
{
    while (t < params.tau)
    {
        t += orbital_period(params.elements);
    }
    auto local = compute_two_body_state(params.elements, t - params.tau);

    auto x1 = local.body1.position_x;
    auto y1 = local.body1.position_y;
    auto x2 = local.body2.position_x;
    auto y2 = local.body2.position_y;

    auto vx1 = local.body1.velocity_x;
    auto vy1 = local.body1.velocity_y;
    auto vx2 = local.body2.velocity_x;
    auto vy2 = local.body2.velocity_y;

    auto c = std::cos(-params.pomega);
    auto s = std::sin(-params.pomega);

    auto x1_rot = +x1 * c + y1 * s;
    auto y1_rot = -x1 * s + y1 * c;
    auto x2_rot = +x2 * c + y2 * s;
    auto y2_rot = -x2 * s + y2 * c;

    auto vx1_rot = +vx1 * c + vy1 * s;
    auto vy1_rot = -vx1 * s + vy1 * c;
    auto vx2_rot = +vx2 * c + vy2 * s;
    auto vy2_rot = -vx2 * s + vy2 * c;

    auto x1_rot_trans = x1_rot + params.cm_position_x;
    auto y1_rot_trans = y1_rot + params.cm_position_y;
    auto x2_rot_trans = x2_rot + params.cm_position_x;
    auto y2_rot_trans = y2_rot + params.cm_position_y;

    auto vx1_rot_trans = vx1_rot + params.cm_velocity_x;
    auto vy1_rot_trans = vy1_rot + params.cm_velocity_y;
    auto vx2_rot_trans = vx2_rot + params.cm_velocity_x;
    auto vy2_rot_trans = vy2_rot + params.cm_velocity_y;

    return {
        {
            local.body1.mass,
            x1_rot_trans,
            y1_rot_trans,
            vx1_rot_trans,
            vy1_rot_trans,
        },
        {
            local.body2.mass,
            x2_rot_trans,
            y2_rot_trans,
            vx2_rot_trans,
            vy2_rot_trans, 
        },
    };
}




//=============================================================================
mara::full_orbital_elements_t mara::make_full_orbital_elements_with_zeros()
{
    auto result = full_orbital_elements_t{};
    result.elements.mass_ratio = 0.0;
    result.elements.separation = 0.0;
    result.elements.total_mass = 0.0;
    result.elements.eccentricity = 0.0;
    return result;
}

mara::full_orbital_elements_t mara::make_full_orbital_elements(orbital_elements_t E)
{
    auto result = full_orbital_elements_t();
    result.elements = E;
    return result;
}




//=============================================================================
mara::full_orbital_elements_t mara::compute_orbital_elements(const two_body_state_t& two_body, double t)
{
    using two_body::detail::clamp;


    auto c1 = two_body.body1;
    auto c2 = two_body.body2;


    // component masses, total mass, and mass ratio
    double M1 = c1.mass;
    double M2 = c2.mass;
    double M = M1 + M2;
    double q = M2 / M1;


    // position and velocity of the CM frame
    double x_cm  = (c1.position_x * c1.mass + c2.position_x * c2.mass) / M;
    double y_cm  = (c1.position_y * c1.mass + c2.position_y * c2.mass) / M;
    double vx_cm = (c1.velocity_x * c1.mass + c2.velocity_x * c2.mass) / M;
    double vy_cm = (c1.velocity_y * c1.mass + c2.velocity_y * c2.mass) / M;


    // positions and velocities of the components in the CM frame
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
    double v1 = std::sqrt(vx1 * vx1 + vy1 * vy1);


    // energy and angular momentum
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
    double e = std::sqrt(clamp(0.0, 1.0, 1.0 - b * b / a / a));
    double omega = std::sqrt(M / a / a / a);


    // semi-major and semi-minor axes of the primary
    double a1 = a * q / (1.0 + q);
    double b1 = b * q / (1.0 + q);


    // cos of nu and f: phase angle and true anomaly
    double cn = e == 0.0 ? x1 / r1 : (1.0 - r1 / a1) / e;
    double cf = a1 / r1 * (cn - e);


    // sin of nu and f
    double sn = e == 0.0 ? y1 / r1 : (vx1 * x1 + vy1 * y1) / (e * v1 * r1) * std::sqrt(1.0 - e * e * cn * cn);
    double sf = (b1 / r1) * sn;


    // cos and sin of eccentric anomaly
    double cE = (e + cf)                    / (1.0 + e * cf);
    double sE = std::sqrt(1.0 - e * e) * sf / (1.0 + e * cf);


    // mean anomaly and tau
    double EE = std::atan2(sE, cE);
    double MM = EE - e * sE;
    double tau = t - MM / omega;


    // cartesian components of semi-major axis, and the argument of periapse
    // double ax = -x1 - y1 * sf / cf;
    // double ay = -y1 + x1 * sf / cf;
    double ax = +(cn - e) * x1 + sn * std::sqrt(1.0 - e * e) * y1;
    double ay = +(cn - e) * y1 - sn * std::sqrt(1.0 - e * e) * x1;
    double pomega = std::atan2(ay, ax);


    if (E >= 0.0)
        throw std::invalid_argument("mara::compute_orbital_elements (two_body_state does not correspond to a bound orbit)");


    //=========================================================================
    auto P = full_orbital_elements_t();
    P.tau = tau;
    P.pomega = pomega;
    P.cm_position_x = x_cm;
    P.cm_position_y = y_cm;
    P.cm_velocity_x = vx_cm;
    P.cm_velocity_y = vy_cm;
    P.elements.separation = a;
    P.elements.total_mass = M;
    P.elements.mass_ratio = q;
    P.elements.eccentricity = e;
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

double mara::orbital_period(orbital_elements_t elements)
{    
    auto M = elements.total_mass;
    auto a = elements.separation;
    return 2 * M_PI / std::sqrt(M / a / a / a);
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

double mara::total_energy(two_body_state_t s)
{
    double T1 = 0.5 * s.body1.mass * (s.body1.velocity_x * s.body1.velocity_x + s.body1.velocity_y * s.body1.velocity_y);
    double T2 = 0.5 * s.body2.mass * (s.body2.velocity_x * s.body2.velocity_x + s.body2.velocity_y * s.body2.velocity_y);
    double U12 = -s.body1.mass * s.body2.mass / separation(s);
    return T1 + T2 + U12;
}

double mara::total_mass(two_body_state_t s)
{
    return s.body1.mass + s.body2.mass;
}

double mara::separation(two_body_state_t s)
{
    return std::sqrt(std::pow(s.body1.position_x - s.body2.position_x, 2) + std::pow(s.body1.position_y - s.body2.position_y, 2));
}

double mara::delta_a_over_a(two_body_state_t s2, two_body_state_t s1)
{
    double E = total_energy(s1);
    double M1 = s1.body1.mass;
    double M2 = s1.body2.mass;
    double dM1 = s2.body1.mass - s1.body1.mass;
    double dM2 = s2.body2.mass - s1.body2.mass;

    double ax1 = s2.body1.velocity_x - s1.body1.velocity_x;
    double ay1 = s2.body1.velocity_y - s1.body1.velocity_y;
    double ax2 = s2.body2.velocity_x - s1.body2.velocity_x;
    double ay2 = s2.body2.velocity_y - s1.body2.velocity_y;

    double vx1 = s1.body1.velocity_x;
    double vy1 = s1.body1.velocity_y;
    double vx2 = s1.body2.velocity_x;
    double vy2 = s1.body2.velocity_y;

    double T1 = 0.5 * M1 * (vx1 * vx1 + vy1 * vy1);
    double T2 = 0.5 * M2 * (vx2 * vx2 + vy2 * vy2);
    double dT1 = M1 * (ax1 * vx1 + ay1 * vy1);
    double dT2 = M2 * (ax2 * vx2 + ay2 * vy2);

    return (T2 * dM1 / M1 + T1 * dM2 / M2) / E - (dT1 + dT2) / E;
}

double mara::mean_anomaly(const full_orbital_elements_t& params, double t)
{
    auto P = orbital_period(params.elements);

    while (t < params.tau)
    {
        t += P;
    }
    auto omega = 2 * M_PI / P;
    return omega * t;
}

mara::full_orbital_elements_t mara::diff(const full_orbital_elements_t& a, const full_orbital_elements_t& b)
{
    auto wrap = [] (auto delta, auto period)
    {
        auto a = delta;
        auto b = delta + period;
        auto c = delta - period;

        if (std::abs(a) < std::min(std::abs(b), std::abs(c)))
            return a;
        if (std::abs(b) < std::abs(c))
            return b;
        return c;
    };

    return {
        wrap(b.pomega - a.pomega, 2 * M_PI),
        wrap(b.tau    - a.tau, orbital_period(b.elements)),
        b.cm_position_x - a.cm_position_x,
        b.cm_position_y - a.cm_position_y,
        b.cm_velocity_x - a.cm_velocity_x,
        b.cm_velocity_y - a.cm_velocity_y,
        {
            b.elements.separation   - a.elements.separation,
            b.elements.total_mass   - a.elements.total_mass,
            b.elements.mass_ratio   - a.elements.mass_ratio,
            b.elements.eccentricity - a.elements.eccentricity,
        },
    };
}

mara::full_orbital_elements_t mara::diff_cm(const full_orbital_elements_t& a, double dt)
{
    auto result = make_full_orbital_elements_with_zeros();
    result.cm_position_x = a.cm_velocity_x * dt;
    result.cm_position_y = a.cm_velocity_y * dt;
    return result;
}

mara::full_orbital_elements_t mara::full_orbital_elements_t::operator+(const full_orbital_elements_t& other) const
{
    return {
        pomega        + other.pomega,
        tau           + other.tau,
        cm_position_x + other.cm_position_x,
        cm_position_y + other.cm_position_y,
        cm_velocity_x + other.cm_velocity_x,
        cm_velocity_y + other.cm_velocity_y,
        {
            elements.separation   + other.elements.separation,
            elements.total_mass   + other.elements.total_mass,
            elements.mass_ratio   + other.elements.mass_ratio,
            elements.eccentricity + other.elements.eccentricity,
        },
    };
}

mara::full_orbital_elements_t mara::full_orbital_elements_t::operator-(const full_orbital_elements_t& other) const
{
    return {
        pomega - other.pomega,
        tau    - other.tau,
        cm_position_x - other.cm_position_x,
        cm_position_y - other.cm_position_y,
        cm_velocity_x - other.cm_velocity_x,
        cm_velocity_y - other.cm_velocity_y,
        {
            elements.separation   - other.elements.separation,
            elements.total_mass   - other.elements.total_mass,
            elements.mass_ratio   - other.elements.mass_ratio,
            elements.eccentricity - other.elements.eccentricity,
        },
    };
}

mara::full_orbital_elements_t mara::full_orbital_elements_t::operator*(double scale) const
{
    return {
        pomega * scale,
        tau    * scale,
        cm_position_x * scale,
        cm_position_y * scale,
        cm_velocity_x * scale,
        cm_velocity_y * scale,
        {
            elements.separation   * scale,
            elements.total_mass   * scale,
            elements.mass_ratio   * scale,
            elements.eccentricity * scale,
        },
    };
}
