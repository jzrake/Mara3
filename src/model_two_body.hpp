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
#include <cmath>




//=============================================================================
namespace mara
{



    //=========================================================================
    struct two_body_parameters_t
    {
        double separation     = 1.0;
        double total_mass     = 1.0;
        double eccentricity   = 0.0;
        double mass_ratio     = 1.0;
    };




    //=========================================================================
    struct point_mass_t
    {
        double mass       = 1.0;
        double position_x = 0.0;
        double position_y = 0.0;
        double position_z = 0.0;
    };




    //=========================================================================
    struct two_body_state_t
    {
        point_mass_t body1;
        point_mass_t body2;
    };




    //=========================================================================
    inline two_body_state_t compute_two_body_state(const two_body_parameters_t& params, double t);
}




//=============================================================================
namespace mara::two_body::detail
{
    template<typename Function, typename Derivative>
    double solve_newton_rapheson(Function f, Derivative g, double starting_guess, double tolerance=1e-8);
}




//=============================================================================
template<typename Function, typename Derivative>
double mara::two_body::detail::solve_newton_rapheson(Function f, Derivative g, double starting_guess, double tolerance)
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




//=============================================================================
mara::two_body_state_t mara::compute_two_body_state(const two_body_parameters_t& params, double t)
{
    double e = params.eccentricity;
    double q = params.mass_ratio;
    double a = params.separation;
    double omega_binary = a == 0.0 ? 0.0 : std::sqrt(params.total_mass / a / a / a);
    double mu = q / (1.0 + q);

    auto make_result_from_f_and_r = [&] (double f, double r)
    {
        // f is the true anomoly (omega * t for circular orbits)
        // r is the effective separation (a for circular orbits)
        auto result = two_body_state_t();
        result.body1.mass = params.total_mass * (1 - mu);
        result.body2.mass = params.total_mass * mu;
        result.body1.position_x = mu * r * std::cos(f);
        result.body1.position_y = mu * r * std::sin(f);
        result.body2.position_x = mu * r * std::cos(f + M_PI) / q;
        result.body2.position_y = mu * r * std::sin(f + M_PI) / q;
        return result;
    };

    if (params.eccentricity > 0.0)
    {
        auto M = omega_binary * t; // mean anomoly
        auto anomoly_equation   = [e, M] (double E) { return E - e * std::sin(E) - M; };
        auto anomoly_derivative = [e]    (double E) { return 1 - e * std::cos(E); };

        double E = two_body::detail::solve_newton_rapheson(anomoly_equation, anomoly_derivative, M);
        double x = a * (std::cos(E) - e);
        double y = a * (std::sin(E) * std::sqrt(1.0 - e * e));
        double r = std::sqrt(x * x + y * y);
        double f = std::atan2(y, x);
        return make_result_from_f_and_r(f, r);
    }
    else
    {
        double f = omega_binary * t;
        double r = params.separation;
        return make_result_from_f_and_r(f, r);
    }
}
