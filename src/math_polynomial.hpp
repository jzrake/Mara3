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
#include <algorithm> // std::max
#include <cmath>     // std::sqrt, std::pow, std::sin, std::cos, etc




//=============================================================================
namespace mara
{
    struct cubic_roots_t
    {
        double x1, x2, x3;
        int num_roots;
    };

    struct quartic_roots_t
    {
        double x1, x2, x3, x4;
        bool pair_1_is_real;
        bool pair_2_is_real;
    };

    inline cubic_roots_t cubic_roots(double c3, double c2, double c1, double c0);
    inline quartic_roots_t quartic_roots(double d4, double d3, double d2, double d1, double d0);
    inline std::pair<double, double> parabola_vertex(double x1, double x2, double x3, double y1, double y2, double y3);
}




/**
 * @brief      Compute the zeros of a cubic polynomial given by sum(c_n x^n)
 *
 * @param[in]  c3    The coefficient to x^3
 * @param[in]  c2    The coefficient to x^2
 * @param[in]  c1    The coefficient to x^1
 * @param[in]  c0    The coefficient to x^0
 *
 * @return     The real root(s), including the number that exist (either 1 or 3)
 */
mara::cubic_roots_t mara::cubic_roots(double c3, double c2, double c1, double c0)
{
    auto a2 = c2 / c3;
    auto a1 = c1 / c3;
    auto a0 = c0 / c3;
    auto q = a1 / 3.0 - a2 * a2 / 9.0;
    auto r = (a1 * a2 - 3.0 * a0) / 6.0 - a2 * a2 * a2 / 27.0;
    auto delta = q * q * q + r * r;
    auto result = cubic_roots_t();

    if (delta > 0.0)
    {
        double s1 = r + std::sqrt(delta);
        double s2 = r - std::sqrt(delta);
        s1 = (s1 >= 0.0) ? std::pow(s1, 1. / 3) : -std::pow(-s1, 1. / 3);
        s2 = (s2 >= 0.0) ? std::pow(s2, 1. / 3) : -std::pow(-s2, 1. / 3);
        result.x1 = (s1 + s2) - a2 / 3.0;
        result.x2 = -0.5 * (s1 + s2) - a2 / 3.0;
        result.x3 = -0.5 * (s1 + s2) - a2 / 3.0;
        result.num_roots = 1;
    }
    else if (delta < 0.0)
    {
        auto theta = std::acos(r / std::sqrt(-q * q * q)) / 3.0;
        auto costh = std::cos(theta);
        auto sinth = std::sin(theta);
        auto sq = std::sqrt(-q);
        result.x1 = 2.0 * sq * costh - a2 / 3.0;
        result.x2 = -sq * costh - a2 / 3.0 - std::sqrt(3) * sq * sinth;
        result.x3 = -sq * costh - a2 / 3.0 + std::sqrt(3) * sq * sinth;
        result.num_roots = 3;
    }
    else
    {
        auto s = (r >= 0.0) ? std::pow(r, 1. / 3) : -std::pow(-r, 1. / 3);
        result.x1 = 2.0 * s - a2 / 3.0;
        result.x2 = -s - a2 / 3.0;
        result.x3 = -s - a2 / 3.0;
        result.num_roots = 3;
    }
    return result;
}




/**
 * @brief      Compute the zeros of a quartic polynomial given by sum(d_n x^n)
 *
 * @param[in]  d4    The coefficient of x^4
 * @param[in]  d3    The coefficient of x^3
 * @param[in]  d2    The coefficient of x^2
 * @param[in]  d1    The coefficient of x^1
 * @param[in]  d0    The coefficient of x^0
 *
 * @return     The real roots, including whether the first and second pairs
 *             exist
 */
mara::quartic_roots_t mara::quartic_roots(double d4, double d3, double d2, double d1, double d0)
{
    auto a3 = d3 / d4;
    auto a2 = d2 / d4;
    auto a1 = d1 / d4;
    auto a0 = d0 / d4;
    auto au2 = -a2;
    auto au1 = (a1 * a3 - 4 * a0) ;
    auto au0 = 4.0 * a0 * a2 - a1 * a1 - a0 * a3 * a3;

    auto cubic = cubic_roots(1.0, au2, au1, au0);
    auto u1 = cubic.num_roots == 1 ? cubic.x1 : std::max(cubic.x1, cubic.x3);
    auto R2 = 0.25 * a3 * a3 + u1 - a2;
    auto R = (R2 > 0.0) ? std::sqrt(R2) : 0.0;

    double D2, E2;
    auto result = quartic_roots_t();

    if (R != 0.0)
    {
        auto f = 0.75 * a3 * a3 - R2 - 2.0 * a2;
        auto g = 0.25 * (4.0 * a3 * a2 - 8.0 * a1 - a3 * a3 * a3) / R;
        D2 = f + g;
        E2 = f - g;
    }
    else
    {
        auto f = 0.75 * a3 * a3 - 2.0 * a2;
        auto g = 2.00 * std::sqrt(u1 * u1 - 4.0 * a0);
        D2 = f + g;
        E2 = f - g;
    }

    if (D2 >= 0.0)
    {
        auto D = std::sqrt(D2);
        result.x1 = -0.25 * a3 + 0.5 * R - 0.5 * D;
        result.x2 = -0.25 * a3 + 0.5 * R + 0.5 * D;
        result.pair_1_is_real = true;
    }
    else
    {
        result.x1 = -0.25 * a3 + 0.5 * R;
        result.x2 = -0.25 * a3 + 0.5 * R;
        result.pair_1_is_real = false;
    }

    if (E2 >= 0.0)
    {
        auto E = std::sqrt(E2);
        result.x3 = -0.25 * a3 - 0.5 * R - 0.5 * E;
        result.x4 = -0.25 * a3 - 0.5 * R + 0.5 * E;
        result.pair_2_is_real = true;
    }
    else
    {
        result.x3 = -0.25 * a3 - 0.5 * R;
        result.x4 = -0.25 * a3 + 0.5 * R;
        result.pair_2_is_real = false;
    }
    return result;
}




/**
 * @brief      Find the vertex of the parabola passing through three points
 *
 * @param[in]  x1    The x coordinate of point 1
 * @param[in]  x2    The x coordinate of point 2
 * @param[in]  x3    The x coordinate of point 3
 * @param[in]  y1    The y coordinate of point 1
 * @param[in]  y2    The y coordinate of point 2
 * @param[in]  y3    The y coordinate of point 3
 *
 * @return     The x and y positions of the vertex
 */
std::pair<double, double> mara::parabola_vertex(double x1, double x2, double x3, double y1, double y2, double y3)
{
    auto d = (x1 - x2) * (x1 - x3) * (x2 - x3);
    auto A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / d;
    auto B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / d;
    auto C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / d;
    auto x = -B / (2 * A);
    auto y = C - B * B / (4 * A);
    return std::make_pair(x, y);
}
