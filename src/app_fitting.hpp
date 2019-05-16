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
#include <utility>




// This header doesn't really belong in 'app' - more like math_algebra.hpp,
// math_statistics.hpp, math_stencils.hpp, etc.




//=============================================================================
namespace mara
{

    inline std::pair<double, double> parabola_vertex(double x1, double x2, double x3, double y1, double y2, double y3);
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
