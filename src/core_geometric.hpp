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
#include <string>
#include <cmath>
#include "core_sequence.hpp"
#include "core_dimensional.hpp"




//=============================================================================
namespace mara
{
    class unit_vector_t;
    using spatial_coordinate_t = covariant_sequence_t<unit_length<double>, 3>;
    using area_element_t       = covariant_sequence_t<unit_area  <double>, 3>;

    inline auto make_spatial_coordinate(double x1, double x2, double x3);
    inline auto make_area_element(double da1, double da2, double da3);
}




/**
 * @brief      A class encapsulating a direction in 3D space. Cannot be added or
 *             scaled, just constructed. The sum of the squares of the
 *             components is always 1.
 */
class mara::unit_vector_t
{
public:

    //=========================================================================
    static unit_vector_t on_axis_1() { return {1.0, 0.0, 0.0}; }
    static unit_vector_t on_axis_2() { return {0.0, 1.0, 0.0}; }
    static unit_vector_t on_axis_3() { return {0.0, 0.0, 1.0}; }
    static unit_vector_t on_axis(std::size_t axis)
    {
        switch (axis)
        {
            case 0: return on_axis_1();
            case 1: return on_axis_2();
            case 2: return on_axis_3();
        }
        throw std::invalid_argument("can only construct unit vector on axis 0, 1, or 2");
    }

    unit_vector_t(double n1, double n2, double n3) : n1(n1), n2(n2), n3(n3)
    {
        auto n = std::sqrt(n1 * n1 + n2 * n2 + n3 * n3);
        n1 /= n;
        n2 /= n;
        n3 /= n;
    }

    template<typename ScalarType>
    ScalarType project(ScalarType v1, ScalarType v2, ScalarType v3) const
    {
        return v1 * n1 + v2 * n2 + v3 * n3;
    }

    const double& get_n1() const { return n1; }
    const double& get_n2() const { return n2; }
    const double& get_n3() const { return n3; }

private:
    //=========================================================================
    double n1 = 1.0;
    double n2 = 0.0;
    double n3 = 0.0;
};




//=============================================================================
auto mara::make_spatial_coordinate(double x1, double x2, double x3)
{
    return spatial_coordinate_t {{ mara::make_length(x1), mara::make_length(x2), mara::make_length(x3) }};
}

auto mara::make_area_element(double da1, double da2, double da3)
{
    return area_element_t {{ mara::make_area(da1), mara::make_area(da2), mara::make_area(da3) }};
}
