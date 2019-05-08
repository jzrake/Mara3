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
#include "ndarray.hpp"




//=============================================================================
namespace nd
{
    inline auto linspace(double x0, double x1, std::size_t count);
    inline auto midpoint_on_axis(std::size_t axis);
    inline auto select_first(std::size_t count, std::size_t axis);
    inline auto select_final(std::size_t count, std::size_t axis);
    inline auto difference_on_axis(std::size_t axis);
    inline auto intercell_flux_on_axis(std::size_t axis);
    inline auto extend_periodic_on_axis(std::size_t axis);
}




//=============================================================================
auto nd::linspace(double x0, double x1, std::size_t count)
{
    auto mapping = [x0, x1, count] (auto index)
    {
        return x0 + (x1 - x0) * index[0] / (count - 1);
    };
    return make_array(mapping, nd::make_shape(count));
}

auto nd::midpoint_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return (
        (array | nd::select_axis(axis).from(0).to(1).from_the_end()) +
        (array | nd::select_axis(axis).from(1).to(0).from_the_end())) * 0.5;
    };
}

auto nd::select_first(std::size_t count, std::size_t axis)
{
    return [count, axis] (auto array)
    {
        auto shape = array.shape();
        auto start = nd::make_uniform_index<shape.size()>(0);
        auto final = shape.last_index();

        final[axis] = start[axis] + count;

        return array | nd::select(nd::make_access_pattern(shape).with_start(start).with_final(final));
    };
}

auto nd::select_final(std::size_t count, std::size_t axis)
{
    return [count, axis] (auto array)
    {
        auto shape = array.shape();
        auto start = nd::make_uniform_index<shape.size()>(0);
        auto final = shape.last_index();

        start[axis] = final[axis] - count;

        return array | nd::select(nd::make_access_pattern(shape).with_start(start).with_final(final));
    };
}

auto nd::difference_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return (
        (array | nd::select_axis(axis).from(1).to(0).from_the_end()) -
        (array | nd::select_axis(axis).from(0).to(1).from_the_end()));
    };
}

auto nd::intercell_flux_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return array | nd::select_axis(axis).from(0).to(1).from_the_end();        
    };
}

auto nd::extend_periodic_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        auto xl = array | select_first(1, 0);
        auto xr = array | select_final(1, 0);
        return xr | nd::concat(array).on_axis(axis) | nd::concat(xl).on_axis(axis);
    };
}