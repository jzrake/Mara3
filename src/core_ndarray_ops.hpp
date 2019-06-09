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
#include "core_ndarray.hpp"




//=============================================================================
namespace nd
{
    template<typename Multiplier> auto multiply(Multiplier arg);
    template<typename Multiplier> auto divide(Multiplier arg);
    template<typename Function> auto apply(Function fn);
    inline auto select_first(std::size_t count, std::size_t axis);
    inline auto select_final(std::size_t count, std::size_t axis);
    inline auto midpoint_on_axis(std::size_t axis);
    inline auto difference_on_axis(std::size_t axis);
    inline auto zip_adjacent2_on_axis(std::size_t axis);
    inline auto zip_adjacent3_on_axis(std::size_t axis);
    inline auto extend_periodic_on_axis(std::size_t axis, std::size_t guard_count=1);
    inline auto extend_zero_gradient(std::size_t axis);
    inline auto extend_zeros(std::size_t axis);
}




//=============================================================================
template<typename Multiplier>
auto nd::multiply(Multiplier arg)
{
    return [rhs=arg] (auto lhs) { return lhs * rhs; };
};

template<typename Multiplier>
auto nd::divide(Multiplier arg)
{
    return [rhs=arg] (auto lhs) { return lhs / rhs; };
};

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

auto nd::midpoint_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return (
        (array | nd::select_axis(axis).from(0).to(1).from_the_end()) +
        (array | nd::select_axis(axis).from(1).to(0).from_the_end())) * 0.5;
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

auto nd::zip_adjacent2_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return nd::zip_arrays(
        array | nd::select_axis(axis).from(0).to(1).from_the_end(),
        array | nd::select_axis(axis).from(1).to(0).from_the_end());
    };
}

auto nd::zip_adjacent3_on_axis(std::size_t axis)
{
    return [axis] (auto array)
    {
        return nd::zip_arrays(
        array | nd::select_axis(axis).from(0).to(2).from_the_end(),
        array | nd::select_axis(axis).from(1).to(1).from_the_end(),
        array | nd::select_axis(axis).from(2).to(0).from_the_end());
    };
}

auto nd::extend_periodic_on_axis(std::size_t axis, std::size_t guard_count)
{
    return [axis, guard_count] (auto array)
    {
        auto xl = array | select_first(guard_count, axis);
        auto xr = array | select_final(guard_count, axis);
        return xr | nd::concat(array).on_axis(axis) | nd::concat(xl).on_axis(axis);
    };
}

auto nd::extend_zero_gradient(std::size_t axis)
{
    return [axis] (auto array)
    {
        auto xl = array | nd::select_first(1, axis);
        auto xr = array | nd::select_final(1, axis);
        return xl | nd::concat(array).on_axis(axis) | nd::concat(xr).on_axis(axis);
    };
}

auto nd::extend_zeros(std::size_t axis)
{
    return [axis] (auto array)
    {
        auto xl = array | nd::select_first(1, axis) | nd::multiply(0);
        auto xr = array | nd::select_final(1, axis) | nd::multiply(0);
        return xl | nd::concat(array).on_axis(axis) | nd::concat(xr).on_axis(axis);
    };
}
