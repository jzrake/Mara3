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
#include "core_ndarray.hpp"
#include "core_sequence.hpp"




//=============================================================================
namespace mara::amr
{
    template<std::size_t Rank> auto refine_points();
    template<std::size_t Rank> auto prolong_shape(nd::shape_t<Rank> shape, std::size_t axis);
    template<std::size_t Rank> auto coarsen_index_lower(nd::index_t<Rank> fi, std::size_t axis);
    template<std::size_t Rank> auto coarsen_index_upper(nd::index_t<Rank> fi, std::size_t axis);

    inline auto prolong_points(std::size_t axis);
    inline auto bisect_points(std::size_t axis);
    inline auto bisect_points_lower(std::size_t axis);
    inline auto bisect_points_upper(std::size_t axis);
}




//=============================================================================
template<std::size_t Rank>
auto mara::amr::prolong_shape(nd::shape_t<Rank> shape, std::size_t axis)
{
    shape[axis] = shape[axis] * 2 - 1;
    return shape;
}

template<std::size_t Rank>
auto mara::amr::coarsen_index_lower(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c0 = fi;
    c0[axis] = fi[axis] / 2;
    return c0;
}

template<std::size_t Rank>
auto mara::amr::coarsen_index_upper(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c1 = fi;
    c1[axis] = fi[axis] / 2 + (fi[axis] % 2 == 0 ? 0 : 1);
    return c1;
}

auto mara::amr::prolong_points(std::size_t axis)
{
    return [axis] (auto coarse)
    {
        if (coarse.rank() <= axis)
            throw std::invalid_argument("prolong_points: cannot prolong on axis greater than or eaual to rank");

        return nd::make_array([axis, coarse] (auto i)
        {
            return (
                coarse(coarsen_index_lower(i, axis)) +
                coarse(coarsen_index_upper(i, axis))) * 0.5;
        }, prolong_shape(coarse.shape(), axis));
    };
}

auto mara::amr::bisect_points(std::size_t axis)
{
    return [axis] (auto parent)
    {
        if (parent.rank() <= axis)
            throw std::invalid_argument("bisect_points: cannot bisect on axis greater than or eaual to rank");

        if (parent.shape(axis) % 2 == 0)
            throw std::invalid_argument("bisect_points: must have an odd number of points");

        auto h0 = parent | nd::select_axis(axis).from(0).to(parent.shape(axis) / 2 + 1);
        auto h1 = parent | nd::select_axis(axis).from(parent.shape(axis) / 2).to(0).from_the_end();
        return std::make_tuple(h0, h1);
    };
}

auto mara::amr::bisect_points_lower(std::size_t axis)
{
    return [axis] (auto parent)
    {
        return std::get<0>(parent | bisect_points(axis));
    };
}

auto mara::amr::bisect_points_upper(std::size_t axis)
{
    return [axis] (auto parent)
    {
        return std::get<1>(parent | bisect_points(axis));
    };
}




//=============================================================================
template<>
inline auto mara::amr::refine_points<1>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_points(0) | bisect_points_lower(0),
            array | prolong_points(0) | bisect_points_upper(0));
    };
}

template<>
inline auto mara::amr::refine_points<2>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_points(0) | prolong_points(1) | bisect_points_lower(0) | bisect_points_lower(1),
            array | prolong_points(0) | prolong_points(1) | bisect_points_upper(0) | bisect_points_lower(1),
            array | prolong_points(0) | prolong_points(1) | bisect_points_lower(0) | bisect_points_upper(1),
            array | prolong_points(0) | prolong_points(1) | bisect_points_upper(0) | bisect_points_upper(1));
    };
}
