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
namespace mara
{
    template<std::size_t Rank> auto prolong_shape_verts(nd::shape_t<Rank> shape, std::size_t axis);
    template<std::size_t Rank> auto prolong_shape_cells(nd::shape_t<Rank> shape, std::size_t axis);
    template<std::size_t Rank> auto coarsen_index_lower(nd::index_t<Rank> fi, std::size_t axis);
    template<std::size_t Rank> auto coarsen_index_upper(nd::index_t<Rank> fi, std::size_t axis);
    template<std::size_t Rank> auto coarsen_index_cells(nd::index_t<Rank> fi, std::size_t axis);
    template<std::size_t Rank> auto refine_verts();
    template<std::size_t Rank> auto refine_cells();
    template<std::size_t Rank> auto coarsen_verts();
    template<std::size_t Rank> auto coarsen_cells();

    template<typename ArrayType> auto combine_verts(const arithmetic_sequence_t<ArrayType, 2>& vert_arrays);
    template<typename ArrayType> auto combine_verts(const arithmetic_sequence_t<ArrayType, 4>& vert_arrays);
    template<typename ArrayType> auto combine_verts(const arithmetic_sequence_t<ArrayType, 8>& vert_arrays);
    template<typename ArrayType> auto combine_cells(const arithmetic_sequence_t<ArrayType, 2>& cell_arrays);
    template<typename ArrayType> auto combine_cells(const arithmetic_sequence_t<ArrayType, 4>& cell_arrays);
    template<typename ArrayType> auto combine_cells(const arithmetic_sequence_t<ArrayType, 8>& cell_arrays);

    inline auto restrict_verts    (std::size_t axis);
    inline auto restrict_cells    (std::size_t axis);
    inline auto prolong_verts     (std::size_t axis);
    inline auto prolong_cells     (std::size_t axis);
    inline auto bisect_verts      (std::size_t axis);
    inline auto bisect_cells      (std::size_t axis);
    inline auto bisect_verts_lower(std::size_t axis);
    inline auto bisect_verts_upper(std::size_t axis);
    inline auto bisect_cells_lower(std::size_t axis);
    inline auto bisect_cells_upper(std::size_t axis);
}




//=============================================================================
template<std::size_t Rank>
auto mara::prolong_shape_verts(nd::shape_t<Rank> shape, std::size_t axis)
{
    shape[axis] = shape[axis] * 2 - 1;
    return shape;
}

template<std::size_t Rank>
auto mara::prolong_shape_cells(nd::shape_t<Rank> shape, std::size_t axis)
{
    shape[axis] = shape[axis] * 2;
    return shape;
}

template<std::size_t Rank>
auto mara::coarsen_index_lower(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c0 = fi;
    c0[axis] = fi[axis] / 2;
    return c0;
}

template<std::size_t Rank>
auto mara::coarsen_index_upper(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c1 = fi;
    c1[axis] = fi[axis] / 2 + (fi[axis] % 2 == 0 ? 0 : 1);
    return c1;
}

template<std::size_t Rank>
auto mara::coarsen_index_cells(nd::index_t<Rank> fi, std::size_t axis)
{
    auto c1 = fi;
    c1[axis] = fi[axis] / 2;
    return c1;
}




//=============================================================================
auto mara::restrict_verts(std::size_t axis)
{
    return [axis] (auto parent)
    {
        return parent | nd::select_axis(axis).from(0).to(0).from_the_end().jumping(2);
    };
}

auto mara::restrict_cells(std::size_t axis)
{
    return [axis] (auto parent)
    {
        auto h0 = parent | nd::select_axis(axis).from(0).to(1).from_the_end().jumping(2);
        auto h1 = parent | nd::select_axis(axis).from(1).to(0).from_the_end().jumping(2);
        return (h0 + h1) / 2;
    };
}




//=============================================================================
auto mara::prolong_verts(std::size_t axis)
{
    return [axis] (auto coarse)
    {
        return nd::make_array([axis, coarse] (auto i)
        {
            return (
                coarse(coarsen_index_lower(i, axis)) +
                coarse(coarsen_index_upper(i, axis))) * 0.5;
        }, prolong_shape_verts(coarse.shape(), axis));
    };
}

auto mara::prolong_cells(std::size_t axis)
{
    return [axis] (auto coarse)
    {
        return nd::make_array([axis, coarse] (auto i)
        {
            // This is a cheap second-order cell prolongation
            auto Im = i;
            auto I0 = i;
            auto Ip = i;
            auto N0 = coarse.shape(axis);

            Im[axis] = i[axis] / 2 - 1;
            I0[axis] = i[axis] / 2;
            Ip[axis] = i[axis] / 2 + 1;

            auto dx =  i[axis] % 2 == 0 ? -0.25 : +0.25;
            auto Dp = I0[axis] < N0 - 1 ? coarse(Ip) - coarse(I0) : coarse(I0) - coarse(Im);
            auto Dm = I0[axis] >      0 ? coarse(I0) - coarse(Im) : coarse(Ip) - coarse(I0);

            auto yi = coarse(I0) + (Dm + Dp) * dx * 0.5;
            return yi;

            // This is a first-order prolongation
            // return coarse(coarsen_index_cells(i, axis));
        }, prolong_shape_cells(coarse.shape(), axis));
    };
}




//=============================================================================
auto mara::bisect_verts(std::size_t axis)
{
    return [axis] (auto parent)
    {
        auto h0 = parent | nd::select_axis(axis).from(0).to(parent.shape(axis) / 2 + 1);
        auto h1 = parent | nd::select_axis(axis).from(parent.shape(axis) / 2).to(0).from_the_end();
        return std::make_tuple(h0, h1);
    };
}

auto mara::bisect_cells(std::size_t axis)
{
    return [axis] (auto parent)
    {
        auto h0 = parent | nd::select_axis(axis).from(0).to(parent.shape(axis) / 2);
        auto h1 = parent | nd::select_axis(axis).from(parent.shape(axis) / 2).to(0).from_the_end();
        return std::make_tuple(h0, h1);
    };
}

auto mara::bisect_verts_lower(std::size_t a) { return [a] (auto parent) { return std::get<0>(parent | bisect_verts(a)); }; }
auto mara::bisect_verts_upper(std::size_t a) { return [a] (auto parent) { return std::get<1>(parent | bisect_verts(a)); }; }
auto mara::bisect_cells_lower(std::size_t a) { return [a] (auto parent) { return std::get<0>(parent | bisect_cells(a)); }; }
auto mara::bisect_cells_upper(std::size_t a) { return [a] (auto parent) { return std::get<1>(parent | bisect_cells(a)); }; }




//=============================================================================
template<typename ArrayType> auto mara::combine_verts(const arithmetic_sequence_t<ArrayType, 2>& vert_arrays)
{
    auto drop_last = [] (std::size_t a) { return nd::select_axis(a).from(0).to(1).from_the_end(); };
    return vert_arrays[0] | drop_last(0) | nd::concat(vert_arrays[1]).on_axis(0);
}

template<typename ArrayType> auto mara::combine_verts(const arithmetic_sequence_t<ArrayType, 4>& vert_arrays)
{
    auto drop_last = [] (std::size_t a) { return nd::select_axis(a).from(0).to(1).from_the_end(); };
    auto combined_01 = vert_arrays[0] | drop_last(0) | nd::concat(vert_arrays[1]).on_axis(0);
    auto combined_23 = vert_arrays[2] | drop_last(0) | nd::concat(vert_arrays[3]).on_axis(0);
    return combined_01 | drop_last(1) | nd::concat(combined_23).on_axis(1);
}

template<typename ArrayType> auto mara::combine_verts(const arithmetic_sequence_t<ArrayType, 8>& vert_arrays)
{
    throw std::logic_error("combine_verts in 3d not implemented");
}




//=============================================================================
template<typename ArrayType> auto mara::combine_cells(const arithmetic_sequence_t<ArrayType, 2>& cell_arrays)
{
    return cell_arrays[0] | nd::concat(cell_arrays[1]).on_axis(0);
}

template<typename ArrayType> auto mara::combine_cells(const arithmetic_sequence_t<ArrayType, 4>& cell_arrays)
{
    auto combined_01 = cell_arrays[0] | nd::concat(cell_arrays[1]).on_axis(0);
    auto combined_23 = cell_arrays[2] | nd::concat(cell_arrays[3]).on_axis(0);
    return combined_01 | nd::concat(combined_23).on_axis(1);
}

template<typename ArrayType> auto mara::combine_cells(const arithmetic_sequence_t<ArrayType, 8>& cell_arrays)
{
    auto combined_01 = cell_arrays[0] | nd::concat(cell_arrays[1]).on_axis(0);
    auto combined_23 = cell_arrays[2] | nd::concat(cell_arrays[3]).on_axis(0);
    auto combined_45 = cell_arrays[4] | nd::concat(cell_arrays[5]).on_axis(0);
    auto combined_67 = cell_arrays[6] | nd::concat(cell_arrays[7]).on_axis(0);
    auto combined_0123 = combined_01 | nd::concat(combined_23).on_axis(1);
    auto combined_4567 = combined_45 | nd::concat(combined_67).on_axis(1);
    return combined_0123 | nd::concat(combined_4567).on_axis(2);
}




//=============================================================================
template<>
inline auto mara::refine_verts<1>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_verts(0) | bisect_verts_lower(0),
            array | prolong_verts(0) | bisect_verts_upper(0));
    };
}

template<>
inline auto mara::refine_cells<1>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_cells(0) | bisect_cells_lower(0),
            array | prolong_cells(0) | bisect_cells_upper(0));
    };
}

template<>
inline auto mara::refine_verts<2>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_verts(0) | prolong_verts(1) | bisect_verts_lower(0) | bisect_verts_lower(1),
            array | prolong_verts(0) | prolong_verts(1) | bisect_verts_upper(0) | bisect_verts_lower(1),
            array | prolong_verts(0) | prolong_verts(1) | bisect_verts_lower(0) | bisect_verts_upper(1),
            array | prolong_verts(0) | prolong_verts(1) | bisect_verts_upper(0) | bisect_verts_upper(1));
    };
}

template<>
inline auto mara::refine_cells<2>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_cells(0) | prolong_cells(1) | bisect_cells_lower(0) | bisect_cells_lower(1),
            array | prolong_cells(0) | prolong_cells(1) | bisect_cells_upper(0) | bisect_cells_lower(1),
            array | prolong_cells(0) | prolong_cells(1) | bisect_cells_lower(0) | bisect_cells_upper(1),
            array | prolong_cells(0) | prolong_cells(1) | bisect_cells_upper(0) | bisect_cells_upper(1));
    };
}

template<>
inline auto mara::refine_verts<3>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_lower(0) | bisect_verts_lower(1) | bisect_verts_lower(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_upper(0) | bisect_verts_lower(1) | bisect_verts_lower(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_lower(0) | bisect_verts_upper(1) | bisect_verts_lower(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_upper(0) | bisect_verts_upper(1) | bisect_verts_lower(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_lower(0) | bisect_verts_lower(1) | bisect_verts_upper(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_upper(0) | bisect_verts_lower(1) | bisect_verts_upper(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_lower(0) | bisect_verts_upper(1) | bisect_verts_upper(2),
            array | prolong_verts(0) | prolong_verts(1) | prolong_verts(2) | bisect_verts_upper(0) | bisect_verts_upper(1) | bisect_verts_upper(2));
    };
}

template<>
inline auto mara::refine_cells<3>()
{
    return [] (auto array)
    {
        return mara::make_sequence(
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_lower(0) | bisect_cells_lower(1) | bisect_cells_lower(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_upper(0) | bisect_cells_lower(1) | bisect_cells_lower(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_lower(0) | bisect_cells_upper(1) | bisect_cells_lower(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_upper(0) | bisect_cells_upper(1) | bisect_cells_lower(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_lower(0) | bisect_cells_lower(1) | bisect_cells_upper(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_upper(0) | bisect_cells_lower(1) | bisect_cells_upper(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_lower(0) | bisect_cells_upper(1) | bisect_cells_upper(2),
            array | prolong_cells(0) | prolong_cells(1) | prolong_cells(2) | bisect_cells_upper(0) | bisect_cells_upper(1) | bisect_cells_upper(2));
    };
}




//=============================================================================
template<> inline auto mara::coarsen_verts<1>() { return [] (auto array) { return array | restrict_verts(0); }; }
template<> inline auto mara::coarsen_verts<2>() { return [] (auto array) { return array | restrict_verts(0) | restrict_verts(1); }; }
template<> inline auto mara::coarsen_verts<3>() { return [] (auto array) { return array | restrict_verts(0) | restrict_verts(1) | restrict_verts(2); }; }

template<> inline auto mara::coarsen_cells<1>() { return [] (auto array) { return array | restrict_cells(0); }; }
template<> inline auto mara::coarsen_cells<2>() { return [] (auto array) { return array | restrict_cells(0) | restrict_cells(1); }; }
template<> inline auto mara::coarsen_cells<3>() { return [] (auto array) { return array | restrict_cells(0) | restrict_cells(1) | restrict_cells(2); }; }
