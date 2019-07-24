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
#include "core_dimensional.hpp"
#include "core_sequence.hpp"
#include "core_tuple.hpp"




//=============================================================================
namespace mara
{
    inline double plm_gradient(double yl, double y0, double yr, double theta);

    template<int C, int G, int S, typename ValueType>
    auto plm_gradient(
        dimensional_value_t<C, G, S, ValueType> yl,
        dimensional_value_t<C, G, S, ValueType> y0,
        dimensional_value_t<C, G, S, ValueType> yr, double theta);

    template<typename ValueType, std::size_t Rank>
    auto plm_gradient(
        arithmetic_sequence_t<ValueType, Rank> yl,
        arithmetic_sequence_t<ValueType, Rank> y0,
        arithmetic_sequence_t<ValueType, Rank> yr, double theta);

    template<typename ValueType, std::size_t Rank, typename DerivedType>
    auto plm_gradient(
        derivable_sequence_t<ValueType, Rank, DerivedType> yl,
        derivable_sequence_t<ValueType, Rank, DerivedType> y0,
        derivable_sequence_t<ValueType, Rank, DerivedType> yr, double theta);

    template<typename... ValueType>
    auto plm_gradient(
        arithmetic_tuple_t<ValueType...> yl,
        arithmetic_tuple_t<ValueType...> y0,
        arithmetic_tuple_t<ValueType...> yr, double theta);




    //=========================================================================
    namespace detail
    {
        template<typename Function, typename TupleType, std::size_t... Is>
        auto map_three_tuples(Function&& fn, TupleType a, TupleType b, TupleType c, std::index_sequence<Is...>)
        {
            return make_arithmetic_tuple(fn(mara::get<Is>(a), mara::get<Is>(b), mara::get<Is>(c))...);
        }
    }
}




//=============================================================================
double mara::plm_gradient(double yl, double y0, double yr, double theta)
{
    using std::min, std::abs, std::copysign;
    auto minabs = [] (double a, double b, double c) { return min(min(abs(a), abs(b)), abs(c)); };
    auto sgn = [] (double x) { return copysign(1.0, x); };
    double a = (y0 - yl) * theta;
    double b = (yr - yl) * 0.5;
    double c = (yr - y0) * theta;
    return 0.25 * abs(sgn(a) + sgn(b)) * (sgn(a) + sgn(c)) * minabs(a, b, c);
}

template<int C, int G, int S, typename ValueType>
auto mara::plm_gradient(
    dimensional_value_t<C, G, S, ValueType> yl,
    dimensional_value_t<C, G, S, ValueType> y0,
    dimensional_value_t<C, G, S, ValueType> yr, double theta)
{
    return dimensional_value_t<C, G, S, ValueType>{plm_gradient(yl.value, y0.value, yr.value, theta)};
}

template<typename ValueType, std::size_t Rank>
auto mara::plm_gradient(
    arithmetic_sequence_t<ValueType, Rank> yl,
    arithmetic_sequence_t<ValueType, Rank> y0,
    arithmetic_sequence_t<ValueType, Rank> yr, double theta)
{
    return iota<Rank>().map([=] (auto i) { return plm_gradient(yl[i], y0[i], yr[i], theta); });
}

template<typename ValueType, std::size_t Rank, typename DerivedType>
auto mara::plm_gradient(
    derivable_sequence_t<ValueType, Rank, DerivedType> yl,
    derivable_sequence_t<ValueType, Rank, DerivedType> y0,
    derivable_sequence_t<ValueType, Rank, DerivedType> yr, double theta)
{
    return DerivedType{{plm_gradient(yl.__impl, y0.__impl, yr.__impl, theta)}};
}

template<typename... ValueType>
auto mara::plm_gradient(
    arithmetic_tuple_t<ValueType...> yl,
    arithmetic_tuple_t<ValueType...> y0,
    arithmetic_tuple_t<ValueType...> yr, double theta)
{
    return detail::map_three_tuples([theta] (auto a, auto b, auto c)
    {
        return plm_gradient(a, b, c, theta);
    }, yl, y0, yr, std::make_index_sequence<sizeof...(ValueType)>());
}
