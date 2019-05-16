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
#include <tuple>
#include <functional>




//=============================================================================
namespace mara
{
    template<typename DerivedClass, typename... Types>
    class covariant_tuple_t;
}




//=============================================================================
template<typename DerivedClass, typename... Types>
class mara::covariant_tuple_t
{
public:

    //=========================================================================
    covariant_tuple_t() {}
    covariant_tuple_t(std::tuple<Types...> the_tuple) : the_tuple(the_tuple) {}

    DerivedClass operator+(const DerivedClass& other) const { return transform(other, std::plus<>()); }
    DerivedClass operator-(const DerivedClass& other) const { return transform(other, std::minus<>()); }

    template<std::size_t Index>
    const auto& get() const { return std::get<Index>(the_tuple); }

    template<typename Function>
    auto transform(Function&& fn) const
    {
        auto is = std::make_index_sequence<sizeof...(Types)>();
        return transform_impl(std::forward<Function>(fn), std::move(is));
    }

    template<typename Function>
    auto transform(const DerivedClass& other, Function&& fn) const
    {
        auto is = std::make_index_sequence<sizeof...(Types)>();
        return transform_impl(other, std::forward<Function>(fn), std::move(is));
    }

private:
    //=========================================================================
    template<typename Function, std::size_t... Is>
    auto transform_impl(Function&& fn, std::index_sequence<Is...>&&) const
    {
        return std::make_tuple(fn(std::get<Is>(the_tuple))...);
    }

    template<typename Function, std::size_t... Is>
    auto transform_impl(const DerivedClass& other, Function&& fn, std::index_sequence<Is...>&&) const
    {
        return std::make_tuple(fn(std::get<Is>(the_tuple), std::get<Is>(other.the_tuple))...);
    }

    std::tuple<Types...> the_tuple;
};

