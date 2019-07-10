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
#include <functional> // std::plus, std::multiplies, etc...
#include <tuple>      // std::tuple
#include "core_sequence.hpp"




//=============================================================================
namespace mara
{
    template<typename... Types> struct arithmetic_tuple_t;
    template<typename DerivedType, typename... Types> struct derivable_tuple_t;

    template<typename... Types>
    auto make_arithmetic_tuple(Types... args);

    template<std::size_t Index, typename... Types>
    const auto& get(const arithmetic_tuple_t<Types...>& tup);
}




/**
 * @brief      Class for working with tuples of arithmetic types. Arithmetic
 *             operations apply to the values element-wise.
 *
 * @tparam     Types  The tuple types
 *
 * @note       Arithmetic operations are supported between tuples of different
 *             types and identical sizes, as long as the operation is defined on
 *             the corresponding elements.
 */
template<typename... Types>
struct mara::arithmetic_tuple_t final
{




    /**
     * @brief      Immutable set method; return a new tuple with the value at
     *             the given index replaced.
     *
     * @param[in]  value      The value to put in instead
     *
     * @tparam     Index      The index to replace
     * @tparam     ValueType  The value type (cast to the corresponding type in
     *                        this tuple).
     *
     * @return     A new tuple
     */
    template<std::size_t Index, typename ValueType>
    auto set(const ValueType& value) const
    {
        auto result = *this;
        std::get<Index>(result.__impl) = value;
        return result;
    }




    /**
     * @brief      Map a function over the elements of this tuple, returning
     *             another whose value type is the function's return applied to
     *             each element.
     *
     * @param      fn        The function to map
     *
     * @tparam     Function  The type of the function object
     *
     * @return     A new tuple
     *
     * @note       This method makes a tuple into a "functor" (a formal functor,
     *             not the common misnomer in C++ really meaning function
     *             object).
     */
    template<typename Function>
    auto map(Function&& fn) const
    {
        return unary_op(std::forward<Function>(fn));
    }

    auto to_sequence() const
    {
        return to_sequence_impl<std::common_type_t<Types...>>(std::make_index_sequence<sizeof...(Types)>());
    }


    //=========================================================================
    template<typename T> auto operator+ (const T&a) const { return binary_op(a, std::plus<>()); }
    template<typename T> auto operator- (const T&a) const { return binary_op(a, std::minus<>()); }
    template<typename T> auto operator* (const T&a) const { return binary_op(a, std::multiplies<>()); }
    template<typename T> auto operator/ (const T&a) const { return binary_op(a, std::divides<>()); }
    template<typename T> auto operator&&(const T&a) const { return binary_op(a, std::logical_and<>()); }
    template<typename T> auto operator||(const T&a) const { return binary_op(a, std::logical_or<>()); }
    template<typename T> auto operator==(const T&a) const { return binary_op(a, std::equal_to<>()); }
    template<typename T> auto operator!=(const T&a) const { return binary_op(a, std::not_equal_to<>()); }
    template<typename T> auto operator<=(const T&a) const { return binary_op(a, std::less_equal<>()); }
    template<typename T> auto operator>=(const T&a) const { return binary_op(a, std::greater_equal<>()); }
    template<typename T> auto operator< (const T&a) const { return binary_op(a, std::less<>()); }
    template<typename T> auto operator> (const T&a) const { return binary_op(a, std::greater<>()); }

    template<typename... Ts> auto operator+ (const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::plus<>()); }
    template<typename... Ts> auto operator- (const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::minus<>()); }
    template<typename... Ts> auto operator* (const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::multiplies<>()); }
    template<typename... Ts> auto operator/ (const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::divides<>()); }
    template<typename... Ts> auto operator&&(const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::logical_and<>()); }
    template<typename... Ts> auto operator||(const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::logical_or<>()); }
    template<typename... Ts> auto operator==(const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::equal_to<>()); }
    template<typename... Ts> auto operator!=(const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::not_equal_to<>()); }
    template<typename... Ts> auto operator<=(const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::less_equal<>()); }
    template<typename... Ts> auto operator>=(const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::greater_equal<>()); }
    template<typename... Ts> auto operator< (const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::less<>()); }
    template<typename... Ts> auto operator> (const arithmetic_tuple_t<Ts...>& v) const { return binary_op(v, std::greater<>()); }

    auto operator+() const { return unary_op([] (auto&& x) { return +x; }); }
    auto operator-() const { return unary_op([] (auto&& x) { return -x; }); }




    //=========================================================================
    template<typename Function, std::size_t... Is>
    auto unary_op_impl(Function&& fn, std::index_sequence<Is...>) const
    {
        return make_arithmetic_tuple(fn(mara::get<Is>(*this))...);
    }

    template<typename Function, typename T, std::size_t... Is>
    auto binary_op_impl(Function&& fn, const T& a, std::index_sequence<Is...>) const
    {
        return make_arithmetic_tuple(fn(mara::get<Is>(*this), a)...);
    }

    template<typename Function, typename... OtherTypes, std::size_t... Is>
    auto binary_op_impl(Function&& fn, const arithmetic_tuple_t<OtherTypes...>& v, std::index_sequence<Is...>) const
    {
        return make_arithmetic_tuple(fn(mara::get<Is>(*this), mara::get<Is>(v))...);
    }

    template<typename TargetType, std::size_t... Is>
    auto to_sequence_impl(std::index_sequence<Is...>) const
    {
        return make_sequence(TargetType(mara::get<Is>(*this))...);
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        return unary_op_impl(fn, std::make_index_sequence<sizeof...(Types)>());
    }

    template<typename T, typename Function>
    auto binary_op(const T& a, Function&& fn) const
    {
        return binary_op_impl(fn, a, std::make_index_sequence<sizeof...(Types)>());
    }

    template<typename... OtherTypes, typename Function>
    auto binary_op(const arithmetic_tuple_t<OtherTypes...>& v, Function&& fn) const
    {
        static_assert(sizeof...(Types) == sizeof...(OtherTypes), "binary operation between tuples of different size");
        return binary_op_impl(fn, v, std::make_index_sequence<sizeof...(Types)>());
    }




    //=========================================================================
    std::tuple<Types...> __impl;
};




/**
 * @brief      Class for working with tuples of native types - floats, doubles,
 *             etc.
 *
 * @tparam     ValueType    The underlying value type
 * @tparam     Rank         The dimensionality
 * @tparam     DerivedType  The CRTP class (google 'curiously recurring template
 *                          pattern')
 *
 * @note       You can add/subtract other sequences of the same rank and type,
 *             and you can multiply the whole sequence by scalars of the same
 *             value type. Arithmetic operations all return another instance of
 *             the same class type. For sequences that are covariant in the
 *             value type, see the arithmetic_tuple_t. This class is not used
 *             directly; you should inherit it with the CRTP pattern. This means
 *             that type identity is defined by the name of the derived class
 *             (different derived classes with the same rank and value type are
 *             not considered equal by the compiler).
 */
template<typename DerivedType, typename... Types>
struct mara::derivable_tuple_t
{
    //=========================================================================
    bool operator==(const DerivedType& other) const { return __impl.operator==(other.__impl).to_sequenc().all(); }
    bool operator!=(const DerivedType& other) const { return __impl.operator!=(other.__impl).to_sequenc().any(); }
    template<typename T> DerivedType operator*(const T& other)   const { return DerivedType{{__impl.operator*(other)}}; }
    template<typename T> DerivedType operator/(const T& other)   const { return DerivedType{{__impl.operator/(other)}}; }
    DerivedType operator+(const DerivedType& other) const { return DerivedType{{__impl.operator+(other.__impl)}}; }
    DerivedType operator-(const DerivedType& other) const { return DerivedType{{__impl.operator-(other.__impl)}}; }
    DerivedType operator+()                         const { return DerivedType{{__impl.operator+()}}; }
    DerivedType operator-()                         const { return DerivedType{{__impl.operator-()}}; }

    //=========================================================================
    arithmetic_tuple_t<Types...> __impl;
};





/**
 * @brief      Make a new sequence with inferred type and size.
 *
 * @param[in]  args       The elements
 *
 * @tparam     Args       The element types
 * @tparam     ValueType  The inferred type, if a common type can be inferred
 *
 * @return     The sequence
 * @note       You can override the inferred value type by doing e.g.
 *             make_sequence<size_t>(1, 2).
 */
template<typename... Types>
auto mara::make_arithmetic_tuple(Types... args)
{
    return arithmetic_tuple_t<Types...>{std::tuple<Types...>(args...)};
}




/**
 * @brief      Type-safe indexing (preferred over operator[] where possible
 *             for safety).
 *
 * @tparam     Index  The index to get
 *
 * @return     The value at the given index
 */
template<std::size_t Index, typename... Types>
const auto& mara::get(const arithmetic_tuple_t<Types...>& tup)
{
    return std::get<Index>(tup.__impl);
}
