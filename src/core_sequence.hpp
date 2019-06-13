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
#include <string>     // std::to_string
#include <array>      // std::array




//=============================================================================
namespace mara
{
    template<typename ValueType, std::size_t Rank, typename DerivedType> struct fixed_length_sequence_t;
    template<typename ValueType, std::size_t Rank, typename DerivedType> struct arithmetic_sequence_t;
    template<typename ValueType, std::size_t Rank> struct covariant_sequence_t;

    template<typename ValueType, std::size_t Rank, typename DerivedType>
    auto to_string(const fixed_length_sequence_t<ValueType, Rank, DerivedType>& sequence);

    template<typename Function>
    auto lift(Function f);

    template<typename... Args, typename ValueType=std::common_type_t<Args...>>
    auto make_std_array(Args... args);

    template<typename... Args, typename ValueType=std::common_type_t<Args...>>
    auto make_sequence(Args&&... args);
}




/**
 * @brief      Class for fixed length sequences that are iterable, constructible
 *             from other iterables, and element-wise comparable to other
 *             instances of the same type. This class is meant to be inherited
 *             from using the CRTP pattern.
 *
 * @tparam     Rank         The dimensionality
 * @tparam     ValueType    The underlying value type
 * @tparam     DerivedType  The CRTP class (google 'curiously recurring template pattern')
 */
template<typename ValueType, std::size_t Rank, typename DerivedType>
struct mara::fixed_length_sequence_t
{
    using value_type = ValueType;

    //=========================================================================
    bool operator==(const DerivedType& other) const
    { for (std::size_t n = 0; n < Rank; ++n) { if (memory[n] != other[n]) return false; } return true; }

    bool operator!=(const DerivedType& other) const
    { for (std::size_t n = 0; n < Rank; ++n) { if (memory[n] != other[n]) return true; } return false; }

    constexpr std::size_t size() const { return Rank; }
    const ValueType* data() const { return memory; }
    const ValueType* begin() const { return memory; }
    const ValueType* end() const { return memory + Rank; }
    const ValueType& operator[](std::size_t n) const { return memory[n]; }

    ValueType* data() { return memory; }
    ValueType* begin() { return memory; }
    ValueType* end() { return memory + Rank; }
    ValueType& operator[](std::size_t n) { return memory[n]; }

    template<typename Function>
    auto transform(Function&& fn) const
    {
        auto result = DerivedType();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(memory[n]);
        }
        return result;
    }

    ValueType memory[Rank];
};




/**
 * @brief      Class for working with a fixed-length sequence of native types -
 *             floats, doubles, etc.
 *
 * @tparam     Rank         The dimensionality
 * @tparam     ValueType    The underlying value type
 * @tparam     DerivedType  The CRTP class (google 'curiously recurring template
 *                          pattern')
 *
 * @note       You can add/subtract other sequences of the same rank and type,
 *             and you can multiply the whole sequence by scalars of the same
 *             value type. Arithmetic operations all return another instance of
 *             the same class type. For sequences that are covariant in the
 *             value type, see the covariant_sequence_t. This class is not
 *             used directly; you should inherit it with the CRTP pattern. This
 *             means that type identity is defined by the name of the derived
 *             class (different derived classes with the same rank and value
 *             type are not considered equal by the compiler).
 */
template<typename ValueType, std::size_t Rank, typename DerivedType>
struct mara::arithmetic_sequence_t : public fixed_length_sequence_t<ValueType, Rank, DerivedType>
{
public:

    //=========================================================================
    using fixed_length_sequence_t<ValueType, Rank, DerivedType>::fixed_length_sequence_t;

    //=========================================================================
    DerivedType operator*(const ValueType& other) const { return binary_op(other, std::multiplies<>()); }
    DerivedType operator/(const ValueType& other) const { return binary_op(other, std::divides<>()); }
    DerivedType operator+(const DerivedType& other) const { return binary_op(other, std::plus<>()); }
    DerivedType operator-(const DerivedType& other) const { return binary_op(other, std::minus<>()); }
    DerivedType operator+() const { return unary_op([] (auto&& x) { return x; }); }
    DerivedType operator-() const { return unary_op([] (auto&& x) { return x; }); }

private:
    //=========================================================================
    template<typename Function>
    auto binary_op(const DerivedType& other, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = DerivedType();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n], other[n]);
        }
        return result;
    }

    template<typename Function>
    auto binary_op(const ValueType& value, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = DerivedType();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n], value);
        }
        return result;
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        const auto& _ = *this;
        auto result = DerivedType();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n]);
        }
        return result;
    }
};




/**
 * @brief      Class for working with sequences of dimensional arithmetic types.
 *
 * @tparam     Rank       The dimensionality
 * @tparam     ValueType  The underlying value type
 *
 * @note       You can add/subtract sequences of the same rank and value type,
 *             and you can multiply/divide by scalars that multiply/divide the
 *             value type. These operations are covariant in the value type -
 *             they return another sequence with the same rank, but a value type
 *             determined by the result of the binary operation. This class is
 *             marked final - it is not inherited by base classes, so type
 *             identity is defined by the rank and the identity of the value
 *             type.
 */
template<typename ValueType, std::size_t Rank>
struct mara::covariant_sequence_t final : public fixed_length_sequence_t<ValueType, Rank, covariant_sequence_t<ValueType, Rank>>
{
public:

    //=========================================================================
    template <typename T> auto operator*(const T& a) const { return binary_op(a, std::multiplies<>()); }
    template <typename T> auto operator/(const T& a) const { return binary_op(a, std::divides<>()); }
    auto operator+(const covariant_sequence_t& v) const { return binary_op(v, std::plus<>()); }
    auto operator-(const covariant_sequence_t& v) const { return binary_op(v, std::minus<>()); }
    auto operator-() const { return unary_op(std::negate<>()); }


    /**
     * @brief      Return a new sequence by mapping the elements of this one
     *             through a function f, which may return a value type other
     *             than the type of this sequence's elements.
     *
     * @param      fn        The function to transform by
     *
     * @tparam     Function  The type of the function object
     *
     * @return     The new sequence
     */
    template<typename Function>
    auto transform(Function&& fn) const
    {
        return unary_op(std::forward<Function>(fn));
    }


    /**
     * @brief      Return a new sequence built from the final Rank - 1 elements
     *             of this one.
     *
     * @return     The new sequence
     */
    auto drop_first() const
    {
        auto result = covariant_sequence_t<ValueType, Rank - 1>();

        for (std::size_t n = 0; n < Rank - 1; ++n)
        {
            result.memory[n] = this->operator[](n + 1);
        }
        return result;
    }


    /**
     * @brief      Return a new sequence built from the first Rank - 1 elements
     *             of this one.
     *
     * @return     The new sequence
     */
    auto drop_final() const
    {
        auto result = covariant_sequence_t<ValueType, Rank - 1>();

        for (std::size_t n = 0; n < Rank - 1; ++n)
        {
            result.memory[n] = this->operator[](n);
        }
        return result;
    }


private:
    //=========================================================================
    template<typename T, typename Function>
    auto binary_op(const covariant_sequence_t<T, Rank>& v, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = covariant_sequence_t<std::invoke_result_t<Function, ValueType, T>, Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n], v[n]);
        }
        return result;
    }

    template<typename T, typename Function>
    auto binary_op(const T& a, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = covariant_sequence_t<std::invoke_result_t<Function, ValueType, T>, Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n], a);
        }
        return result;
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        const auto& _ = *this;
        auto result = covariant_sequence_t<std::invoke_result_t<Function, ValueType>, Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n]);
        }
        return result;
    }
};




//=============================================================================
template<typename ValueType, std::size_t Rank, typename DerivedType>
auto mara::to_string(const mara::fixed_length_sequence_t<ValueType, Rank, DerivedType>& sequence)
{
    using std::to_string;

    auto result = std::string("( ");

    for (std::size_t axis = 0; axis < Rank; ++axis)
    {
        if constexpr (std::is_same<ValueType, double>::value)
        {
            result += to_string(sequence[axis]) + " ";
        }
        else
        {
            result += to_string(sequence[axis]) + " ";            
        }
    }
    return result + ")";
}




/**
 * @brief      Make a C++ std::array - planned for C++20
 *
 * @param      t     The arguments to be put in the array (must have uniform
 *                   type)
 *
 * @tparam     D     Type to force conversion to
 *
 * @return     A std::array made from the arguments
 *
 * @note       See https://en.cppreference.com/w/cpp/experimental/make_array for
 *             implementation.
 */
template<typename... Args, typename ValueType>
auto mara::make_std_array(Args... args)
{
    return std::array<ValueType, sizeof...(Args)> {ValueType(args)...};
}




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
template<typename... Args, typename ValueType>
auto mara::make_sequence(Args&&... args)
{
    return covariant_sequence_t<ValueType, sizeof...(Args)> {{ValueType(args)...}};
}




/**
 * @brief      Higher order function that turns its argument into a function
 *             that operates element-wise on fixed length sequences.
 *
 * @param[in]  f         The function
 *
 * @tparam     Function  The type of the function: arguments and return type
 *                       must all be the same
 *
 * @return     F := (a, b, ...) -> [f(a[i], b[i], ...) for i=0,rank]
 */
template<typename Function>
auto mara::lift(Function f)
{
    return [f] (auto... args)
    {
        auto arg_array = make_std_array(args...);
        auto result = typename decltype(arg_array)::value_type();
        constexpr std::size_t size = arg_array[0].size(); // ensures sequence is fixed length

        for (std::size_t n = 0; n < size; ++n)
        {
            result[n] = f(args[n]...);
        }
        return result;
    };
}
