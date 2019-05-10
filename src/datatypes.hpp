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




//=============================================================================
namespace mara
{
    template<typename DerivedType> struct dimensional_t;

    class unit_vector_t;
    template<typename ValueType, std::size_t Rank, typename DerivedType> class fixed_length_sequence_t;
    template<typename ValueType, std::size_t Rank, typename DerivedType> class arithmetic_sequence_t;
    template<typename ValueType, std::size_t Rank> class dimensional_sequence_t;
    template<std::size_t Rank> class spatial_coordinate_t;

    template<typename ValueType, std::size_t Rank, typename DerivedType>
    auto to_string(const mara::fixed_length_sequence_t<ValueType, Rank, DerivedType>&);

    template<typename DerivedType>
    auto to_string(const mara::dimensional_t<DerivedType>&);
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




/**
 * @brief      A data structure intended to help constrain arithmetic operations
 *             on double-precision floating point numbers, which correspond to
 *             physical quantities like time, mass, energy, etc.
 *
 * @tparam     DerivedType  The CRTP class (google 'curiously recurring template
 *                          pattern')
 */
template<typename DerivedType>
struct mara::dimensional_t
{
    dimensional_t() {}
    dimensional_t(double value) : value(value) {}

    DerivedType operator+(dimensional_t v) const { return {{ value + v.value }}; }
    DerivedType operator-(dimensional_t v) const { return {{ value - v.value }}; }
    DerivedType operator*(double s) const { return {{ value * s }}; }
    DerivedType operator/(double s) const { return {{ value / s }}; }
    double operator/(DerivedType s) const { return value / s.value; }

    double value = 0.0;
};




//=============================================================================
namespace mara
{
    struct time_delta_t : dimensional_t<time_delta_t> {};
    struct area_t       : dimensional_t<area_t> {};
    struct volume_t     : dimensional_t<volume_t> {};
    struct velocity_t   : dimensional_t<velocity_t> {};
    struct intrinsic_t  : dimensional_t<intrinsic_t> {}; // e.g. energy / volume
    struct extrinsic_t  : dimensional_t<extrinsic_t> {}; // intrinsic * volume
    struct flow_rate_t  : dimensional_t<flow_rate_t> {}; // extrinsic / time
    struct flux_t       : dimensional_t<flux_t> {};      // flow_rate_t / area

    inline auto make_time_delta(double value) { return time_delta_t { value }; }
    inline auto make_area      (double value) { return area_t       { value }; }
    inline auto make_volume    (double value) { return volume_t     { value }; }
    inline auto make_velocity  (double value) { return velocity_t   { value }; }

    inline extrinsic_t operator*(intrinsic_t i, volume_t v)     { return { i.value * v.value }; }
    inline intrinsic_t operator/(extrinsic_t e, volume_t v)     { return { e.value / v.value }; }
    inline extrinsic_t operator*(flow_rate_t e, time_delta_t t) { return { e.value * t.value }; }
    inline flow_rate_t operator/(extrinsic_t e, time_delta_t t) { return { e.value / t.value }; }
    inline flow_rate_t operator*(flux_t f, area_t a)            { return { f.value * a.value }; }
    inline flux_t operator/(flow_rate_t r, area_t a)            { return { r.value / a.value }; }
    inline flux_t operator*(intrinsic_t i, velocity_t v)        { return { i.value * v.value }; }
    inline intrinsic_t operator/(flux_t f, velocity_t v)        { return { f.value / v.value }; }

    inline auto make_area_element(double da1, double da2, double da3);
    inline auto make_unit_vector(double n1, double n2, double n3);

    using area_element_t = dimensional_sequence_t<area_t, 3>;
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
class mara::fixed_length_sequence_t
{
public:
    using value_type = ValueType;

    //=========================================================================
    template<typename Container>
    static DerivedType from_range(Container container)
    {
        DerivedType result;
        std::size_t n = 0;

        if (container.size() != Rank)
        {
            throw std::invalid_argument("size of container does not match rank");
        }
        for (auto item : container)
        {
            result.memory[n++] = item;
        }
        return result;
    }

    static DerivedType uniform(ValueType arg)
    {
        DerivedType result;

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result.memory[n] = arg;
        }
        return result;
    }

    fixed_length_sequence_t()
    {
        for (std::size_t n = 0; n < Rank; ++n)
        {
            memory[n] = ValueType();
        }
    }

    fixed_length_sequence_t(std::initializer_list<ValueType> args)
    {
        if (args.size() != Rank)
        {
            throw std::invalid_argument("size of initializer list does not match rank");
        }
        std::size_t n = 0;

        for (auto a : args)
        {
            memory[n++] = a;
        }
    }

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

private:
    //=========================================================================
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
 *             value type, see the dimensional_sequence_t. This class is not
 *             used directly; you should inherit it with the CRTP pattern. This
 *             means that type identity is defined by the name of the derived
 *             class (different derived classes with the same rank and value
 *             type are not considered equal by the compiler).
 */
template<typename ValueType, std::size_t Rank, typename DerivedType>
class mara::arithmetic_sequence_t : public fixed_length_sequence_t<ValueType, Rank, DerivedType>
{
public:

    //=========================================================================
    DerivedType operator*(const ValueType& other) const { return binary_op(other, std::multiplies<>()); }
    DerivedType operator/(const ValueType& other) const { return binary_op(other, std::divides<>()); }
    DerivedType operator+(const DerivedType& other) const { return binary_op(other, std::plus<>()); }
    DerivedType operator-(const DerivedType& other) const { return binary_op(other, std::minus<>()); }
    DerivedType operator-() const { return unary_op(std::negate<>()); }

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
class mara::dimensional_sequence_t final : public fixed_length_sequence_t<ValueType, Rank, dimensional_sequence_t<ValueType, Rank>>
{
public:

    //=========================================================================
    template <typename T> auto operator*(const T& a) const { return binary_op(a, std::multiplies<>()); }
    template <typename T> auto operator/(const T& a) const { return binary_op(a, std::divides<>()); }
    auto operator+(const dimensional_sequence_t& v) const { return binary_op(v, std::plus<>()); }
    auto operator-(const dimensional_sequence_t& v) const { return binary_op(v, std::minus<>()); }
    auto operator-() const { return unary_op(std::negate<>()); }

private:
    //=========================================================================
    template<typename Function>
    auto binary_op(const dimensional_sequence_t& v, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = dimensional_sequence_t();

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
        auto result = dimensional_sequence_t<std::invoke_result_t<Function, ValueType, T>, Rank>();

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
        auto result = dimensional_sequence_t<std::invoke_result_t<Function, ValueType>, Rank>();

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result[n] = fn(_[n]);
        }
        return result;
    }
};




//=============================================================================
template<std::size_t Rank>
class mara::spatial_coordinate_t : public arithmetic_sequence_t<double, Rank, spatial_coordinate_t<Rank>>
{
public:
    using arithmetic_sequence_t<double, Rank, spatial_coordinate_t<Rank>>::arithmetic_sequence_t;
};




//=============================================================================
auto mara::make_area_element(double da1, double da2, double da3)
{
    return area_element_t {{ make_area(da1), make_area(da2), make_area(da3) }};
};

auto mara::make_unit_vector(double n1, double n2, double n3)
{
    return unit_vector_t(n1, n2, n3);
}




//=============================================================================
template<typename ValueType, std::size_t Rank, typename DerivedType>
auto mara::to_string(const mara::fixed_length_sequence_t<ValueType, Rank, DerivedType>& sequence)
{
    auto result = std::string("( ");

    for (std::size_t axis = 0; axis < Rank; ++axis)
    {
        if constexpr (std::is_same<ValueType, double>::value)
        {
            result += std::to_string(sequence[axis]) + " ";
        }
        else
        {
            result += mara::to_string(sequence[axis]) + " ";            
        }
    }
    return result + ")";
}

template<typename DerivedType>
auto mara::to_string(const mara::dimensional_t<DerivedType>& x)
{
    return std::to_string(x.value);
}
