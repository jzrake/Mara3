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




//=============================================================================
namespace mara
{
    template<std::size_t Rank, typename ValueType, typename DerivedType> class arithmetic_sequence_t;
    template<std::size_t Rank, typename ValueType> class dimensional_sequence_t;
    template<std::size_t Rank> class spatial_coordinate_t;
}





/**
 * @brief      Class for working with sequence of native types - floats,
 *             doubles, etc. Operations cannot change the underlying value
 *             types.
 *
 * @tparam     Rank         The dimensionality
 * @tparam     ValueType    The underlying value type
 * @tparam     DerivedType  The CRTP class (google 'curiously recurring template pattern')
 */
template<std::size_t Rank, typename ValueType, typename DerivedType>
class mara::arithmetic_sequence_t
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

    arithmetic_sequence_t()
    {
        for (std::size_t n = 0; n < Rank; ++n)
        {
            memory[n] = ValueType();
        }
    }

    arithmetic_sequence_t(std::initializer_list<ValueType> args)
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

    bool operator==(const DerivedType& other) const { for (std::size_t n = 0; n < Rank; ++n) { if (memory[n] != other[n]) return false; } return true; }
    bool operator!=(const DerivedType& other) const { for (std::size_t n = 0; n < Rank; ++n) { if (memory[n] != other[n]) return true; } return false; }
    DerivedType operator+(const DerivedType& other) const { return binary_op(other, std::plus<>()); }
    DerivedType operator-(const DerivedType& other) const { return binary_op(other, std::minus<>()); }
    DerivedType operator*(const DerivedType& other) const { return binary_op(other, std::multiplies<>()); }
    DerivedType operator/(const DerivedType& other) const { return binary_op(other, std::divides<>()); }
    DerivedType operator+(const ValueType& other) const { return binary_op(other, std::plus<>()); }
    DerivedType operator-(const ValueType& other) const { return binary_op(other, std::minus<>()); }
    DerivedType operator*(const ValueType& other) const { return binary_op(other, std::multiplies<>()); }
    DerivedType operator/(const ValueType& other) const { return binary_op(other, std::divides<>()); }
    DerivedType operator-() const { return unary_op(std::negate<>()); }

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
    template<typename Function>
    auto binary_op(const DerivedType& other, Function&& fn) const
    {
        DerivedType result;

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = fn(memory[n], other.memory[n]);

        return result;
    }

    template<typename Function>
    auto binary_op(const ValueType& value, Function&& fn) const
    {
        DerivedType result;

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = fn(memory[n], value);

        return result;
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        DerivedType result;

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = fn(memory[n]);

        return result;
    }

    ValueType memory[Rank];
};





/**
 * @brief      Class for working with sequences of dimensional arithmetic types.
 *             Operations are covariant with respect to the underlying value
 *             type.
 *
 * @tparam     Rank       The dimensionality
 * @tparam     ValueType  The underlying value type
 */
template<std::size_t Rank, typename ValueType>
class mara::dimensional_sequence_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    template<typename Container>
    static dimensional_sequence_t from_range(Container container)
    {
        dimensional_sequence_t result;
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

    static dimensional_sequence_t uniform(ValueType arg)
    {
        dimensional_sequence_t result;

        for (std::size_t n = 0; n < Rank; ++n)
        {
            result.memory[n] = arg;
        }
        return result;
    }

    dimensional_sequence_t()
    {
        for (std::size_t n = 0; n < Rank; ++n)
        {
            memory[n] = ValueType();
        }
    }

    dimensional_sequence_t(std::initializer_list<ValueType> args)
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

    bool operator==(const dimensional_sequence_t& v) const { for (std::size_t n = 0; n < Rank; ++n) { if (memory[n] != v[n]) return false; } return true; }
    bool operator!=(const dimensional_sequence_t& v) const { for (std::size_t n = 0; n < Rank; ++n) { if (memory[n] != v[n]) return true; } return false; }
    auto operator+(const dimensional_sequence_t& v) const { return binary_op(v, std::plus<>()); }
    auto operator-(const dimensional_sequence_t& v) const { return binary_op(v, std::minus<>()); }
    template <typename T> auto operator*(const T& a) const { return binary_op(a, std::multiplies<>()); }
    template <typename T> auto operator/(const T& a) const { return binary_op(a, std::divides<>()); }
    auto operator-() const { return unary_op(std::negate<>()); }

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
    template<typename Function>
    auto binary_op(const dimensional_sequence_t& v, Function&& fn) const
    {
        auto result = dimensional_sequence_t();

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = fn(memory[n], v.memory[n]);

        return result;
    }

    template<typename T, typename Function>
    auto binary_op(const T& a, Function&& fn) const
    {
        auto result = dimensional_sequence_t<Rank, std::invoke_result_t<Function, ValueType, T>>();

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = fn(memory[n], a);

        return result;
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        auto result = dimensional_sequence_t<Rank, std::invoke_result_t<Function, ValueType>>();

        for (std::size_t n = 0; n < Rank; ++n)
            result[n] = fn(memory[n]);

        return result;
    }

    ValueType memory[Rank];
};





//=============================================================================
template<std::size_t Rank>
class mara::spatial_coordinate_t : public arithmetic_sequence_t<Rank, double, spatial_coordinate_t<Rank>>
{
public:
    using arithmetic_sequence_t<Rank, double, spatial_coordinate_t<Rank>>::arithmetic_sequence_t;
};
