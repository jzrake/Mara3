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
    template<std::size_t Rank> class spatial_coordinate_t;
}




//=============================================================================
template<std::size_t Rank, typename ValueType, typename DerivedType>
class mara::arithmetic_sequence_t
{
public:

    using value_type = ValueType;

    //=========================================================================
    static DerivedType uniform(ValueType arg)
    {
        DerivedType result;

        for (auto n : nd::range(Rank))
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
            throw std::logic_error("size of initializer list does not match rank");
        }
        std::size_t n = 0;

        for (auto a : args)
        {
            memory[n++] = a;
        }
    }

    bool operator==(const DerivedType& other) const { return std::memcmp(memory, other.memory, Rank * sizeof(ValueType)) == 0; }
    bool operator!=(const DerivedType& other) const { return std::memcmp(memory, other.memory, Rank * sizeof(ValueType)) != 0; }
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




//=============================================================================
template<std::size_t Rank>
class mara::spatial_coordinate_t : public arithmetic_sequence_t<Rank, double, spatial_coordinate_t<Rank>>
{
public:
    using arithmetic_sequence_t<Rank, double, spatial_coordinate_t<Rank>>::arithmetic_sequence_t;
};


