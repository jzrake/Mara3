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
    template<int C, int G, int S, typename T> struct dimensional_value_t;
    template<int C, int G, int S, typename T> auto make_dimensional(T value);

    template<typename T> using unit_scalar       = dimensional_value_t< 0, 0, 0, T>;
    template<typename T> using unit_length       = dimensional_value_t< 1, 0, 0, T>;
    template<typename T> using unit_mass         = dimensional_value_t< 0, 1, 0, T>;
    template<typename T> using unit_time         = dimensional_value_t< 0, 0, 1, T>;
    template<typename T> using unit_rate         = dimensional_value_t< 0, 0,-1, T>;
    template<typename T> using unit_area         = dimensional_value_t< 2, 0, 0, T>;
    template<typename T> using unit_velocity     = dimensional_value_t< 1, 0,-1, T>;
    template<typename T> using unit_volume       = dimensional_value_t< 3, 0, 0, T>;
    template<typename T> using unit_flux         = dimensional_value_t<-2, 1,-1, T>;
    template<typename T> using unit_flow         = dimensional_value_t< 0, 1,-1, T>;
    template<typename T> using unit_mass_density = dimensional_value_t<-3, 1, 0, T>;
    template<typename T> using unit_flow_density = dimensional_value_t<-3, 1,-1, T>;

    template<typename T> auto make_scalar(T value);
    template<typename T> auto make_length(T value);
    template<typename T> auto make_mass(T value);
    template<typename T> auto make_time(T value);
    template<typename T> auto make_rate(T value);
    template<typename T> auto make_area(T value);
    template<typename T> auto make_velocity(T value);
    template<typename T> auto make_volume(T value);
    template<typename T> auto make_flux(T value);
    template<typename T> auto make_flow(T value);
    template<typename T> auto make_mass_density(T value);
    template<typename T> auto make_flow_density(T value);

    template<int C, int G, int S, typename T> auto to_string(dimensional_value_t<C, G, S, T> x);
}




/**
 * @brief      A type which includes its dimension of length, mass, and time in
 *             addition to the underlying value.
 *
 * @tparam     C     Dimensions of length
 * @tparam     G     Dimensions of mass
 * @tparam     S     Dimensions of time
 * @tparam     T     The value type
 */
template<int C, int G, int S, typename T>
struct mara::dimensional_value_t
{

    dimensional_value_t() {}
    dimensional_value_t(T value) : value(value) {}


    /**
     * @brief      Add another value, with same dimensions
     *
     * @param[in]  other  The other value
     *
     * @return     A new value with the same dimensions
     */
    auto operator+(dimensional_value_t<C, G, S, T> other) const
    {
        return dimensional_value_t { value + other.value };
    }


    /**
     * @brief      Subtract another value, with same dimensions
     *
     * @param[in]  other  The other value
     *
     * @return     A new value with the same dimensions
     */
    auto operator-(dimensional_value_t<C, G, S, T> other) const
    {
        return dimensional_value_t { value - other.value };
    }


    /**
     * @brief      Multiply another dimensional value, of any dimension
     *
     * @param[in]  other  The other value
     *
     * @tparam     C1     Length unit of the other
     * @tparam     G1     Mass unit of the other
     * @tparam     S1     Time unit of the other
     *
     * @return     The product of the values
     */
    template <int C1, int G1, int S1>
    auto operator*(dimensional_value_t<C1, G1, S1, T> other) const
    {
        return dimensional_value_t<C + C1, G + G1, S + S1, T> { value * other.value };
    }


    /**
     * @brief      Divide another dimensional value, of any dimension
     *
     * @param[in]  other  The other value
     *
     * @tparam     C1     Length unit of the other
     * @tparam     G1     Mass unit of the other
     * @tparam     S1     Time unit of the other
     *
     * @return     The ratio of the values
     */
    template <int C1, int G1, int S1>
    auto operator/(dimensional_value_t<C1, G1, S1, T> other) const
    {
        return dimensional_value_t<C - C1, G - G1, S - S1, T> { value / other.value };
    }


    /**
     * @brief      Return the negation of this value
     *
     * @return     The value
     */
    auto operator-() const
    {
        return dimensional_value_t { -value };
    }


    /**
     * @brief      Multiply a dimensionless number
     *
     * @param[in]  scale  The scale
     *
     * @return     A scaled quantity
     */
    auto operator*(T scale) const
    {
        return dimensional_value_t { value * scale };
    }


    /**
     * @brief      Divide a dimensionless number
     *
     * @param[in]  inverse_scale  The number to divide by
     *
     * @return     A scaled quantity
     */
    auto operator/(T inverse_scale) const
    {
        return dimensional_value_t { value / inverse_scale };
    }


    /**
     * @brief      Return the ratio of this value and another
     *
     * @param[in]  other  The other value
     *
     * @return     The ratio
     */
    T operator/(dimensional_value_t other) const
    {
        return value / other.value;
    }


    /**
     * @brief      Take the power of this dimensional number, preserving the
     *             units.
     *
     * @tparam     E     The exponent
     *
     * @return     value^E
     *
     * @note       Since non-integer dimensions are not supported, all units
     *             must divide the exponent, if it is negative.
     */
    template<int N, unsigned D=1>
    auto pow() const
    {
        static_assert((C * N) % D == 0, "non-integer dimensions not supported");
        static_assert((G * N) % D == 0, "non-integer dimensions not supported");
        static_assert((S * N) % D == 0, "non-integer dimensions not supported");

        return dimensional_value_t<C * N / D, G * N / D, S * N / D, T>
        {
            std::pow(value, T(N) / D)
        };
    }


    /**
     * @brief      Return a compile-time tuple of this type's length/mass/time
     *             dimensions
     *
     * @return     The tuple
     */
    constexpr std::tuple<int, int, int> dimensions() const
    {
        return std::make_tuple(C, G, S);
    }


    auto operator+=(dimensional_value_t<C, G, S, T> other) { value += other.value; return *this; }
    auto operator-=(dimensional_value_t<C, G, S, T> other) { value -= other.value; return *this; }
    auto operator==(dimensional_value_t<C, G, S, T> other) const { return value == other.value; }
    auto operator!=(dimensional_value_t<C, G, S, T> other) const { return value != other.value; }
    auto operator>=(dimensional_value_t<C, G, S, T> other) const { return value >= other.value; }
    auto operator<=(dimensional_value_t<C, G, S, T> other) const { return value <= other.value; }
    auto operator>(dimensional_value_t<C, G, S, T> other) const { return value > other.value; }
    auto operator<(dimensional_value_t<C, G, S, T> other) const { return value < other.value; }


    T value = 0;
};




//=============================================================================
template<int C, int G, int S, typename T>
auto mara::make_dimensional(T value) { return dimensional_value_t<C, G, S, T> { value }; }
template<typename T> auto mara::make_scalar      (T value) { return make_dimensional< 0, 0, 0>(value); }
template<typename T> auto mara::make_length      (T value) { return make_dimensional< 1, 0, 0>(value); }
template<typename T> auto mara::make_mass        (T value) { return make_dimensional< 0, 1, 0>(value); }
template<typename T> auto mara::make_time        (T value) { return make_dimensional< 0, 0, 1>(value); }
template<typename T> auto mara::make_rate        (T value) { return make_dimensional< 0, 0,-1>(value); }
template<typename T> auto mara::make_area        (T value) { return make_dimensional< 2, 0, 0>(value); }
template<typename T> auto mara::make_volume      (T value) { return make_dimensional< 3, 0, 0>(value); }
template<typename T> auto mara::make_velocity    (T value) { return make_dimensional< 1, 0,-1>(value); }
template<typename T> auto mara::make_flux        (T value) { return make_dimensional<-2, 1,-1>(value); }
template<typename T> auto mara::make_flow        (T value) { return make_dimensional< 0, 1,-1>(value); }
template<typename T> auto mara::make_mass_density(T value) { return make_dimensional<-3, 1, 0>(value); }
template<typename T> auto mara::make_flow_density(T value) { return make_dimensional<-3, 1,-1>(value); }




//=============================================================================
template<int C, int G, int S, typename T>
auto mara::to_string(dimensional_value_t<C, G, S, T> x)
{
    return std::to_string(x.value) + " (" +
    std::to_string(std::get<0>(x.dimensions())) + " " +
    std::to_string(std::get<1>(x.dimensions())) + " " +
    std::to_string(std::get<2>(x.dimensions())) + ")";
}
