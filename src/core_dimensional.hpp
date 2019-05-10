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




//=============================================================================
namespace mara
{
    template<int C, int G, int S, typename T> struct dimensional_value_t;
    template<int C, int G, int S, typename T> auto to_string(dimensional_value_t<C, G, S, T> x);
    template<int C, int G, int S, typename T> auto make_dimensional(T value);
    template<typename T> auto make_length(T value);
    template<typename T> auto make_mass  (T value);
    template<typename T> auto make_time  (T value);
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
     * @brief      Multiply a dimensionless number
     *
     * @param[in]  scale  The scale
     *
     * @return     A scaled quantity
     */
    auto operator*(double scale) const
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
    auto operator/(double inverse_scale) const
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
    double operator/(dimensional_value_t other) const
    {
        return value / other.value;
    }


    /**
     * @brief      Convert this value to a double, if it's dimensionless
     */
    // template<typename = typename std::enable_if<C == 0 && G == 0 && S == 0>>
    operator double() const
    {
        static_assert(C == 0 && G == 0 && S == 0, "cannot convert dimensional value to scalar");
        return value;
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


    double value;
};




//=============================================================================
template<int C, int G, int S, typename T>
auto mara::to_string(dimensional_value_t<C, G, S, T> x)
{
    return std::to_string(x.value) + " (" +
    std::to_string(std::get<0>(x.dimensions())) + " " +
    std::to_string(std::get<1>(x.dimensions())) + " " +
    std::to_string(std::get<2>(x.dimensions())) + ")";
}

template<int C, int G, int S, typename T>
auto mara::make_dimensional(T value) { return dimensional_value_t<C, G, S, T> { value }; }
template<typename T> auto mara::make_length(T value) { return make_dimensional<1, 0, 0>(value); }
template<typename T> auto mara::make_mass  (T value) { return make_dimensional<0, 1, 0>(value); }
template<typename T> auto mara::make_time  (T value) { return make_dimensional<0, 0, 1>(value); }
