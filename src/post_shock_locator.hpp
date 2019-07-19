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
#include "core_ndarray_ops.hpp"




//=============================================================================
namespace mara
{
    template<typename PrimitiveArray1d>
    auto find_shock_index(PrimitiveArray1d primitive, double gamma_law_index);

    template<typename ArrayType>
    auto find_index_of_maximum_behind(ArrayType array, std::size_t index);

    template<typename PrimitiveArray1d>
    auto find_index_of_maximum_pressure_behind(PrimitiveArray1d primitive, std::size_t index);

    template<typename PrimitiveArray1d>
    auto find_index_of_pressure_plateau_ahead(PrimitiveArray1d primitive, std::size_t index);
}




/**
 * @brief      Return the index of a likely shock in a 1d array of primitive
 *             variables.
 *
 * @param[in]  primitive         The primitives, must provide a method with the
 *                               signature double specific_entropy(double
 *                               gamma_law_index).
 * @param[in]  gamma_law_index   The gamma law index
 *
 * @tparam     PrimitiveArray1d  The type of the array
 *
 * @return     The index of a shock
 *
 * @note       This function works by finding the global minimum of the entropy
 *             derivative.
 */
template<typename PrimitiveArray1d>
auto mara::find_shock_index(PrimitiveArray1d primitive, double gamma_law_index)
{
    using namespace std::placeholders;

    auto s0 = primitive | nd::map([gamma_law_index] (auto p) { return p.specific_entropy(gamma_law_index); });
    auto ds = s0 | nd::difference_on_axis(0);
    auto shock_index = nd::where(ds == nd::min(ds)) | nd::read_index(0);
    return shock_index;
}





/**
 * @brief      Return the index of a local maximum in the array at indexes
 *             smaller than the one given.
 *
 * @param[in]  array      The array of values
 * @param[in]  index      The index at which to begin scanning backwards
 *
 * @tparam     ArrayType  The array type
 *
 * @return     The index containing the first local maximum to the left of the
 *             given index
 */
template<typename ArrayType>
auto mara::find_index_of_maximum_behind(ArrayType array, std::size_t index)
{
    auto y = array | nd::bounds_check();

    try {
        while (y(index - 1) > y(index))
        {
            --index;
        }
        return index;
    }
    catch (const std::exception& e)
    {
        return std::size_t(0);
    }
}




/**
 * @brief      Return the index of a local maximum in the pressure at indexes
 *             smaller than the one given.
 *
 * @param[in]  primitive         A 1d array of primitive states
 * @param[in]  index             The index at which to begin scanning backwards
 *
 * @tparam     PrimitiveArray1d  The type of the array
 *
 * @return     The index
 */
template<typename PrimitiveArray1d>
auto mara::find_index_of_maximum_pressure_behind(PrimitiveArray1d primitive, std::size_t index)
{
    return find_index_of_maximum_behind(primitive | nd::map([] (auto p) { return p.gas_pressure(); }), index);
}




/**
 * @brief      Return the index of a "kink" in the pressure at indexes greater
 *             than the one given.
 *
 * @param[in]  primitive         A 1d array of primitive states
 * @param[in]  index             The index at which to begin scanning forwards
 *
 * @tparam     PrimitiveArray1d  The type of the array
 *
 * @return     The index
 */
template<typename PrimitiveArray1d>
auto mara::find_index_of_pressure_plateau_ahead(PrimitiveArray1d primitive, std::size_t index)
{
    auto dlogp = primitive
    | nd::map([] (auto p) { return p.gas_pressure(); })
    | nd::map([] (auto p) { return std::log(p); })
    | nd::difference_on_axis(0)
    | nd::bounds_check();

    try {
        while (dlogp(index - 1) < 0.5 * dlogp(index - 2))
        {
            ++index;
        }
        return index;
    }
    catch (const std::exception& e)
    {
        // std::printf("find_index_of_pressure_plateau_ahead: %s\n", e.what());
        return std::size_t(0);
    }
}
