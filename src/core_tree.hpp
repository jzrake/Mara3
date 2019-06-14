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
#include <functional>
#include <memory>
#include <variant>
#include "core_sequence.hpp"




//=============================================================================
namespace mara
{
    template<typename ValueType, std::size_t Rank>
    struct arithmetic_binary_tree_t;

    template<std::size_t Bits>
    auto binary_repr(std::size_t value);

    template<std::size_t Bits>
    std::size_t to_integral(const arithmetic_sequence_t<bool, Bits>& steps);
}





template<typename ValueType, std::size_t Rank>
struct mara::arithmetic_binary_tree_t
{
    using value_type = ValueType;
    using children_array_type = mara::arithmetic_sequence_t<arithmetic_binary_tree_t, 1 << Rank>;

    template<std::size_t Depth> using bits_t = arithmetic_sequence_t<bool, Depth>;
    template<std::size_t Depth> using index_t = arithmetic_sequence_t<bits_t<Depth>, Rank>;




    /**
     * @brief      Determines if this node has a value.
     *
     * @return     True if has value, False otherwise.
     */
    bool has_value() const
    {
        return impl.index() == 0;
    }




    /**
     * @brief      Get the value of this node if it is a leaf, or throw an
     *             exception otherwise.
     *
     * @return     The value
     */
    const value_type& value() const
    {
        return std::get<0>(impl);
    }




    /**
     * @brief      Get the children of this node if it is not a leaf, or throw
     *             an exception otherwise.
     *
     * @return     The array of children.
     */
    const children_array_type& children() const
    {
        return *std::get<1>(impl);
    }




    template<std::size_t Depth>
    const arithmetic_binary_tree_t& node_at(const index_t<Depth>& index) const
    {
        if constexpr (Depth == 0)
        {
            return *this;
        }
        else
        {
            return *this;
            // return child_at_index(index[0][0], index[1][0], index[2][0]).node_at(index.tail());
        }
    }



    template<typename T>
    static auto to_shared_ptr(T&& value)
    {
        return std::make_shared<std::decay_t<T>>(std::forward<T>(value));
    }

    std::variant<ValueType, std::shared_ptr<mara::arithmetic_sequence_t<ValueType, 1 << Rank>>> impl;
};






/**
 * @brief      Return a boolean sequence {a} representing a number:
 *
 *             value = a[0] + 2^0 + a[1] + 2^1 + ...
 *
 * @param[in]  value  The number to represent
 *
 * @tparam     Bits   The number of bits to include
 *
 * @return     A boolean sequence, with the most significant bit at the
 *             beginning.
 */
template<std::size_t Bits>
auto mara::binary_repr(std::size_t value)
{
    return mara::iota<Bits>()
    .map([value] (std::size_t n) { return bool(value & (1 << n)); });
}




/**
 * @brief      Turn a binary representation of a number into a 64-bit unsigned
 *             integer.
 *
 * @param[in]  bits  The bits (as returned from binary_repr)
 *
 * @tparam     Bits  The number of bits
 *
 * @return     The decimal representation
 */
template<std::size_t Bits>
std::size_t mara::to_integral(const arithmetic_sequence_t<bool, Bits>& bits)
{
    return iota<Bits>()
    .map([] (auto e) { return 1 << e; })
    .map([] (auto x) { return [x] (auto y) { return x * y; }; })
    .ap(bits)
    .sum();
}
