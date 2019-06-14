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
    template<std::size_t Rank>
    struct tree_index_t;

    template<typename ValueType, std::size_t Rank>
    struct arithmetic_binary_tree_t;

    template<std::size_t Bits>
    auto binary_repr(std::size_t value);

    template<std::size_t Bits>
    std::size_t to_integral(const arithmetic_sequence_t<bool, Bits>& steps);

    template<std::size_t Rank, typename ValueType>
    auto tree_of(ValueType value);

    template<std::size_t Rank, typename ValueType>
    auto tree_of(const arithmetic_sequence_t<ValueType, 1 << Rank>& child_values);
}

namespace mara::tree::detail
{
    template<typename T>
    static auto to_shared_ptr(T&& value)
    {
        return std::make_shared<std::decay_t<T>>(std::forward<T>(value));
    }
}




/**
 * @brief      A struct that identifies a node's global position in the
 *             tree: its level, and its coordinates with respect to the
 *             origin at its level.
 */
template<std::size_t Rank>
struct mara::tree_index_t
{
    bool operator==(const tree_index_t& other) const { return level == other.level && (coordinates == other.coordinates).all(); }
    std::size_t level = 0;
    arithmetic_sequence_t<std::size_t, Rank> coordinates = {};
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct mara::arithmetic_binary_tree_t
{




    //=========================================================================
    using value_type = ValueType;
    using children_array_type = mara::arithmetic_sequence_t<arithmetic_binary_tree_t, 1 << Rank>;
    template<std::size_t Depth> using bit_path_t = arithmetic_sequence_t<bool, Depth>;
    template<std::size_t Depth> using bit_path_nd_t = arithmetic_sequence_t<bit_path_t<Depth>, Rank>;




    /**
     * @brief      Return the numer of nodes at and below this one.
     *
     * @return     True if has value, False otherwise.
     */
    std::size_t size() const
    {
        return has_value() ? 1 : children().map([] (auto&& c) { return c.size(); }).sum();
    }




    /**
     * @brief      Determines if this node has a value.
     *
     * @return     True if has value, False otherwise.
     */
    bool has_value() const
    {
        return __impl.index() == 0;
    }




    /**
     * @brief      Get the value of this node if it is a leaf, or throw an
     *             exception otherwise.
     *
     * @return     The value
     */
    const value_type& value() const
    {
        return std::get<0>(__impl);
    }




    /**
     * @brief      Get the children of this node if it is not a leaf, or throw
     *             an exception otherwise.
     *
     * @return     The array of children.
     */
    const children_array_type& children() const
    {
        return *std::get<1>(__impl);
    }




    /**
     * @brief      Return the child at a binary location in this node, if it is
     *             not a leaf. If it's a leaf then throw an exception.
     *
     * @param[in]  location  The binary location e.g. (0, 1, 0) for a 3d tree
     *
     * @return     The child node
     */
    const arithmetic_binary_tree_t& child_at(const arithmetic_sequence_t<bool, Rank>& location) const
    {
        return children()[mara::to_integral(location)];
    }




    /**
     * @brief      Convenience method for the one above.
     */
    template<typename... Args>
    const arithmetic_binary_tree_t& child_at(Args... args) const
    {
        return child_at(make_sequence(bool(args)...));
    }




    /**
     * @brief      Return the node at an arbitrarily deep binary path below this
     *             node.
     *
     * @param[in]  path   A binary path of arbitrary depth
     *
     * @tparam     Depth  The length of the path
     *
     * @return     A node, if one exists at the path
     */
    template<std::size_t Depth>
    const arithmetic_binary_tree_t& node_at(const bit_path_nd_t<Depth>& path) const
    {
        if constexpr (Depth == 0)
        {
            return *this;
        }
        else
        {
            return child_at(path.head().transpose()).node_at(path.tail());
        }
    }




    /**
     * @brief      Return a tree of tree_index_t, mirroring the structure of
     *             this tree, where the value of each child node is its global
     *             index (level, coordinates) in the tree.
     *
     * @param[in]  index_in_parent  The starting index (generally you'd omit
     *                              this argument)
     *
     * @return     A tree of indexes
     */
    auto indexes(tree_index_t<Rank> index_in_parent={}) const
    {
        if (has_value())
        {
            return tree_of<Rank>(index_in_parent);
        }
        return arithmetic_binary_tree_t<tree_index_t<Rank>, Rank>{
            tree::detail::to_shared_ptr(
            iota<1 << Rank>()
            .map([this, index_in_parent] (std::size_t n)
            {
                return children()[n].indexes({
                    index_in_parent.level + 1,
                    index_in_parent.coordinates * 2 + binary_repr<Rank>(n),
                });
            }))
        };
    }




    /**
     * @brief      Map a function over the values of this tree.
     *
     * @param      fn        The function to map
     *
     * @tparam     Function  The type of the function object
     *
     * @return     A new tree, with the same shape as this one, and its values
     *             transformed element-wise.
     */
    template<typename Function,
             typename ResultTreeType = arithmetic_binary_tree_t<std::invoke_result_t<Function, ValueType>, Rank>>
    auto map(Function&& fn) const -> ResultTreeType
    {
        return has_value()
        ? ResultTreeType{fn(value())}
        : ResultTreeType{tree::detail::to_shared_ptr(children().map([fn] (auto&& t) { return t.map(fn); }))};
    }




    /**
     * @brief      Return a tree of values by applying this tree of functions to
     *             it, if this is a tree of functions.
     *
     * @param[in]  other      A tree of arguments given to this array of (unary)
     *                        functions
     *
     * @tparam     OtherType  The value type of the argument tree
     *
     * @return     A new tree
     *
     * @note       This method is conventionally referred to as "ap" in
     *             functional programming. With respect to this method, a tree
     *             is an "applicative functor".
     */
    template<typename OtherType,
             typename ResultTreeType = arithmetic_binary_tree_t<std::invoke_result_t<ValueType, OtherType>, Rank>>
    auto apply_to(const arithmetic_binary_tree_t<OtherType, Rank>& other) const -> ResultTreeType
    {
        try {
            return has_value()
            ? ResultTreeType{value()(other.value())}
            : ResultTreeType{tree::detail::to_shared_ptr(
            iota<1 << Rank>()
            .map([this, &other] (std::size_t n)
            {
                return children()[n].apply_to(other.children()[n]);
            }))};
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("mara::arithmetic_binary_tree_t::apply_to (differently shaped trees)");
        }
    }




    /**
     * @brief      Pair this tree with another one of the same shape.
     *
     * @param[in]  other      The other tree
     *
     * @tparam     OtherType  The value type of the other tree
     *
     * @return     A tree of tuples with the same shape as this one
     *
     * @note       Only the rank can be ensured correct at compile-time. This
     *             function will throw an exception if the shapes of this tree
     *             and the other one are different.
     */
    template<typename OtherType,
             typename ResultTreeType = arithmetic_binary_tree_t<std::pair<ValueType, OtherType>, Rank>>
    auto pair(const arithmetic_binary_tree_t<OtherType, Rank>& other) const -> ResultTreeType
    {
        try {
            return has_value()
            ? ResultTreeType{std::make_pair(value(), other.value())}
            : ResultTreeType{tree::detail::to_shared_ptr(
            iota<1 << Rank>()
            .map([this, &other] (std::size_t n)
            {
                return children()[n].pair(other.children()[n]);
            }))};
        }
        catch (const std::exception&)
        {
            throw std::invalid_argument("mara::arithmetic_binary_tree_t::pair (differently shaped trees)");
        }
    }




    //=========================================================================
    std::variant<ValueType, std::shared_ptr<children_array_type>> __impl;
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
    .map([] (std::size_t e) { return [e] (bool y) { return (1 << e) * y; }; })
    .apply_to(bits)
    .sum();
}




/**
 * @brief      Return a leaf binary tree of the given rank from a single value.
 *
 * @param[in]  value      The value to put in the leaf
 *
 * @tparam     Rank       The dimensionality of the tree
 * @tparam     ValueType  The tree value type
 *
 * @return     A single-node tree
 */
template<std::size_t Rank, typename ValueType>
auto mara::tree_of(ValueType value)
{
    return arithmetic_binary_tree_t<ValueType, Rank>{value};
}




/**
 * @brief      Return a binary tree with leaf children having the given values.
 *
 * @param[in]  child_values  The child values to put in the tree
 *
 * @tparam     Rank          The dimensionality of the tree
 * @tparam     ValueType     The tree value type
 *
 * @return     A tree with leaf children
 */
template<std::size_t Rank, typename ValueType>
auto mara::tree_of(const arithmetic_sequence_t<ValueType, 1 << Rank>& child_values)
{
    return arithmetic_binary_tree_t<ValueType, Rank>
    {
        tree::detail::to_shared_ptr(child_values.map([] (auto&& c) { return tree_of<Rank>(c); }))
    };
}
