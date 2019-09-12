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
#include <optional>
#include <variant>
#include <future>
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

    template<typename... Args>
    auto make_tree_index(Args...);

    template<std::size_t Rank, typename ValueType>
    auto tree_of(ValueType value);

    template<std::size_t Rank, typename ValueType>
    auto tree_of(const arithmetic_sequence_t<ValueType, 1 << Rank>& child_values);

    inline std::size_t hilbert_index(tree_index_t<2> index);
    inline std::size_t global_hilbert_index(tree_index_t<2> index);


    //=========================================================================
    namespace detail
    {
        template<typename T>
        static auto to_shared_ptr(T&& value)
        {
            return std::make_shared<std::decay_t<T>>(std::forward<T>(value));
        }
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




    /**
     * @brief      Determine if this is a valid index (whether it is in-bounds
     *             on its level).
     *
     * @return     True or false
     */
    bool valid() const
    {
        return (coordinates < (1 << level)).all();
    }




    /**
     * @brief      Transform this index, so that it points to the same node as
     *             it does now, but as seen by the child of this node which is
     *             in the direction of the target node.
     *
     * @return     The index with level - 1 and the coordinates offset according
     *             to the orthant value
     */
    tree_index_t advance_level() const
    {
        return {level - 1, coordinates - orthant() * (1 << (level - 1))};
    }




    /**
     * @brief      Return this index as it is relative to the parent block.
     *
     * @return     The index as seen by the parent.
     */
    tree_index_t relative_to_parent() const
    {
        return {1, coordinates.map([] (auto c) { return c % 2; })};
    }




    /**
     * @brief      Return the tree index at level - 1 which contains this one as
     *             a child.
     *
     * @return     The parent index
     */
    tree_index_t parent_index() const
    {
        return {level - 1, coordinates / 2};
    }




    /**
     * @brief      Return a sequence of tree indexes corresponding to the 2^Rank
     *             indexes of this node's children.
     *
     * @return     A sequence of tree indexes.
     */
    arithmetic_sequence_t<tree_index_t, 1 << Rank> child_indexes() const
    {
        return iota<1 << Rank>().map([this] (auto i) { return tree_index_t{level + 1, coordinates * 2 + binary_repr<Rank>(i)}; });
    }




    /**
     * @brief      Return the number of indexes in all levels below a given indexes
     *             level, assuming a complete tree
     *
     * @return     The number of indexes
     */
    std::size_t indexes_below()
    {
        std::size_t lvl = level, ibelow = 1;
        while (lvl > 1)
        {
            ibelow += pow(1 << Rank, level-1);
            lvl--;
        }
        return ibelow;
    }




    /**
     * @brief      Return a new index with the same coordinates but a different
     *             level.
     *
     * @param[in]  new_level  The level of the returned index
     *
     * @return     A new index
     */
    tree_index_t with_level(std::size_t new_level) const
    {
        return {new_level, coordinates};
    }




    /**
     * @brief      Return the orthant (ray, quadrant, octant) of this index.
     *
     * @return     A sequence of bool's
     *
     * @note       https://en.wikipedia.org/wiki/Orthant
     */
    arithmetic_sequence_t<bool, Rank> orthant() const
    {
        return coordinates.map([this] (auto x) -> bool { return x / (1 << (level - 1)); });
    }




    /**
     * @brief      Return an index shifted up on the given axis. The index is
     *             automatically wrapped to the min / max value at this level.
     *
     * @param[in]  a     The axis to shift on
     *
     * @return     A new index, on the same level
     */
    tree_index_t next_on(std::size_t a) const { return {level, coordinates.set(a, (coordinates.at(a) + (1 << level) + 1) % (1 << level))}; }
    tree_index_t prev_on(std::size_t a) const { return {level, coordinates.set(a, (coordinates.at(a) + (1 << level) - 1) % (1 << level))}; }




    //=========================================================================
    bool operator==(const tree_index_t& other) const { return level == other.level && (coordinates == other.coordinates).all(); }
    bool operator!=(const tree_index_t& other) const { return level != other.level || (coordinates != other.coordinates).any(); }




    //=========================================================================
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
     * @brief      Return the numer of leafs in the tree.
     *
     * @return     The tree size
     */
    std::size_t size() const
    {
        return has_value() ? 1 : children().map([] (auto&& c) { return c.size(); }).sum();
    }




    /**
     * @brief      Return the tree's maximum depth below this node.
     *
     * @param[in]  d     Starting depth (mainly for internal use)
     *
     * @return     The tree's depth
     */
    std::size_t depth(std::size_t d=0) const
    {
        return has_value() ? d : children().map([d] (auto&& c) { return c.depth(d + 1); }).max();
    }




    /**
     * @brief      Return the value that would be hit first in a depth-first
     *             traversal.
     *
     * @return     The value
     */
    const ValueType& front() const
    {
        return has_value() ? value() : children().at(0).front();
    }




    /**
     * @brief      Determines if this node has a value.
     *
     * @return     True if this node is a leaf, false otherwise
     */
    bool has_value() const
    {
        return __impl.index() == 0;
    }




    /**
     * @brief      Get the value of this node if it is a leaf, or throw
     *             out-of-range otherwise.
     *
     * @return     The value
     */
    const value_type& value() const
    {
        return std::get<0>(__impl);
    }




    /**
     * @brief      Get the children of this node if it is not a leaf, or throw
     *             an out-of-range otherwise.
     *
     * @return     The array of children.
     */
    const children_array_type& children() const
    {
        return *std::get<1>(__impl);
    }




    /**
     * @brief      Return the child at a binary orthant in this node, if it is
     *             not a leaf. Throws out-of-range if this is a leaf node.
     *
     * @param[in]  orthant  which ray, quadrant, octant, etc.
     *
     * @return     The child node
     *
     * @note       https://en.wikipedia.org/wiki/Orthant
     */
    const arithmetic_binary_tree_t& child_at(const arithmetic_sequence_t<bool, Rank>& orthant) const
    {
        return children()[mara::to_integral(orthant)];
    }




    /**
     * @brief      Convenience method for the one above. Throws out-of-range if
     *             this is a leaf node.
     *
     * @param[in]  args  The bits (one per dimension) of the child node
     *
     * @tparam     Args  The argument types
     *
     * @return     The child node, if this node is not a leaf
     */
    template<typename... Args>
    const arithmetic_binary_tree_t& child_at(Args... args) const
    {
        return child_at(make_sequence(bool(args)...));
    }




    /**
     * @brief      Return the node at an arbitrarily depth path below this node
     *             using a sequence of bits. The sequence contains one bit per
     *             dimension per level to traverse. Throws out-of-range if the
     *             bit sequence is non-empty and this node is a leaf.
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
     * @brief      Like the above, except the target node is identified by a
     *             tree index rather than a bit sequence. Throws out-of-range if
     *             the index is non-zero and this node is a leaf.
     *
     * @param[in]  index  The target index (level, coordinates)
     *
     * @return     A node
     */
    const arithmetic_binary_tree_t& node_at(const tree_index_t<Rank>& index) const
    {
        if (index.level == 0)
        {
            if (index.coordinates.any())
            {
                throw std::out_of_range("mara::arithmetic_binary_tree_t::node_at");
            }
            return *this;
        }
        if (has_value())
        {
            throw std::out_of_range("mara::arithmetic_binary_tree_t::node_at");
        }
        return child_at(index.orthant()).node_at(index.advance_level());
    }




    /**
     * @brief      Determines if the tree contains a node (leaf or otherwise) at
     *             the specified index.
     *
     * @param[in]  index  The target index
     *
     * @return     True if the tree as a node there, false otherwise.
     */
    bool contains_node(const tree_index_t<Rank>& index) const
    {
        if (index.level == 0)
        {
            return ! index.coordinates.any();
        }
        if (has_value())
        {
            return false;
        }
        return child_at(index.orthant()).contains_node(index.advance_level());
    }




    /**
     * @brief      Return a reference to the value at the given index. Throws
     *             out-of-range if no leaf node exists at that index.
     *
     * @param[in]  index  The target index
     *
     * @return     A const reference to the value
     */
    const value_type& at(const tree_index_t<Rank>& index) const
    {
        return node_at(index).value();
    }




    /**
     * @brief      Return optional<value_type> for the given index. Can't fail.
     *
     * @param[in]  index  The target index
     *
     * @return     The value, if it exists
     */
    std::optional<value_type> find(const tree_index_t<Rank>& index) const
    {
        try {
            return at(index);
        }
        catch (const std::out_of_range&)
        {
            return {};
        }
    }




    /**
     * @brief      Determine whether a value exists at the given tree index.
     *
     * @param[in]  index  The index to check
     *
     * @return     True or false
     */
    bool contains(const tree_index_t<Rank>& index) const
    {
        if (has_value())
        {
            return index.level == 0 && ! index.coordinates.any();
        }
        return child_at(index.orthant()).contains(index.advance_level());
    }




    bool      any() const { return has_value() ? bool(value()) : children().map([] (auto&& c) { return c.any(); }).any(); }
    bool      all() const { return has_value() ? bool(value()) : children().map([] (auto&& c) { return c.all(); }).all(); }
    ValueType min() const { return has_value() ?      value()  : children().map([] (auto&& c) { return c.min(); }).min(); }
    ValueType max() const { return has_value() ?      value()  : children().map([] (auto&& c) { return c.max(); }).max(); }
    ValueType sum() const { return has_value() ?      value()  : children().map([] (auto&& c) { return c.sum(); }).sum(); }




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
    auto indexes(const tree_index_t<Rank>& index_in_parent={}) const
    -> arithmetic_binary_tree_t<tree_index_t<Rank>, Rank>
    {
        if (has_value())
        {
            return tree_of<Rank>(index_in_parent);
        }
        return {detail::to_shared_ptr(index_in_parent
                .child_indexes()
                .map([] (auto i) { return [i] (auto&& c) { return c.indexes(i); }; })
                .apply_to(children()))};
    }



    /**
     * @brief      Invoke a function on the value of each leaf node at or below
     *             this one.
     *
     * @param      fn        The function to invoke.
     *
     * @tparam     Function  The function type
     *
     * @note       This function is useful in outputting a tree to e.g. HDF5
     *             format.
     */
    template<typename Function>
    void sink(Function&& fn) const
    {
        if (has_value())
        {
            fn(value());
        }
        else
        {
            for (const auto& c : children())
            {
                c.sink(fn);
            }
        }
    }




    /**
     * @brief      Map a function over the values of this tree.
     *
     * @param      fn              The function to map
     *
     * @tparam     Function        The type of the function object
     * @tparam     ResultTreeType  The result tree type (deduced)
     *
     * @return     A new tree, with the same shape as this one, and its values
     *             transformed element-wise.
     */
    template<typename Function,
             typename ResultTreeType = arithmetic_binary_tree_t<std::invoke_result_t<Function, ValueType>, Rank>>
    auto map(Function&& fn) const -> ResultTreeType
    {
        if (has_value())
        {
            return {fn(value())};
        }
        return {detail::to_shared_ptr(children().map([fn] (auto&& t) { return t.map(fn); }))};
    }




    /**
     * @brief      Overload of map, taking an async execution policy.
     *
     * @param      fn           The function to map asynchronously
     * @param[in]  launch_mode  The launch mode: std::launch::async or
     *                          std::launch::deferred
     *
     * @tparam     Function     The function type
     *
     * @return     A tree identical to the result of map above, but evaluated on
     *             separate threads (or a single thread if deferred is given as
     *             the launch_mode)
     */
    template<typename Function>
    auto map(Function&& fn, std::launch launch_mode)
    {
        return map([fn, launch_mode] (auto&& v)
        {
            return std::make_shared
            <std::future
            <std::invoke_result_t<Function, ValueType>>>
            (std::async(launch_mode, fn, std::forward<decltype(v)>(v)));
        })
        .map([] (auto&& future_ptr) { return future_ptr->get(); });
    }




    /**
     * @brief      Convenience function for mapping a function of multiple
     *             arguments over a tree of tuple values:
     *             
     *             A.pair(B).apply([] (auto a, auto b) { return ...; })
     *
     * @param      fn        The function to map
     *
     * @tparam     Function  The function type
     *
     * @return     A new tree
     */
    template<typename Function>
    auto apply(Function&& fn) const
    {
        return map([fn] (auto&& t) { return std::apply(fn, t); });
    }

    template<typename Function>
    auto apply(Function&& fn, std::launch launch_mode)
    {
        return map([fn] (auto&& t) { return std::apply(fn, t); }, launch_mode);
    }




    /**
     * @brief      Return a tree like this one, but with the node x at the given
     *             index replaced by x.map(fn);
     *
     * @param[in]  index     The index of the node to update
     * @param      fn        The function to be mapped
     *
     * @tparam     Function  The function type
     *
     * @return     A new tree, with the same shape and value type as this one
     */
    template<typename Function>
    arithmetic_binary_tree_t update(const tree_index_t<Rank>& index, Function&& fn) const
    {
        if (index.level == 0)
        {
            return map(fn);
        }
        return {detail::to_shared_ptr(children().update(to_integral(index.orthant()), [&] (auto&& c)
        {
            return c.update(index.advance_level(), fn);
        }))};
    }




    /**
     * @brief      Insert or replace a value at the given index, creating
     *             intermediate nodes as necessary. Throws an exception if a
     *             non-leaf node already exists at the target index.
     *
     * @param[in]  index  The target index
     * @param[in]  value  The value to insert at that index
     *
     * @return     A new tree
     *
     * @note       This method may generate nodes with default-constructed
     *             values at indexes other than the target index, so the value
     *             type must be default-constructible. Its most likely use is in
     *             loading data into the tree from a file.
     */
    arithmetic_binary_tree_t insert(const tree_index_t<Rank>& index, const ValueType& value) const
    {
        if (index.level == 0)
        {
            if (index.coordinates.any() || ! has_value())
            {
                throw std::out_of_range("mara::arithmetic_binary_tree_t::insert (target node already has children)");
            }
            return arithmetic_binary_tree_t{value};
        }
        if (has_value())
        {
            return tree_of<Rank>(mara::iota<1 << Rank>().map([] (auto&&) { return ValueType(); })).insert(index, value);
        }
        return {detail::to_shared_ptr(children().update(to_integral(index.orthant()), [next=index.advance_level(), &value] (auto&& c)
        {
            return c.insert(next, value);
        }))};
    }




    /**
     * @brief      Map a binary function over this tree and another of the same
     *             shape.
     *
     * @param[in]  other           The other tree
     * @param      fn              The binary function
     *
     * @tparam     OtherType       The value type of the other tree
     * @tparam     BinaryOperator  The type of the binary operator
     * @tparam     ResultTreeType  The resulting type (deduced)
     *
     * @return     The new tree
     */
    template<typename OtherType,
             typename BinaryOperator,
             typename ResultTreeType = arithmetic_binary_tree_t<std::invoke_result_t<BinaryOperator, ValueType, OtherType>, Rank>>
    auto binary_op(const arithmetic_binary_tree_t<OtherType, Rank>& other, BinaryOperator&& fn) const -> ResultTreeType
    {
        if (has_value())
        {
            return {fn(value(), other.value())};
        }
        if (other.has_value())
        {
            throw std::invalid_argument("mara::arithmetic_binary_tree_t::binary_op (trees have different shapes)");
        }
        return {detail::to_shared_ptr(iota<1 << Rank>().map([this, &fn, &other] (std::size_t n)
        {
            return children()[n].binary_op(other.children()[n], std::forward<BinaryOperator>(fn));
        }))};
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
    template<typename OtherType>
    auto pair(const arithmetic_binary_tree_t<OtherType, Rank>& other) const
    {
        return binary_op(other, [] (auto&& a, auto&& b) { return std::make_pair(a, b); });
    }




    /**
     * @brief      Convenience function for tree.indexes().pair(tree), which is
     *             a pattern that comes up whenever an operation on the tree
     *             values depends on the values in neighboring nodes.
     *
     * @return     A tree of (index, value) pairs
     */
    auto pair_indexes() const
    {
        return indexes().pair(*this);
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
    template<typename OtherType>
    auto apply_to(const arithmetic_binary_tree_t<OtherType, Rank>& other) const
    {
        return pair(other).apply([] (auto&& fn, auto&& arg) { return fn(arg); });
    }




    /**
     * @brief      Apply a bifurcate function to the values of this tree that
     *             satisfy a predicate. The function must return sequences whose
     *             value type is the same as the tree's value type.
     *
     * @param      predicate  The predicate to test the leaf values against:
     *                        value_type -> boolean
     * @param      bifurcate  The bifurcate function to apply to the values:
     *                        value_type -> sequence<value_type>
     *
     * @tparam     Predicate  The type of the predicate function
     * @tparam     Bifurcate  The type of the bifurcate function
     *
     * @return     A tree with some bifurcated leafs
     */
    template<typename Predicate, typename Bifurcate>
    auto bifurcate_if(Predicate&& predicate, Bifurcate&& bifurcate) const -> arithmetic_binary_tree_t
    {
        if (has_value())
        {
            return predicate(value()) ? tree_of<Rank>(bifurcate(value())) : *this;
        }
        return {detail::to_shared_ptr(children().map([&] (auto&& c) { return c.bifurcate_if(predicate, bifurcate); }))};
    }




    /**
     * @brief      Apply a function to all the values in this tree, mapping them
     *             into sequences. Since all the node values are replaced, the
     *             bifurcate function may return sequences of any value type.
     *
     * @param      bifurcate       The bifurcate function to apply to the
     *                             values: value_type -> sequence<T>
     *
     * @tparam     Bifurcate       The type of the bifurcate function
     * @tparam     ResultTreeType  The function's return value (deduced)
     *
     * @return     A tree with all leafs turned into sequences
     */
    template<typename Bifurcate,
             typename ResultTreeType = arithmetic_binary_tree_t<typename std::invoke_result_t<Bifurcate, ValueType>::value_type, Rank>>
    auto bifurcate_all(Bifurcate&& bifurcate) const -> ResultTreeType
    {
        if (has_value())
        {
            return tree_of<Rank>(bifurcate(value()));
        }
        return {detail::to_shared_ptr(children().map([&] (auto&& c) { return c.bifurcate_all(bifurcate); }))};
    }




    //=========================================================================
    template<typename T> auto operator+ (const T& v) const { return map([&v] (auto&& t) { return t +  v; }); }
    template<typename T> auto operator- (const T& v) const { return map([&v] (auto&& t) { return t -  v; }); }
    template<typename T> auto operator* (const T& v) const { return map([&v] (auto&& t) { return t *  v; }); }
    template<typename T> auto operator/ (const T& v) const { return map([&v] (auto&& t) { return t /  v; }); }
    template<typename T> auto operator&&(const T& v) const { return map([&v] (auto&& t) { return t && v; }); }
    template<typename T> auto operator||(const T& v) const { return map([&v] (auto&& t) { return t || v; }); }
    template<typename T> auto operator==(const T& v) const { return map([&v] (auto&& t) { return t == v; }); }
    template<typename T> auto operator!=(const T& v) const { return map([&v] (auto&& t) { return t != v; }); }
    template<typename T> auto operator<=(const T& v) const { return map([&v] (auto&& t) { return t <= v; }); }
    template<typename T> auto operator>=(const T& v) const { return map([&v] (auto&& t) { return t >= v; }); }
    template<typename T> auto operator< (const T& v) const { return map([&v] (auto&& t) { return t <  v; }); }
    template<typename T> auto operator> (const T& v) const { return map([&v] (auto&& t) { return t >  v; }); }

    template<typename T> auto operator+ (const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::plus<>()); }
    template<typename T> auto operator- (const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::minus<>()); }
    template<typename T> auto operator* (const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::multiplies<>()); }
    template<typename T> auto operator/ (const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::divides<>()); }
    template<typename T> auto operator&&(const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::logical_and<>()); }
    template<typename T> auto operator||(const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::logical_or<>()); }
    template<typename T> auto operator==(const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::equal_to<>()); }
    template<typename T> auto operator!=(const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::not_equal_to<>()); }
    template<typename T> auto operator<=(const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::less_equal<>()); }
    template<typename T> auto operator>=(const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::greater_equal<>()); }
    template<typename T> auto operator< (const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::less<>()); }
    template<typename T> auto operator> (const arithmetic_binary_tree_t<T, Rank>& v) const { return binary_op(v, std::greater<>()); }

    auto operator+() const { return map([] (auto&& x) { return +x; }); }
    auto operator-() const { return map([] (auto&& x) { return -x; }); }




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
 * @brief      Construct a tree index at the given coordinates, with inferred
 *             rank. The level is initialized to zero, so you'll probably use
 *             this like:
 *
 *             auto index = mara::make_tree_index(5, 6, 7).at_level(3);
 *
 * @param[in]  args  The coordinatrs
 *
 * @tparam     Args  The arg types
 *
 * @return     A new index
 */
template<typename... Args>
auto mara::make_tree_index(Args... args)
{
    return tree_index_t<sizeof...(Args)>{0, {std::size_t(args)...}};
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
 * @brief      Return a binary tree node, with 2^Rank children in the given
 *             sequence.
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
        detail::to_shared_ptr(child_values.map([] (auto&& c) { return tree_of<Rank>(c); }))
    };
}




/**
 * @brief      Convert the given tree index to a linear index according to a
 *             Hilbert curve
 *
 * @param[in]  index  The tree index
 *
 * @return     The Hilbert linear index of the given tree index
 */
std::size_t mara::hilbert_index(tree_index_t<2> index)
{
    // https://en.wikipedia.org/wiki/Hilbert_curve

    auto rot = [] (int n, int *x, int *y, int rx, int ry)
    {
        if (ry == 0)
        {
            if (rx == 1)
            {
                *x = n-1 - *x;
                *y = n-1 - *y;
            }
            int t = *x;
            *x = *y;
            *y = t;
        }
    };

    auto xy2d = [rot] (int n, int x, int y)
    {
        int rx, ry, s, d=0;

        for (s = n/2; s > 0; s /= 2)
        {
            rx = (x & s) > 0;
            ry = (y & s) > 0;
            d += s * s * ((3 * rx) ^ ry);
            rot(n, &x, &y, rx, ry);
        }
        return d;
    };

    return xy2d(index.level, index.coordinates[0], index.coordinates[1]);
}

std::size_t mara::global_hilbert_index(tree_index_t<2> index)
{
    return mara::hilbert_index(index) + index.indexes_below();
}
