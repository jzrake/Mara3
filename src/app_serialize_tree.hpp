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
#include <sstream>
#include <iomanip>
#include <cmath>
#include "core_hdf5.hpp"
#include "core_tree.hpp"




//=============================================================================
namespace mara
{
    template<std::size_t Rank>
    std::string format_tree_index(const tree_index_t<Rank>& index);

    template<std::size_t Rank>
    tree_index_t<Rank> read_tree_index(std::string str);

    template<typename ValueType, std::size_t Rank>
    void read_tree(h5::Group&& group, arithmetic_binary_tree_t<ValueType, Rank>& tree);
    template<typename ValueType, std::size_t Rank>
    void read(h5::Group& group, std::string name, arithmetic_binary_tree_t<ValueType, Rank>& tree);

    template<typename ValueType, std::size_t Rank>
    void write_tree(h5::Group&& group, const arithmetic_binary_tree_t<ValueType, Rank>& tree);
    template<typename ValueType, std::size_t Rank>
    void write(h5::Group& group, std::string name, const arithmetic_binary_tree_t<ValueType, Rank>& tree);
}




/**
 * @brief      Return a stringified tree index that can be re-parsed by the
 *             read_tree_index method.
 *
 * @param[in]  index  The tree index
 *
 * @tparam     Rank   The rank of the tree
 *
 * @return     A string, formatted like "level:i-j-k"
 */
template<std::size_t Rank>
std::string mara::format_tree_index(const tree_index_t<Rank>& index)
{
    std::stringstream ss;
    ss << index.level;

    for (std::size_t i = 0; i < Rank; ++i)
    {
        ss
        << (i == 0 ? ':' : '-')
        << std::setfill('0')
        << std::setw(1 + std::log10(1 << index.level))
        << index.coordinates[i];
    }
    return ss.str();
}




/**
 * @brief      Return a tree index from a string formatted with
 *             format_tree_index.
 *
 * @param[in]  str   The string representation of the index
 *
 * @tparam     Rank  The rank of the tree
 *
 * @return     The index
 */
template<std::size_t Rank>
mara::tree_index_t<Rank> mara::read_tree_index(std::string str)
{
    if (std::count(str.begin(), str.end(), '-') != Rank - 1)
    {
        throw std::invalid_argument("mara::read_tree_index (wrong rank)");
    }

    auto result = tree_index_t<Rank>();

    result.level = std::stoi(str.substr(0, str.find(':')));
    str.erase(0, str.find(':') + 1);

    for (std::size_t i = 0; i < Rank; ++i)
    {
        result.coordinates[i] = std::stoi(str.substr(0, str.find('-')));
        str.erase(0, str.find('-') + 1);
    }
    return result;
}




/**
 * @brief      Read a tree from an HDF5 location
 *
 * @param      group      The group to read from
 * @param      tree       The tree to modify
 *
 * @tparam     ValueType  The tree's value type
 * @tparam     Rank       The rank of the tree
 */
template<typename ValueType, std::size_t Rank>
void mara::read_tree(h5::Group&& group, arithmetic_binary_tree_t<ValueType, Rank>& tree)
{
    for (auto dataset : group)
    {
        auto value = group.read<std::size_t>(dataset);
        tree = std::move(tree).insert(mara::read_tree_index<2>(dataset), value);
    }
}

template<typename ValueType, std::size_t Rank>
void mara::read(h5::Group& group, std::string name, arithmetic_binary_tree_t<ValueType, Rank>& tree)
{
    read_tree(group.require_group(name), tree);
}




/**
 * @brief      Writes a tree to an HDF5 location
 *
 * @param      group      The group to read from
 * @param      tree       The tree to write
 *
 * @tparam     ValueType  The tree's value type
 * @tparam     Rank       The rank of the tree
 *
 * @note       The tree values are flattened, and written to a single group.
 *             Traversing deeply nested HDF5 files can be annoying, and slow.
 */
template<typename ValueType, std::size_t Rank>
void mara::write_tree(h5::Group&& group, const arithmetic_binary_tree_t<ValueType, Rank>& tree)
{
    tree.indexes().pair(tree).sink([&group] (auto&& index_and_value)
    {
        auto [index, value] = index_and_value;
        group.write(format_tree_index(index), value);
    });
}

template<typename ValueType, std::size_t Rank>
void mara::write(h5::Group& group, std::string name, const arithmetic_binary_tree_t<ValueType, Rank>& tree)
{
    write_tree(group.open_group(name), tree);
}
