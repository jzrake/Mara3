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
}




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
