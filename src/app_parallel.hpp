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
#include <vector>
#include <thread>
#include "core_ndarray.hpp"
#include "core_tree.hpp"




//=============================================================================
namespace mara
{
    template<std::size_t NumThreads>
    auto evaluate_on();

    template<std::size_t Rank>
    auto propose_block_decomposition(std::size_t number_of_subdomains);

    template<std::size_t Rank>
    auto create_access_pattern_array(nd::shape_t<Rank> global_shape, nd::shape_t<Rank> blocks_shape);

    //=========================================================================
    template<typename ValueType, std::size_t Rank>
    auto build_rank_tree(mara::arithmetic_binary_tree_t<ValueType, Rank> topology, std::size_t size);

    template<std::size_t Rank>
    auto get_target_ranks(mara::arithmetic_binary_tree_t<std::size_t, Rank> rank_tree, mara::tree_index_t<Rank> target);

    inline auto get_comm_map(arithmetic_binary_tree_t<std::size_t, 2> rank_tree, tree_index_t<2> idx);
    // inline auto get_comm_map(mara::arithmetic_binary_tree_t<std::size_t, 3> rank_tree, mara::tree_index_t<3> idx);
}




//=============================================================================
namespace mara::parallel::detail
{
    inline std::pair<int, int> factor_once(int num);
    inline void prime_factors_impl(std::vector<int>& result, int num);
    inline nd::shared_array<int, 1> prime_factors(int num);
}




/**
 * @brief      Return a shared-memory nd::array evaluator for the given number
 *             of cores.
 *
 * @tparam     NumThreads  The number of cores to use
 *
 * @return     The evaluator
 *
 * @note       mara::evaluate_on<CoreCount>() is a drop-in replacement for
 *             nd::to_shared().
 */
template<std::size_t NumThreads>
auto mara::evaluate_on()
{
    return [] (auto array)
    {
        using value_type = typename decltype(array)::value_type;
        auto provider = nd::make_unique_provider<value_type>(array.shape());
        auto evaluate_partial = [&] (auto accessor)
        {
            return [accessor, array, &provider]
            {
                for (auto index : accessor)
                {
                    provider(index) = array(index);
                }
            };
        };
        auto threads = nd::basic_sequence_t<std::thread, NumThreads>();
        auto regions = nd::partition_shape<NumThreads>(array.shape());

        for (std::size_t n = 0; n < NumThreads; ++n)
            threads[n] = std::thread(evaluate_partial(regions[n]));

        for (auto& thread : threads)
            thread.join();

        return nd::make_array(std::move(provider).shared());
    };
}




/**
 * @brief      Return an nd::shape_t whose volume is equal to the given number
 *             of subdomains, and whose sizes on each axis are as similar as
 *             possible.
 *
 * @param[in]  number_of_subdomains  The number of subdomains
 *
 * @tparam     Rank                  The rank of the decomposed domain
 *
 * @return     The shape of the decomposed blocks
 */
template<std::size_t Rank>
auto mara::propose_block_decomposition(std::size_t number_of_subdomains)
{
    auto product = [] (auto g) { return std::accumulate(g.begin(), g.end(), 1, std::multiplies<>()); };
    auto result = nd::shape_t<Rank>();
    std::size_t n = 0;

    for (auto dim : parallel::detail::prime_factors(number_of_subdomains) | nd::divvy(Rank) | nd::map(product))
    {
        result[n++] = dim;
    }
    return result;
}




/**
 * @brief      Creates an N-dimensional array of N-dimensional
 *             nd::access_pattern_t instances, representing a block
 *             decomposition of an N-dimensional array.
 *
 * @param[in]  global_shape  The shape of the array to decompose
 * @param[in]  blocks_shape  The shape of the decomposed subgrid blocks
 *
 * @tparam     Rank          The dimensionality N
 *
 * @return     The array
 */
template<std::size_t Rank>
auto mara::create_access_pattern_array(nd::shape_t<Rank> global_shape, nd::shape_t<Rank> blocks_shape)
{
    nd::basic_sequence_t<std::vector<std::size_t>, Rank> block_start_indexes;
    nd::basic_sequence_t<std::vector<std::size_t>, Rank> block_sizes;

    for (std::size_t axis = 0; axis < Rank; ++axis)
    {
        for (auto index_group : nd::arange(global_shape[axis]) | nd::divvy(blocks_shape[axis]))
        {
            if (index_group.size() == 0)
            {
                throw std::logic_error("too many blocks for global domain size");
            }
            block_sizes        [axis].push_back( index_group.size());
            block_start_indexes[axis].push_back(*index_group.begin());
        }
    }

    auto mapping = [=] (auto index)
    {
        auto block = nd::access_pattern_t<Rank>();

        for (std::size_t axis = 0; axis < Rank; ++axis)
        {
            block.start[axis] = block_start_indexes[axis][index[axis]];
            block.final[axis] = block.start[axis] + block_sizes[axis][index[axis]];
        }
        return block;
    };
    return nd::make_array(mapping, blocks_shape) | nd::bounds_check();
}




//=============================================================================
std::pair<int, int> mara::parallel::detail::factor_once(int num)
{
    for (int d = 2; ; ++d)
    {
        if (num % d == 0)
        {
            return {d, num / d};
        }
        if (d * d > num)
        {
            break;
        }
    }
    return {num, 1};
}

void mara::parallel::detail::prime_factors_impl(std::vector<int>& result, int num)
{
    auto once = factor_once(num);

    if (once.second == 1)
    {
        result.push_back(once.first);
    }
    else
    {
        prime_factors_impl(result, once.first);
        prime_factors_impl(result, once.second);
    }
}

nd::shared_array<int, 1> mara::parallel::detail::prime_factors(int num)
{
    std::vector<int> result;
    prime_factors_impl(result, num);
    return nd::make_array_from(result);
}



//=============================================================================
template<typename ValueType, std::size_t Rank>
auto mara::build_rank_tree(const mara::arithmetic_binary_tree_t<ValueType, Rank> topology, std::size_t size)
{
    auto topo_indexes = topology.indexes();


    // 1. Get Hilbert Indeces
    auto hindexes = topo_indexes.map([] (auto i) { return mara::hilbert_index(i); });


    // 2. Create list of hindex-index pairs
    auto I = mara::linked_list_t<mara::tree_index_t<Rank>>{topo_indexes.begin(), topo_indexes.end()};
    auto H = mara::linked_list_t<std::size_t>{hindexes.begin(), hindexes.end()};
    auto hi_pair = H.pair(I);


    // 3. Sort the pair-list by hilbert index
    auto hi_sorted = hi_pair.sort([] (auto a, auto b) { return a.first < b.first; });
    

    // 4. Convert to nd::array and divvy into equal sized segments
    auto rank_sequences = nd::make_array_from(hi_sorted) | nd::divvy(size);


    // 5. Build the tree of ranks and return
    auto get_rank = [rank_sequences, size] (mara::tree_index_t<Rank> idx)
    {
        //TODO: Better search
        for(std::size_t i = 0; i < size; i++)
        {
            for(std::size_t j = 0; j < rank_sequences(i).size(); j++)
            {
                if (rank_sequences(i)(j).second == idx)
                {
                    return i;
                }
            }
        }
        throw std::logic_error("an index was not found");
    };
    return topo_indexes.map(get_rank);
}




template<std::size_t Rank>
auto mara::get_target_ranks(mara::arithmetic_binary_tree_t<std::size_t, Rank> rank_tree, mara::tree_index_t<Rank> target)
{
    auto result = mara::linked_list_t<std::size_t>();

    if (rank_tree.contains_node(target))  //if the target index is either a node or a leaf in the tree
    {  
       
        if (rank_tree.contains(target)) //if leaf
        {
            return result.prepend(rank_tree.at(target));
        }
        else  //if node
        {
            for(auto i : rank_tree.node_at(target))
            {
                result = result.prepend(i);
            }
            return result.reverse();
        }
    }
    //else: target index is not in the current tree --> need refinement
    //iterate until find first parent that is in the tree
    
    auto parent = target.parent_index();

    while (! rank_tree.contains(parent))
    {
        parent = parent.parent_index();
    }
    return result.prepend(rank_tree.at(parent));
}




inline auto mara::get_comm_map(arithmetic_binary_tree_t<std::size_t, 2> rank_tree, tree_index_t<2> idx) 
{
    std::map<std::string, mara::linked_list_t<std::size_t>> comm_map;
    comm_map.insert(std::make_pair("north", mara::get_target_ranks<2>(rank_tree, idx.next_on(1))));
    comm_map.insert(std::make_pair("south", mara::get_target_ranks<2>(rank_tree, idx.prev_on(1))));
    comm_map.insert(std::make_pair("east" , mara::get_target_ranks<2>(rank_tree, idx.next_on(0))));
    comm_map.insert(std::make_pair("west" , mara::get_target_ranks<2>(rank_tree, idx.prev_on(0))));

    return comm_map;
}




// inline auto mara::get_comm_map(mara::arithmetic_binary_tree_t<std::size_t, 3> rank_tree, mara::tree_index_t<3> idx) 
// {
//     std::map<std::string, mara::linked_list_t<std::size_t>> comm_map;
//     comm_map.insert(std::make_pair("north", mara::get_target_ranks<3>(rank_tree, idx.next_on(0))));
//     comm_map.insert(std::make_pair("south", mara::get_target_ranks<3>(rank_tree, idx.prev_on(0))));
//     comm_map.insert(std::make_pair("east" , mara::get_target_ranks<3>(rank_tree, idx.next_on(1))));
//     comm_map.insert(std::make_pair("west" , mara::get_target_ranks<3>(rank_tree, idx.prev_on(1))));
//     comm_map.insert(std::make_pair("up"   , mara::get_target_ranks<3>(rank_tree, idx.next_on(2))));
//     comm_map.insert(std::make_pair("down" , mara::get_target_ranks<3>(rank_tree, idx.prev_on(2))));

//     return comm_map;
// }
