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
#include "ndh5.hpp"
#include "ndarray.hpp"




//=============================================================================
namespace serialize
{

    //=========================================================================
    template<typename Writable, typename Serializable>
    auto write_struct_providing_keyval_tuples(Writable& location, Serializable&& instance);

    //=========================================================================    
    namespace detail
    {
        template<typename Function, typename Tuple, std::size_t... Is>
        void foreach_tuple_impl(Function&& fn, Tuple&& t, std::index_sequence<Is...>);

        template<typename Function, typename Tuple>
        void foreach_tuple(Function&& fn, Tuple&& t);

        template<typename Function, typename Tuple1, typename Tuple2, std::size_t... Is>
        void foreach_tuple_impl(Function&& fn, Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>);

        template<typename Function, typename Tuple1, typename Tuple2>
        void foreach_tuple(Function&& fn, Tuple1&& t1, Tuple2&& t2);

        template<typename Tuple1, typename Tuple2, std::size_t... Is>
        auto tuple_pair_impl(Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>);

        template<typename Tuple1, typename Tuple2>
        auto tuple_pair(Tuple1&& t1, Tuple2&& t2);
    }
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct h5::hdf5_type_info<nd::shared_array<ValueType, Rank>>
{
    static auto make_datatype_for(const nd::shared_array<ValueType, Rank>&) { return h5::make_datatype_for(ValueType()); }
    static auto make_dataspace_for(const nd::shared_array<ValueType, Rank>& value) { return Dataspace::simple(value.shape()); }
    static auto get_address(const nd::shared_array<ValueType, Rank>& value) { return value.data(); }
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct h5::hdf5_type_info<nd::unique_array<ValueType, Rank>>
{
    static auto make_datatype_for(const nd::unique_array<ValueType, Rank>&) { return h5::make_datatype_for(ValueType()); }
    static auto make_dataspace_for(const nd::unique_array<ValueType, Rank>& value) { return Dataspace::simple(value.shape()); }
    static auto prepare(const Datatype&, const Dataspace& space) { return nd::make_unique_array<ValueType>(nd::shape_t<Rank>::from_range(space.extent())); }
    static auto get_address(nd::unique_array<ValueType, Rank>& value) { return value.data(); }
};




//=============================================================================
template<typename Writable, typename Serializable>
auto serialize::write_struct_providing_keyval_tuples(Writable& location, Serializable&& instance)
{
    auto f = [&] (auto&& name, auto&& value)
    {
        location.write(name, value);
    };
    detail::foreach_tuple(f, instance.keys(), instance.values());
}




//=============================================================================
template<typename Function, typename Tuple, std::size_t... Is>
void serialize::detail::foreach_tuple_impl(Function&& fn, Tuple&& t, std::index_sequence<Is...>)
{
    (fn(std::get<Is>(t)), ...);
}

template<typename Function, typename Tuple>
void serialize::detail::foreach_tuple(Function&& fn, Tuple&& t)
{
    return foreach_tuple_impl(
        std::forward<Function>(fn),
        std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

template<typename Function, typename Tuple1, typename Tuple2, std::size_t... Is>
void serialize::detail::foreach_tuple_impl(Function&& fn, Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>)
{
    (fn(std::get<Is>(t1), std::get<Is>(t2)), ...);
}

template<typename Function, typename Tuple1, typename Tuple2>
void serialize::detail::foreach_tuple(Function&& fn, Tuple1&& t1, Tuple2&& t2)
{
    return foreach_tuple_impl(
        std::forward<Function>(fn),
        std::forward<Tuple1>(t1),
        std::forward<Tuple2>(t2),
        std::make_index_sequence<std::tuple_size<Tuple1>::value>());
}

template<typename Tuple1, typename Tuple2, std::size_t... Is>
auto serialize::detail::tuple_pair_impl(Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>)
{
    return std::make_tuple(std::make_pair(std::get<Is>(t1), std::get<Is>(t2))...);
}

template<typename Tuple1, typename Tuple2>
auto serialize::detail::tuple_pair(Tuple1&& t1, Tuple2&& t2)
{
    return tuple_pair_impl(
        std::forward<Tuple1>(t1),
        std::forward<Tuple2>(t2),
        std::make_index_sequence<std::tuple_size<Tuple1>::value>());
}
