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
#include <variant>
#include "ndh5.hpp"
#include "ndarray.hpp"
#include "app_config.hpp"
#include "app_schedule.hpp"
#include "core_geometric.hpp"




//=============================================================================
namespace mara
{
    inline void write_schedule(h5::Group&& group, const mara::schedule_t& schedule);
    inline auto read_schedule(h5::Group&& group);
    inline void write_config(h5::Group&& group, mara::config_t run_config);
    inline auto read_config(h5::Group&& group);
    template<std::size_t Rank> auto make_hdf5_hyperslab(const nd::access_pattern_t<Rank>& sel);

    inline std::string create_numbered_filename(std::string prefix, int count, std::string extension);

    template<typename Writable, typename Serializable>
    auto write_struct_providing_keyval_tuples(Writable& location, Serializable&& instance);
}




//=============================================================================
namespace mara::serialize::detail
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
};




//=============================================================================
void mara::write_schedule(h5::Group&& group, const mara::schedule_t& schedule)
{
    for (auto task : schedule)
    {
        auto h5_task = group.require_group(task.first);
        h5_task.write("name", task.second.name);
        h5_task.write("num_times_performed", task.second.num_times_performed + 1);
        h5_task.write("last_performed", task.second.last_performed);
    }
}

auto mara::read_schedule(h5::Group&& group)
{
    auto schedule = mara::schedule_t();

    for (auto task_name : group)
    {
        auto task = mara::schedule_t::task_t();
        auto h5_task = group.open_group(task_name);
        task.name = task_name;
        task.num_times_performed = h5_task.read<int>("num_times_performed");
        task.last_performed = h5_task.read<double>("last_performed");
        schedule.insert(task);
    }
    return schedule;
}




//=============================================================================
void mara::write_config(h5::Group&& group, mara::config_t run_config)
{
    for (auto item : run_config)
    {
        group.write(item.first, item.second);
    }
}

auto mara::read_config(h5::Group&& group)
{
    auto config = mara::config_parameter_map_t();

    for (auto item_name : group)
    {
        config[item_name] = group.read<mara::config_parameter_t>(item_name);
    }
    return config;
}

template<std::size_t Rank>
auto mara::make_hdf5_hyperslab(const nd::access_pattern_t<Rank>& sel)
{
    auto sel_shape = sel.shape();
    auto result = h5::hyperslab_t();
    result.start = std::vector<hsize_t>(sel.start.begin(), sel.start.end());
    result.skips = std::vector<hsize_t>(sel.jumps.begin(), sel.jumps.end());
    result.count = std::vector<hsize_t>(sel_shape.begin(), sel_shape.end());
    result.block = std::vector<hsize_t>(sel.rank, 1);
    return result;
}




//=============================================================================
std::string mara::create_numbered_filename(std::string prefix, int count, std::string extension)
{
    char filename[1024];
    std::snprintf(filename, 1024, "%s.%04d.%s", prefix.data(), count, extension.data());
    return filename;
}




//=============================================================================
template<typename Writable, typename Serializable>
auto mara::write_struct_providing_keyval_tuples(Writable& location, Serializable&& instance)
{
    auto f = [&] (auto&& name, auto&& value)
    {
        location.write(name, value);
    };
    mara::serialize::detail::foreach_tuple(f, instance.keys(), instance.values());
}




//=============================================================================
template<typename Function, typename Tuple, std::size_t... Is>
void mara::serialize::detail::foreach_tuple_impl(Function&& fn, Tuple&& t, std::index_sequence<Is...>)
{
    (fn(std::get<Is>(t)), ...);
}

template<typename Function, typename Tuple>
void mara::serialize::detail::foreach_tuple(Function&& fn, Tuple&& t)
{
    return foreach_tuple_impl(
        std::forward<Function>(fn),
        std::forward<Tuple>(t),
        std::make_index_sequence<std::tuple_size<Tuple>::value>());
}

template<typename Function, typename Tuple1, typename Tuple2, std::size_t... Is>
void mara::serialize::detail::foreach_tuple_impl(Function&& fn, Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>)
{
    (fn(std::get<Is>(t1), std::get<Is>(t2)), ...);
}

template<typename Function, typename Tuple1, typename Tuple2>
void mara::serialize::detail::foreach_tuple(Function&& fn, Tuple1&& t1, Tuple2&& t2)
{
    return foreach_tuple_impl(
        std::forward<Function>(fn),
        std::forward<Tuple1>(t1),
        std::forward<Tuple2>(t2),
        std::make_index_sequence<std::tuple_size<Tuple1>::value>());
}

template<typename Tuple1, typename Tuple2, std::size_t... Is>
auto mara::serialize::detail::tuple_pair_impl(Tuple1&& t1, Tuple2&& t2, std::index_sequence<Is...>)
{
    return std::make_tuple(std::make_pair(std::get<Is>(t1), std::get<Is>(t2))...);
}

template<typename Tuple1, typename Tuple2>
auto mara::serialize::detail::tuple_pair(Tuple1&& t1, Tuple2&& t2)
{
    return tuple_pair_impl(
        std::forward<Tuple1>(t1),
        std::forward<Tuple2>(t2),
        std::make_index_sequence<std::tuple_size<Tuple1>::value>());
}




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
template<typename ValueType, std::size_t Rank>
struct h5::hdf5_type_info<mara::covariant_sequence_t<ValueType, Rank>>
{
    using native_type = mara::covariant_sequence_t<ValueType, Rank>;
    static auto make_datatype_for(const native_type& value) { return h5::make_datatype_for(ValueType()).as_array(Rank); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto get_address(const native_type& value) { return &value[0]; }
    static auto get_address(native_type& value) { return &value[0]; }
};




//=============================================================================
template<int C, int G, int S, typename T>
struct h5::hdf5_type_info<mara::dimensional_value_t<C, G, S, T>>
{
    using native_type = mara::dimensional_value_t<C, G, S, T>;
    static auto make_datatype_for(const native_type& value) { return h5::make_datatype_for(value.value); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto get_address(const native_type& value) { return &value.value; }
    static auto get_address(native_type& value) { return &value.value; }
};




//=============================================================================
template<>
struct h5::hdf5_type_info<mara::config_parameter_t>
{
    static auto make_dataspace_for(const mara::config_parameter_t& value)
    {
        switch (value.index())
        {
            case 0: return h5::make_dataspace_for(std::get<0>(value));
            case 1: return h5::make_dataspace_for(std::get<1>(value));
            case 2: return h5::make_dataspace_for(std::get<2>(value));
        }
        throw;
    }
    static auto prepare(const Datatype& dtype, const Dataspace& space)
    {
        if (dtype == Datatype::native_int())    return mara::config_parameter_t(h5::prepare<int>        (dtype, space));
        if (dtype == Datatype::native_double()) return mara::config_parameter_t(h5::prepare<double>     (dtype, space));
        if (dtype == Datatype::c_s1())          return mara::config_parameter_t(h5::prepare<std::string>(dtype, space));
        throw std::invalid_argument("invalid hdf5 type for parameter variant");
    }
    static auto make_datatype_for(const mara::config_parameter_t& value)
    {
        switch (value.index())
        {
            case 0: return h5::make_datatype_for(std::get<0>(value));
            case 1: return h5::make_datatype_for(std::get<1>(value));
            case 2: return h5::make_datatype_for(std::get<2>(value));
        }
        throw;
    }
    static auto get_address(mara::config_parameter_t& value)
    {
        switch (value.index())
        {
            case 0: return static_cast<void*>(h5::get_address(std::get<0>(value)));
            case 1: return static_cast<void*>(h5::get_address(std::get<1>(value)));
            case 2: return static_cast<void*>(h5::get_address(std::get<2>(value)));
        }
        throw;
    }
    static auto get_address(const mara::config_parameter_t& value)
    {
        switch (value.index())
        {
            case 0: return static_cast<const void*>(h5::get_address(std::get<0>(value)));
            case 1: return static_cast<const void*>(h5::get_address(std::get<1>(value)));
            case 2: return static_cast<const void*>(h5::get_address(std::get<2>(value)));
        }
        throw;
    }
};
