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
#include "core_hdf5.hpp"
#include "core_ndarray.hpp"
#include "core_geometric.hpp"
#include "core_rational.hpp"
#include "app_config.hpp"
#include "app_schedule.hpp"




//=============================================================================
namespace mara
{


    //=========================================================================
    inline void write_schedule(h5::Group&& group, const schedule_t& schedule);
    inline auto read_schedule(h5::Group&& group);
    inline void write_config(h5::Group&& group, const config_t& run_config);
    inline auto read_config(h5::Group&& group);


    //=========================================================================
    template<typename ValueType> void write(h5::Group& group, std::string name, const ValueType& value);
    template<typename ValueType> void read(h5::Group& group, std::string name, ValueType& value);
    template<typename ValueType> auto read(h5::Group&& group, std::string name);


    //=========================================================================
    inline std::string create_numbered_filename(std::string prefix, int count, std::string extension);
    template<std::size_t Rank> auto make_hdf5_hyperslab(const nd::access_pattern_t<Rank>& sel);
}




//=============================================================================
void mara::write_schedule(h5::Group&& group, const schedule_t& schedule)
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
void mara::write_config(h5::Group&& group, const config_t& run_config)
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
        try {
            config[item_name] = group.read<mara::config_parameter_t>(item_name);
        }
        catch (const std::exception& e)
        {
            std::printf("%s: %s\n", e.what(), item_name.data());
        }
    }
    return config;
}




//=============================================================================
template<typename ValueType>
void mara::write(h5::Group& group, std::string name, const ValueType& value)
{
    group.write(name, value);
}

template<typename ValueType>
void mara::read(h5::Group& group, std::string name, ValueType& value)
{
    group.read(name, value);
}

template<typename ValueType>
auto mara::read(h5::Group&& group, std::string name)
{
    ValueType result;
    read(group, name, result);
    return result;
}

template<>
inline void mara::read<mara::schedule_t>(h5::Group& group, std::string name, mara::schedule_t& value)
{
    value = read_schedule(group.open_group(name));
}

template<>
inline void mara::write<mara::schedule_t>(h5::Group& group, std::string name, const schedule_t& schedule)
{
    write_schedule(group.require_group(name), schedule);
}

template<>
inline void mara::write<mara::config_t>(h5::Group& group, std::string name, const config_t& run_config)
{
    write_config(group.require_group(name), run_config);
}




//=============================================================================
template<std::size_t Rank>
auto mara::make_hdf5_hyperslab(const nd::access_pattern_t<Rank>& sel)
{
    auto sel_shape = sel.shape();
    auto result = h5::hyperslab_t();
    result.start = std::vector<hsize_t>(sel.start.begin(), sel.start.end());
    result.skips = std::vector<hsize_t>(sel.jumps.begin(), sel.jumps.end());
    result.count = std::vector<hsize_t>(sel_shape.begin(), sel_shape.end());
    result.block = std::vector<hsize_t>(sel.rank(), 1);
    return result;
}

std::string mara::create_numbered_filename(std::string prefix, int count, std::string extension)
{
    char filename[1024];
    std::snprintf(filename, 1024, "%s.%04d.%s", prefix.data(), count, extension.data());
    return filename;
}




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct h5::hdf5_type_info<nd::shared_array<ValueType, Rank>>
{
    using native_type = nd::shared_array<ValueType, Rank>;
    static auto make_datatype_for(const native_type&) { return h5::make_datatype_for(ValueType()); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::simple(value.shape()); }
    static auto prepare(const Datatype&, const Dataspace& space) { return nd::make_unique_array<ValueType>(nd::shape_t<Rank>::from_range(space.extent())); }
    static auto finalize(nd::unique_array<ValueType, Rank>&& value) { return std::move(value).shared(); }
    static auto get_address(const native_type& value) { return value.data(); }
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct h5::hdf5_type_info<nd::unique_array<ValueType, Rank>>
{
    using native_type = nd::unique_array<ValueType, Rank>;
    static auto make_datatype_for(const native_type&) { return h5::make_datatype_for(ValueType()); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::simple(value.shape()); }
    static auto prepare(const Datatype&, const Dataspace& space) { return nd::make_unique_array<ValueType>(nd::shape_t<Rank>::from_range(space.extent())); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(native_type& value) { return value.data(); }
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct h5::hdf5_type_info<mara::arithmetic_sequence_t<ValueType, Rank>>
{
    using native_type = mara::arithmetic_sequence_t<ValueType, Rank>;
    static auto make_datatype_for(const native_type& value) { return h5::make_datatype_for(ValueType()).as_array(Rank); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
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
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(const native_type& value) { return &value.value; }
    static auto get_address(native_type& value) { return &value.value; }
};




//=============================================================================
template<>
struct h5::hdf5_type_info<mara::rational_number_t>
{
    using native_type = mara::rational_number_t;
    static auto make_datatype_for(const native_type& value) { return h5::Datatype::native_int().as_array(2); }
    static auto make_dataspace_for(const native_type& value) { return Dataspace::scalar(); }
    static auto prepare(const Datatype&, const Dataspace& space) { return native_type(); }
    static auto finalize(native_type&& value) { return std::move(value); }
    static auto get_address(const native_type& value) { return &value; }
    static auto get_address(native_type& value) { return &value; }
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
    static auto prepare(const Datatype& dtype, const Dataspace& space) -> mara::config_parameter_t
    {
        if (dtype              == Datatype::native_int())    return h5::prepare<int>        (dtype, space);
        if (dtype              == Datatype::native_double()) return h5::prepare<double>     (dtype, space);
        if (dtype.with_size(1) == Datatype::c_s1())          return h5::prepare<std::string>(dtype, space);
        throw std::invalid_argument("invalid HDF5 type for config_parameter_t (std::variant<int, double, std::string>)");
    }
    static auto finalize(mara::config_parameter_t&& value)
    {
        return std::move(value);
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
    static auto get_address(mara::config_parameter_t& value) -> void*
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
