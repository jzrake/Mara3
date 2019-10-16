
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
#include "core_ndarray.hpp"
#include "core_tree.hpp"




//=============================================================================
namespace mara {

template<typename ValueType>
struct string_serializer_t {};




template<typename ValueType>
std::string dumps(ValueType v)
{
    return string_serializer_t<ValueType>::dump_to(v);
}

template<typename ValueType>
ValueType loads(std::string b)
{
    return string_serializer_t<ValueType>::load_from(b);
}

template<typename ValueType>
std::size_t serialized_size(ValueType v)
{
    return string_serializer_t<ValueType>::serialized_size_of(v);
}

template<typename ValueType>
std::size_t deserialized_size(std::string buffer)
{
    return string_serializer_t<ValueType>::deserialized_size_of(buffer);
}




//=============================================================================
template<typename T, typename U>
struct string_serializer_t<std::pair<T, U>>
{
    static std::string dump_to(std::pair<T, U> value)
    {
        return dumps(value.first) + dumps(value.second);
    }
    static std::pair<T, U> load_from(std::string buffer)
    {
        auto first_size = deserialized_size<T>(buffer);
        auto buffer_first  = buffer.substr(0, first_size);
        auto buffer_second = buffer.substr(first_size);
        return std::make_pair(loads<T>(buffer_first), loads<U>(buffer_second));
    }
    static std::size_t serialized_size_of(std::pair<T, U> value)
    {
        return serialized_size(value.first) + serialized_size(value.second);
    }
    static std::size_t deserialized_size_of(std::string buffer)
    {
        auto first_size = deserialized_size<T>(buffer);
        return first_size + deserialized_size<U>(buffer.substr(first_size));
    }
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
struct string_serializer_t<nd::shared_array<ValueType, Rank>>
{
    static std::string dump_to(nd::shared_array<ValueType, Rank> array)
    {
        auto buffer = std::string(array.size() * sizeof(ValueType), 0);
        std::memcpy(buffer.data(), array.data(), buffer.size());
        return dumps(array.shape()) + buffer;
    }

    static nd::shared_array<ValueType, Rank> load_from(std::string buffer)
    {
        auto shape_buffer = buffer.substr(0, serialized_size(nd::shape_t<Rank>()));
        auto array_buffer = buffer.substr(serialized_size(nd::shape_t<Rank>()));
        auto shape = loads<nd::shape_t<Rank>>(shape_buffer);

        if (array_buffer.size() != shape.volume() * sizeof(ValueType))
            throw std::invalid_argument("loads (buffer has wrong size for expected array shape)");

        auto array = nd::make_unique_array<ValueType>(shape);
        std::memcpy(array.data(), array_buffer.data(), array_buffer.size());
        return std::move(array).shared();
    }

    static std::size_t serialized_size_of(nd::shared_array<ValueType, Rank> array)
    {
        return serialized_size(nd::shape_t<Rank>()) + array.size() * sizeof(ValueType);
    }

    static std::size_t deserialized_size_of(std::string buffer)
    {
        auto shape = loads<nd::shape_t<Rank>>(buffer.substr(0, serialized_size(nd::shape_t<Rank>())));
        return serialized_size(nd::shape_t<Rank>()) + shape.volume() * sizeof(ValueType);
    }
};




//=============================================================================
template<std::size_t Rank>
struct string_serializer_t<mara::tree_index_t<Rank>>
{
    static std::string dump_to(mara::tree_index_t<Rank> index)
    {
        static_assert(std::is_trivially_copyable_v<mara::tree_index_t<Rank>>, "value type isn't safe for memcpy");

        auto buffer = std::string(sizeof(index), 0);
        std::memcpy(buffer.data(), &index, buffer.size());
        return buffer;
    }
    static mara::tree_index_t<Rank> load_from(std::string buffer)
    {
        static_assert(std::is_trivially_copyable_v<mara::tree_index_t<Rank>>, "value type isn't safe for memcpy");

        if (buffer.size() != sizeof(mara::tree_index_t<Rank>))
            throw std::invalid_argument("loads (buffer has wrong size for type)");

        auto index = mara::tree_index_t<Rank>{};
        std::memcpy(&index, buffer.data(), buffer.size());
        return index;
    }
    static std::size_t serialized_size_of(mara::tree_index_t<Rank>)
    {
        return sizeof(mara::tree_index_t<Rank>);
    }
    static std::size_t deserialized_size_of(std::string)
    {
        return sizeof(mara::tree_index_t<Rank>);
    }
};




//=============================================================================
template<std::size_t Rank>
struct string_serializer_t<nd::shape_t<Rank>>
{
    static std::string dump_to(nd::shape_t<Rank> shape)
    {
        static_assert(std::is_trivially_copyable_v<nd::shape_t<Rank>>, "value type isn't safe for memcpy");

        auto buffer = std::string(sizeof(shape), 0);
        std::memcpy(buffer.data(), &shape, buffer.size());
        return buffer;
    }
    static nd::shape_t<Rank> load_from(std::string buffer)
    {
        static_assert(std::is_trivially_copyable_v<nd::shape_t<Rank>>, "value type isn't safe for memcpy");

        if (buffer.size() != sizeof(nd::shape_t<Rank>))
            throw std::invalid_argument("loads (buffer has wrong size for type)");

        auto shape = nd::shape_t<Rank>{};
        std::memcpy(&shape, buffer.data(), buffer.size());
        return shape;
    }
    static std::size_t serialized_size_of(nd::shape_t<Rank>)
    {
        return sizeof(nd::shape_t<Rank>);
    }
    static std::size_t deserialized_size_of(std::string)
    {
        return sizeof(nd::shape_t<Rank>);
    }
};

} // namespace mara




// TODO: add these tests to the unit tests

// int main()
// {

//     auto test_index = mara::tree_index_t<5>{2, {3, 4, 5, 6, 7}};
//     auto test_shape = nd::shape_t<3>{7, 5, 9};
//     auto test_array = nd::linspace(0, 1, 100) | nd::to_shared();
//     auto test_pair1 = std::make_pair(test_index, test_array);
//     auto test_pair2 = std::make_pair(test_array, test_index);

//     assert(test_index == loads<decltype(test_index)>(dumps(test_index)));
//     assert(test_shape == loads<decltype(test_shape)>(dumps(test_shape)));
//     assert((test_array == loads<decltype(test_array)>(dumps(test_array)) | nd::all()));

//     assert(std::get<0>(test_pair1) == std::get<0>(loads<decltype(test_pair1)>(dumps(test_pair1))));
//     assert(std::get<1>(test_pair2) == std::get<1>(loads<decltype(test_pair2)>(dumps(test_pair2))));
//     assert((std::get<1>(test_pair1) == std::get<1>(loads<decltype(test_pair1)>(dumps(test_pair1)))) | nd::all());
//     assert((std::get<0>(test_pair2) == std::get<0>(loads<decltype(test_pair2)>(dumps(test_pair2)))) | nd::all());

//     return 0;
// }
