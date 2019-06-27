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
#include <memory>            // std::shared_ptr
#include <initializer_list>  // initializer_list




//=============================================================================
namespace mara
{
    template<typename ValeuType>
    struct linked_list_t;
}




//=============================================================================
template<typename ValueType>
struct mara::linked_list_t
{


    using value_type = ValueType;


    //=========================================================================
    struct iterator
    {
        using iterator_category = std::input_iterator_tag;
        using value_type = linked_list_t::value_type;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;

        iterator& operator++()
        {
            current = current.tail();
            return *this;
        }
        bool operator==(const iterator& other) const { return current.__next == other.current.__next; }
        bool operator!=(const iterator& other) const { return current.__next != other.current.__next; }
        const value_type& operator*() const { return current.head(); }

        linked_list_t current;
    };


    //=========================================================================
    linked_list_t() {}
    linked_list_t(const value_type& value, const std::shared_ptr<linked_list_t>& next) : __value(value), __next(next) {}
    linked_list_t(std::initializer_list<value_type> values)
    {
        linked_list_t result;

        for (auto v : values)
        {
            result = result.prepend(v);
        }
        *this = result.reverse();
    }

    bool empty() const
    {
        return __next == nullptr;
    }

    std::size_t size() const
    {
        return empty() ? 0 : 1 + tail().size();
    }

    linked_list_t prepend(const value_type& value) const
    {
        return linked_list_t(value, std::make_shared<linked_list_t>(*this));
    }

    linked_list_t append(const value_type& value) const
    {
        return empty() ? linked_list_t().prepend(value) : tail().append(value).prepend(head());
    }

    const value_type& head() const
    {
        if (empty())
        {
            throw std::out_of_range("mara::linked_list_t cannot get the head of an empty list");
        }
        return __value;
    }

    const linked_list_t& tail() const
    {
        if (empty())
        {
            throw std::out_of_range("mara::linked_list_t cannot get the tail of an empty list");
        }
        return *__next;
    }

    linked_list_t reverse() const
    {
        return empty() ? *this : tail().reverse().append(head());
    }

    iterator begin() const
    {
        return {*this};
    }

    iterator end() const
    {
        return {linked_list_t()};
    }

    bool operator==(const linked_list_t& other) const
    {
        if (! empty() && ! other.empty())
        {
            return head() == other.head() && tail() == other.tail();
        }
        if (empty() && other.empty())
        {
            return true;
        }
        return false;
    }

    bool operator!=(const linked_list_t& other) const
    {
        return ! operator==(other);
    }


    //=========================================================================
    ValueType __value;
    std::shared_ptr<linked_list_t> __next;
};
