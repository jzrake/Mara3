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
#include <initializer_list>  // std::initializer_list




//=============================================================================
namespace mara
{
    template<typename ValeuType>
    struct linked_list_t;
}




/**
 * @brief      An immutable, singly-linked list implementation. Instances of
 *             this data structure allow for O(1) prepend operations and O(N)
 *             append operations.
 *
 * @tparam     ValueType  The value type stored in the list
 */
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




    /**
     * @brief      Destroys the linked list.
     *
     * @note       Care is taken not to overflow the stack by recursive shared
     *             pointer free's. This destructor is not thread-safe! See Sec.
     *             8.1.4 of Functional Programming in C++ by Ivan Čukić.
     */
    ~linked_list_t()
    {
        auto next_node = std::move(__next);    

        while (next_node)
        {
            if (! next_node.unique())
                break;

            std::shared_ptr<linked_list_t> rest;
            std::swap(rest, next_node->__next);

            next_node.reset();
            next_node = std::move(rest);
        }
    }




    linked_list_t() {}




    /**
     * @brief      Construct a list from a value and a tail.
     *
     * @param[in]  value  The value to be at the front of the list
     * @param[in]  next   The rest of the list
     */
    linked_list_t(const value_type& value, const linked_list_t& next)
    : __value(value)
    , __next(std::make_shared<linked_list_t>(next))
    {
        if (__next == nullptr)
        {
            throw std::bad_alloc();
        }
    }




    /**
     * @brief      Construct a list from a sequence of values.
     *
     * @param[in]  values  The values
     */
    linked_list_t(std::initializer_list<value_type> values)
    {
        linked_list_t result;

        for (auto v : values)
        {
            result = result.prepend(v);
        }
        *this = result.reverse();
    }




    /**
     * @brief      Construct a linked list from a pair of iterators.
     *
     * @param[in]  first   The first iterator
     * @param[in]  second  The second iterator
     *
     * @tparam     First   The first iterator type
     * @tparam     Second  The second iterator type
     */
    template<typename First, typename Second>
    linked_list_t(First first, Second second)
    {
        linked_list_t result;

        while (first != second)
        {
            result = result.prepend(*first);
            ++first;
        }
        *this = result.reverse();
    }




    /**
     * @brief      Determine whether this list is empty; O(1).
     *
     * @return     True or false
     */
    bool empty() const
    {
        return __next == nullptr;
    }




    /**
     * @brief      Return the size N of this list; O(N).
     *
     * @return     The size of the list
     */
    std::size_t size() const
    {
        auto result = std::size_t(0);
        auto it = begin();

        while (it != end())
        {
            ++result;
            ++it;
        }
        return result;
    }




    /**
     * @brief      Return another list with the given value prepended; O(1).
     *
     * @param[in]  value  The value to prepend
     *
     * @return     Another list
     */
    linked_list_t prepend(const value_type& value) const
    {
        return linked_list_t(value, *this);
    }




    /**
     * @brief      Return another list with the given value append; O(1).
     *
     * @param[in]  value  The value to append
     *
     * @return     Another list
     */
    linked_list_t append(const value_type& value) const
    {
        return empty() ? linked_list_t(value, *this) : tail().append(value).prepend(head());
    }




    /**
     * @brief      Concatenate two lists; O(N).
     *
     * @param[in]  other  The other list
     *
     * @return     The elements of this list, followed by the elements the
     *             another
     */
    linked_list_t concat(const linked_list_t& other) const
    {
        return empty() ? other : linked_list_t(head(), tail().concat(other));
    }




    /**
     * @brief      Reverse this list; O(N).
     *
     * @return     A new list
     * 
     */
    linked_list_t reverse() const
    {
        auto A = *this;
        auto B = linked_list_t();

        while (! A.empty())
        {
            B = B.prepend(A.head());
            A = A.tail();
        }
        return B;
    }




    /**
     * @brief      Return the value at the front of the list; O(1). Throws
     *             out_of_range if this list is empty.
     *
     * @return     The value at the front
     */
    const value_type& head() const
    {
        if (empty())
        {
            throw std::out_of_range("mara::linked_list_t cannot get the head of an empty list");
        }
        return __value;
    }




    /**
     * @brief      Return the rest of the list, if this list is non-empty; O(1).
     *
     * @return     The rest of the list
     */
    const linked_list_t& tail() const
    {
        if (empty())
        {
            throw std::out_of_range("mara::linked_list_t cannot get the tail of an empty list");
        }
        return *__next;
    }




    /**
     * @brief      Return an iterator to the beginning of the list.
     *
     * @return     An iterator
     */
    iterator begin() const
    {
        return {*this};
    }




    /**
     * @brief      Return an iterator to the end of the list.
     *
     * @return     An iterator
     */
    iterator end() const
    {
        return {linked_list_t()};
    }




    //=========================================================================
    //                  Sorting Routines
    //=========================================================================

    /**
     * @brief     Return the front half of a list
     * 
     * @return    A list
     */
    const linked_list_t front_half() const
    {
        auto iter = begin();

        linked_list_t<value_type> temp;
        for(std::size_t n = 0; n < size() / 2; n++)
        {
            temp = temp.prepend(*iter);
            ++iter;
        }
        return temp.reverse();
    }




    /**
     * @brief     Returns the back half of a list
     * 
     * @return    A list
     */
    const linked_list_t back_half() const
    {
        auto iter = begin();

        for(std::size_t n = 0; n < size() / 2; n++)
        {
            ++iter;
        }
        return iter.current;
    }




    /**
     * @brief     Sorts and merges two already sorted lists. Both the calling list
     *            and the parameter list must already be sorted.
     *             
     * @param     other  The other sorted list
     *    
     * @return    A sorted list
     */
    const linked_list_t sorted_merge(const linked_list_t other, bool (*cmp)(value_type, value_type)) const
    {
        auto result = linked_list_t<value_type>();

        if (empty())
        {
            return other;
        }
        else if (other.empty())
        {
            return *this;
        }
        else if (cmp(head(), other.head()))
        {
            auto temp = result.prepend(head()); 
            result = temp.concat(tail().sorted_merge(other, cmp));
        }
        else
        {
            auto temp = result.prepend(other.head());
            result = temp.concat(other.tail().sorted_merge(*this, cmp));
        }
        return result;
    }




    /**
     * @brief      Basic function to sort values in ascending order
     * 
     * @return     A sorted list
     */
    const linked_list_t sort() const
    {
        if (size() <= 1)
        {
            return *this;
        }
        
        //Split in half and sort
        auto A = front_half().sort();
        auto B =  back_half().sort();

        //Stitch together sorted lists
        return A.sorted_merge(B, [] (auto x, auto y) { return x < y; });
    }




    /**
     * @brief     Function to sort based on the provided boolean function
     *
     * @param     cmp   A boolean function that compares two parameters of 
     *                  the same type
     *
     * @return    The sorted list
     */
    const linked_list_t sort(bool (*cmp)(value_type, value_type)) const
    {
        if(size() <= 1)
        {
            return *this;
        }

        auto A = front_half().sort(cmp);
        auto B =  back_half().sort(cmp);

        return A.sorted_merge(B, cmp);
    }




    //=========================================================================
    /**
     * @brief      Determine whether two sequences are equal.
     *
     * @param[in]  other  The other list
     *
     * @return     True or false
     */
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




    /**
     * @brief      Determine whether two sequences are unequal.
     *
     * @param[in]  other  The other list
     *
     * @return     True or false
     */
    bool operator!=(const linked_list_t& other) const
    {
        return ! operator==(other);
    }




    //=========================================================================
    ValueType __value;
    std::shared_ptr<linked_list_t> __next;
};
