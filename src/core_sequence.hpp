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
#include <functional> // std::plus, std::multiplies, etc...
#include <string>     // std::to_string
#include <array>      // std::array




//=============================================================================
namespace mara
{
    template<typename ValueType, std::size_t Rank> struct arithmetic_sequence_t;
    template<typename ValueType, std::size_t Rank, typename DerivedType> struct derivable_sequence_t;

    template<typename ValueType, std::size_t Rank>
    auto to_string(const arithmetic_sequence_t<ValueType, Rank>& sequence);

    template<typename ValueType, std::size_t Rank, typename DerivedType>
    auto to_string(const derivable_sequence_t<ValueType, Rank, DerivedType>& sequence);

    template<typename Function>
    auto lift(Function f);

    template<typename... Args, typename ValueType=std::common_type_t<Args...>>
    auto make_std_array(Args... args);

    template<typename... Args, typename ValueType=std::common_type_t<Args...>>
    auto make_sequence(Args&&... args);

    template<std::size_t Rank>
    arithmetic_sequence_t<std::size_t, Rank> iota();

    //=========================================================================
    namespace detail
    {
        template<std::size_t... Is>
        auto iota_impl(std::index_sequence<Is...>)
        {
            return mara::make_sequence(Is...);
        }
    }
}




/**
 * @brief      Class for working with sequences of arithmetic types.
 *
 * @tparam     Rank       The dimensionality
 * @tparam     ValueType  The underlying value type
 *
 * @note       Arithmetic operations on the underlying value type are preserved
 *             by the container: e.g. since multiplies(double, int) -> double,
 *             we also have multiplies(seq<double, Rank>, seq<int, Rank>) ->
 *             seq<double, Rank>). This class is marked final; unlike
 *             derivable_sequence_t (which must be inherited), the type
 *             identity of arithmetic_sequence_t is defined by the rank and the
 *             identity of the value type.
 */
template<typename ValueType, std::size_t Rank>
struct mara::arithmetic_sequence_t
{



    //=========================================================================
    using value_type = ValueType;



    //=========================================================================
    static constexpr std::size_t size()              { return Rank; }
    const ValueType* data()                    const { return __data; }
    const ValueType* begin()                   const { return __data; }
    const ValueType* end()                     const { return __data + Rank; }
    const ValueType& operator[](std::size_t n) const { return __data[n]; }
    const ValueType& at(std::size_t n)         const { if (n < Rank) return __data[n]; throw std::out_of_range("mara::arithmetic_sequence_t"); }
    ValueType* data()                                { return __data; }
    ValueType* begin()                               { return __data; }
    ValueType* end()                                 { return __data + Rank; }
    ValueType& operator[](std::size_t n)             { return __data[n]; }
    ValueType& at(std::size_t n)                     { if (n < Rank) return __data[n]; throw std::out_of_range("mara::arithmetic_sequence_t"); }




    /**
     * @brief      Type-safe indexing (preferred over oeprator[] where possible
     *             for safety).
     *
     * @tparam     Index  The index to get
     *
     * @return     The value at the given index
     */
    template<std::size_t Index> const ValueType& get() const
    {
        static_assert(Index < Rank, "mara::arithmetic_sequence_t out of range");
        return __data[Index];
    }




    /**
     * @brief      Return this sequence with the value at a single index mapped
     *             through a function.
     *
     * @param[in]  index     The index of the value to update
     * @param      fn        The function to map that value through
     *
     * @tparam     Function  The type of the function object
     *
     * @return     The new sequence (same rank and value type)
     */
    template<typename Function>
    auto update(std::size_t index, Function&& fn) const
    {
        return iota<Rank>().map([this, index, fn] (std::size_t n)
        {
            return index == n ? fn(this->operator[](n)) : this->operator[](n);
        });
    }




    /**
     * @brief      Map a function over the elements of this sequence, returning
     *             another whose value type is the function's return value.
     *
     * @param      fn        The function to map
     *
     * @tparam     Function  The type of the function object
     *
     * @return     A new sequence
     *
     * @note       This method makes a sequence into a "functor" (a formal
     *             functor, not the common misnomer in C++ really meaning
     *             function object).
     */
    template<typename Function>
    auto map(Function&& fn) const
    {
        return unary_op(std::forward<Function>(fn));
    }




    /**
     * @brief      Return [f(a) for f, a in zip(this, other)], if this is a
     *             sequence of functions.
     *
     * @param[in]  other  A sequence of arguments given to this sequence of
     *                    (unary) functions
     *
     * @tparam     T      The value type of the argument sequence
     *
     * @return     A new sequence
     *
     * @note       This method is conventionally referred to as "ap" in
     *             functional programming. With respect to this method, a
     *             sequence is an "applicative functor".
     */
    template<typename T>
    auto apply_to(const arithmetic_sequence_t<T, Rank>& other) const
    {
        return iota<Rank>().map([this, other] (std::size_t i) { return this->operator[](i)(other[i]); });
    }




    /**
     * @brief      Return the sum of the elements in this sequence. It is
     *             assumed that ValueType{} is zero-initialized.
     *
     * @return     The sum of this sequence
     */
    ValueType sum() const
    {
        auto result = ValueType{};

        for (std::size_t i = 0; i < Rank; ++i)
            result += this->operator[](i);

        return result;
    }




    /**
     * @brief      Return true if all of the elements in this sequence evaluate
     *             to true.
     *
     * @return     A boolean
     */
    bool all() const
    {
        for (std::size_t i = 0; i < Rank; ++i)
            if (! this->operator[](i))
                return false;
        return true;
    }




    /**
     * @brief      Return true if any of the elements in this sequence evaluate
     *             to true.
     *
     * @return     A boolean
     */
    bool any() const
    {
        for (std::size_t i = 0; i < Rank; ++i)
            if (this->operator[](i))
                return true;
        return false;
    }




    /**
     * @brief      Return a reversed version of this sequence.
     *
     * @return     A sequence of the same rank and value type.
     */
    arithmetic_sequence_t reverse() const
    {
        return iota<Rank>().map([this] (std::size_t n) { return this->operator[](Rank - n - 1); });
    }




    /**
     * @brief      Return the transpose of this sequence, if its value type is
     *             also a sequence:
     *
     *             A.transpose()[i][j] == A[j][i]
     *
     * @return     A sequence of sequences with indexes switched
     */
    auto transpose() const
    {
        return iota<value_type::size()>().map([this] (std::size_t j) {
            return iota<Rank>().map([this, j] (std::size_t i) {
                return this->__data[i][j];
            });
        });
    }




    /**
     * @brief      The methods head and last are as defined in Haskell.
     *
     * @return     The first or last element in the sequence.
     */
    const ValueType& head() const { return this->at(0); }
    const ValueType& last() const { return this->at(Rank - 1); }




    /**
     * @brief      The methods init and tail as defined in Haskell.
     *
     * @return     What remains of the sequence after the last or first element
     *             is removed.
     */
    auto init() const { return iota<Rank - 1>().map([this] (std::size_t n) { return this->operator[](n); }); }
    auto tail() const { return iota<Rank - 1>().map([this] (std::size_t n) { return this->operator[](n + 1); }); }




    //=========================================================================
    template<typename T> auto operator* (const T& a) const { return binary_op(a, std::multiplies<>()); }
    template<typename T> auto operator/ (const T& a) const { return binary_op(a, std::divides<>()); }
    template<typename T> auto operator+ (const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::plus<>()); }
    template<typename T> auto operator- (const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::minus<>()); }
    template<typename T> auto operator* (const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::multiplies<>()); }
    template<typename T> auto operator/ (const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::divides<>()); }
    template<typename T> auto operator&&(const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::logical_and<>()); }
    template<typename T> auto operator||(const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::logical_or<>()); }
    template<typename T> auto operator==(const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::equal_to<>()); }
    template<typename T> auto operator!=(const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::not_equal_to<>()); }
    template<typename T> auto operator<=(const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::less_equal<>()); }
    template<typename T> auto operator>=(const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::greater_equal<>()); }
    template<typename T> auto operator< (const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::less<>()); }
    template<typename T> auto operator> (const arithmetic_sequence_t<T, Rank>& v) const { return binary_op(v, std::greater<>()); }
    auto operator+() const { return unary_op([] (auto&& x) { return +x; }); }
    auto operator-() const { return unary_op([] (auto&& x) { return -x; }); }




    //=========================================================================
    template<typename Function, std::size_t... Is>
    auto unary_op_impl(Function&& fn, std::index_sequence<Is...>) const
    {
        using result_type = arithmetic_sequence_t<std::invoke_result_t<Function, ValueType>, Rank>;
        return result_type{fn(this->template get<Is>())...};
    }

    template<typename Function, typename T, std::size_t... Is>
    auto binary_op_impl(Function&& fn, const T& a, std::index_sequence<Is...>) const
    {
        using result_type = arithmetic_sequence_t<std::invoke_result_t<Function, ValueType, T>, Rank>;
        return result_type{fn(this->template get<Is>(), a)...};
    }

    template<typename Function, typename T, std::size_t... Is>
    auto binary_op_impl(Function&& fn, const arithmetic_sequence_t<T, Rank>& v, std::index_sequence<Is...>) const
    {
        using result_type = arithmetic_sequence_t<std::invoke_result_t<Function, ValueType, T>, Rank>;
        return result_type{fn(this->template get<Is>(), v.template get<Is>())...};
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        return unary_op_impl(fn, std::make_index_sequence<Rank>());
    }

    template<typename T, typename Function>
    auto binary_op(const T& a, Function&& fn) const
    {
        return binary_op_impl(fn, a, std::make_index_sequence<Rank>());
    }

    template<typename T, typename Function>
    auto binary_op(const arithmetic_sequence_t<T, Rank>& v, Function&& fn) const
    {
        return binary_op_impl(fn, v, std::make_index_sequence<Rank>());
    }




    //=========================================================================
    ValueType __data[Rank];
};




/**
 * @brief      Class for working with a fixed-length sequence of native types -
 *             floats, doubles, etc.
 *
 * @tparam     ValueType    The underlying value type
 * @tparam     Rank         The dimensionality
 * @tparam     DerivedType  The CRTP class (google 'curiously recurring template
 *                          pattern')
 *
 * @note       You can add/subtract other sequences of the same rank and type,
 *             and you can multiply the whole sequence by scalars of the same
 *             value type. Arithmetic operations all return another instance of
 *             the same class type. For sequences that are covariant in the
 *             value type, see the arithmetic_sequence_t. This class is not used
 *             directly; you should inherit it with the CRTP pattern. This means
 *             that type identity is defined by the name of the derived class
 *             (different derived classes with the same rank and value type are
 *             not considered equal by the compiler).
 */
template<typename ValueType, std::size_t Rank, typename DerivedType>
struct mara::derivable_sequence_t
{
    //=========================================================================
    using value_type = ValueType;

    //=========================================================================
    constexpr std::size_t size()             const { return __impl.size(); }
    decltype(auto) data()                    const { return __impl.data(); }
    decltype(auto) begin()                   const { return __impl.begin(); }
    decltype(auto) end()                     const { return __impl.end(); }
    decltype(auto) operator[](std::size_t n) const { return __impl[n]; }
    decltype(auto) at(std::size_t n)         const { return __impl.at(n); }
    decltype(auto) data()                          { return __impl.data(); }
    decltype(auto) begin()                         { return __impl.begin(); }
    decltype(auto) end()                           { return __impl.end(); }
    decltype(auto) operator[](std::size_t n)       { return __impl[n]; }
    decltype(auto) at(std::size_t n)               { return __impl.at(n); }

    //=========================================================================
    bool operator==(const DerivedType& other) const { return __impl.operator==(other.__impl).all(); }
    bool operator!=(const DerivedType& other) const { return __impl.operator!=(other.__impl).any(); }
    DerivedType operator*(const ValueType& other)   const { return DerivedType{{__impl.operator*(other)}}; }
    DerivedType operator/(const ValueType& other)   const { return DerivedType{{__impl.operator/(other)}}; }
    DerivedType operator+(const DerivedType& other) const { return DerivedType{{__impl.operator+(other.__impl)}}; }
    DerivedType operator-(const DerivedType& other) const { return DerivedType{{__impl.operator-(other.__impl)}}; }
    DerivedType operator+()                         const { return DerivedType{{__impl.operator+()}}; }
    DerivedType operator-()                         const { return DerivedType{{__impl.operator-()}}; }

    //=========================================================================
    arithmetic_sequence_t<ValueType, Rank> __impl;
};




//=============================================================================
template<typename ValueType, std::size_t Rank>
auto mara::to_string(const arithmetic_sequence_t<ValueType, Rank>& sequence)
{
    using std::to_string;

    auto result = std::string("( ");

    for (std::size_t axis = 0; axis < Rank; ++axis)
    {
        if constexpr (std::is_same<ValueType, double>::value)
        {
            result += to_string(sequence[axis]) + " ";
        }
        else
        {
            result += to_string(sequence[axis]) + " ";            
        }
    }
    return result + ")";
}

template<typename ValueType, std::size_t Rank, typename DerivedType>
auto mara::to_string(const mara::derivable_sequence_t<ValueType, Rank, DerivedType>& sequence)
{
    return to_string(sequence.__impl);
}




/**
 * @brief      Make a C++ std::array - planned for C++20
 *
 * @param      t     The arguments to be put in the array (must have uniform
 *                   type)
 *
 * @tparam     D     Type to force conversion to
 *
 * @return     A std::array made from the arguments
 *
 * @note       See https://en.cppreference.com/w/cpp/experimental/make_array for
 *             implementation.
 */
template<typename... Args, typename ValueType>
auto mara::make_std_array(Args... args)
{
    return std::array<ValueType, sizeof...(Args)> {ValueType(args)...};
}




/**
 * @brief      Make a new sequence with inferred type and size.
 *
 * @param[in]  args       The elements
 *
 * @tparam     Args       The element types
 * @tparam     ValueType  The inferred type, if a common type can be inferred
 *
 * @return     The sequence
 * @note       You can override the inferred value type by doing e.g.
 *             make_sequence<size_t>(1, 2).
 */
template<typename... Args, typename ValueType>
auto mara::make_sequence(Args&&... args)
{
    return arithmetic_sequence_t<ValueType, sizeof...(Args)> {ValueType(args)...};
}




/**
 * @brief      Higher order function that turns its argument into a function
 *             that operates element-wise on fixed length sequences.
 *
 * @param[in]  f         The function
 *
 * @tparam     Function  The type of the function: arguments and return type
 *                       must all be the same
 *
 * @return     F := (a, b, ...) -> [f(a[i], b[i], ...) for i=0,rank]
 */
template<typename Function>
auto mara::lift(Function f)
{
    return [f] (auto... args)
    {
        auto arg_array = make_std_array(args...);
        auto result = typename decltype(arg_array)::value_type();
        constexpr std::size_t size = arg_array[0].size(); // ensures sequence is fixed length

        for (std::size_t n = 0; n < size; ++n)
        {
            result[n] = f(args[n]...);
        }
        return result;
    };
}




/**
 * @brief      Return a sequence of increasing integer values starting at 0.
 *
 * @tparam     Rank  The number of integers needed
 *
 * @return     The sequence
 */
template<std::size_t Rank>
mara::arithmetic_sequence_t<std::size_t, Rank> mara::iota()
{

    return detail::iota_impl(std::make_index_sequence<Rank>());
}
