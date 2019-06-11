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
#include <exception>




//=============================================================================
namespace mara
{
    class bad_arithmetic_alternative_access;
    class mismatching_arithmetic_alternative;

    template<typename ValueType, typename AltType>
    class arithmetic_alternative_t;
}




//=============================================================================
class mara::bad_arithmetic_alternative_access : public std::exception
{
public:
    explicit bad_arithmetic_alternative_access(const char* message) :  msg(message) {}
    explicit bad_arithmetic_alternative_access(const std::string& message) : msg(message) {}
    const char* what() const throw () { return msg.c_str(); }
private:
    std::string msg;
};




//=============================================================================
class mara::mismatching_arithmetic_alternative : public std::exception
{
public:
    explicit mismatching_arithmetic_alternative(const char* message) :  msg(message) {}
    explicit mismatching_arithmetic_alternative(const std::string& message) : msg(message) {}
    const char* what() const throw () { return msg.c_str(); }
private:
    std::string msg;
};




/**
 * @brief      A specialized type like std::optional, but which wraps an
 *             arithmetic primary value and a non-arithmetic alternative value.
 *             Unary arithmetic operations apply to the primary value, and are
 *             pass-through for the alternative. Binary operations f(v1, v2) ->
 *             v3 are valid either when v1 and v2 are both primary, or when v1
 *             and v2 are neither primary but their alternative values compare
 *             equal. Otherwise binary operations throw an exception,
 *             mismatching_arithmetic_alternative. Binary operations yield
 *             arithmetic_alternative types whose primary value has the operator
 *             result type, but are only defined when both arguments have the
 *             same secondary type..
 *
 * @tparam     ValueType  The primary (arithmetic) type
 * @tparam     AltType    The alternative (pass-through) type
 */
template<typename ValueType, typename AltType>
class mara::arithmetic_alternative_t
{
public:

    using value_type = ValueType;
    using alternative_type = AltType;

    //=========================================================================
    arithmetic_alternative_t(ValueType value) : __has_value(true), __value(value) {}
    arithmetic_alternative_t(AltType alt)     : __has_value(false), __alt(alt) {}
    arithmetic_alternative_t()                : __has_value(false) {}

    bool has_value() const
    {
        return __has_value;
    }

    const ValueType& value() const
    {
        if (! has_value())
        {
            throw bad_arithmetic_alternative_access("no value");
        }
        return __value;
    }

    const AltType& alt() const
    {
        if (has_value())
        {
            throw bad_arithmetic_alternative_access("no alternative value");
        }
        return __alt;
    }

    template<typename S> auto operator+ (const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::plus<>()); }
    template<typename S> auto operator- (const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::minus<>()); }
    template<typename S> auto operator* (const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::multiplies<>()); }
    template<typename S> auto operator/ (const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::divides<>()); }
    template<typename S> auto operator&&(const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::logical_and<>()); }
    template<typename S> auto operator||(const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::logical_or<>()); }
    template<typename S> auto operator==(const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::equal_to<>()); }
    template<typename S> auto operator!=(const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::not_equal_to<>()); }
    template<typename S> auto operator<=(const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::less_equal<>()); }
    template<typename S> auto operator>=(const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::greater_equal<>()); }
    template<typename S> auto operator< (const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::less<>()); }
    template<typename S> auto operator> (const arithmetic_alternative_t<S, AltType>& other) const { return bin_op(other, std::greater<>()); }
    auto operator+() const { return unary_op([] (auto&& x) { return +x; }); }
    auto operator-() const { return unary_op([] (auto&& x) { return -x; }); }
    auto operator!() const { return unary_op(std::logical_not<>()); }

private:

    //=========================================================================
    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        using result_type = arithmetic_alternative_t<std::invoke_result_t<Function, ValueType>, AltType>;
        return has_value() ? result_type(fn(value())) : result_type(alt());
    }

    template<typename OtherValueType, typename Function>
    auto bin_op(const arithmetic_alternative_t<OtherValueType, AltType>& other, Function&& fn) const
    {
        using result_type = arithmetic_alternative_t<std::invoke_result_t<Function, ValueType, OtherValueType>, AltType>;
        return has_value()
        ? result_type(fn(value(), other.value()))
        : result_type(or_throw(alt() == other.alt(), alt(), mismatching_arithmetic_alternative("alt values do not compare equal")));
    }

    template<typename Value, typename Exception>
    static auto or_throw(bool ok, Value&& value, Exception&& e)
    {
        if (! ok)
        {
            throw std::move(e);
        }
        return value;
    }

    bool __has_value;
    ValueType __value;
    AltType __alt;
};
