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
#include <string>




//=============================================================================
namespace mara
{
    class rational_number_t;
    inline auto make_rational(int integer);
    inline auto make_rational(int numerator, int denominator);
    inline auto to_string(const rational_number_t& value);
}




/**
 * @brief      Class defining operations on rational numbers
 */
class mara::rational_number_t
{
public:

    //=========================================================================
    rational_number_t() {}
    rational_number_t(int integer) : num(integer), den(1) {}
    rational_number_t(int num, int den)
    : num(num)
    , den(den)
    {
        if (den == 0)
        {
            throw std::invalid_argument("rational_number_t: indeterminate (divide by zero)");
        }
        reduce();
    }

    rational_number_t operator+() const { return rational_number_t(+num, den); }
    rational_number_t operator-() const { return rational_number_t(-num, den); }

    double operator+(double other) const { return as_double() + other; }
    rational_number_t operator+(int other) const { return operator+(rational_number_t(other)); }
    rational_number_t operator+(const rational_number_t& other) const
    {
        auto a = num;
        auto b = den;
        auto c = other.num;
        auto d = other.den;
        return rational_number_t(a * d + b * c, b * d);
    }
    rational_number_t& operator+=(int other) { *this = *this + other; return *this; }

    double operator-(double other) const { return as_double() - other; }
    rational_number_t operator-(int other) const { return operator-(rational_number_t(other)); }
    rational_number_t operator-(const rational_number_t& other) const
    {
        auto a = num;
        auto b = den;
        auto c = other.num;
        auto d = other.den;
        return rational_number_t(a * d - b * c, b * d);
    }
    rational_number_t& operator-=(int other) { *this = *this - other; return *this; }

    double operator*(double other) const { return as_double() * other; }
    rational_number_t operator*(int other) const { return operator*(rational_number_t(other)); }
    rational_number_t operator*(const rational_number_t& other) const
    {
        auto a = num;
        auto b = den;
        auto c = other.num;
        auto d = other.den;
        return rational_number_t(a * c, b * d);
    }

    double operator/(double other) const { return as_double() / other; }
    rational_number_t operator/(int other) const { return operator/(rational_number_t(other)); }
    rational_number_t operator/(const rational_number_t& other) const
    {
        auto a = num;
        auto b = den;
        auto c = other.num;
        auto d = other.den;
        return rational_number_t(a * d, b * c);
    }

    bool operator==(int other) const { return operator==(rational_number_t(other)); }
    bool operator==(const rational_number_t& other) const
    {
        return num == other.num && den == other.den;
    }

    bool operator!=(int other) const { return operator!=(rational_number_t(other)); }
    bool operator!=(const rational_number_t& other) const
    {
        return num != other.num || den != other.den;
    }

    bool operator<(int other) const { return operator<(rational_number_t(other)); }
    bool operator<(const rational_number_t& other) const
    {
        return (*this - other).num < 0;
    }

    bool operator>(int other) const { return operator>(rational_number_t(other)); }
    bool operator>(const rational_number_t& other) const
    {
        return (*this - other).num > 0;
    }

    bool operator<=(int other) const { return operator<=(rational_number_t(other)); }
    bool operator<=(const rational_number_t& other) const
    {
        return (*this - other).num <= 0;
    }

    bool operator>=(int other) const { return operator>=(rational_number_t(other)); }
    bool operator>=(const rational_number_t& other) const
    {
        return (*this - other).num >= 0;
    }

    bool is_integral() const
    {
        return den == 1;
    }

    int as_integral() const
    {
        if (! is_integral())
        {
            throw std::logic_error("cannot convert non-integer rational "
                + std::to_string(num)
                + " / "
                + std::to_string(den)
                + " to integer");
        }
        return num;
    }

    double as_double() const
    {
        return double(num) / den;
    }

    const int& get_numerator() const { return num; }
    const int& get_denominator() const { return den; }

private:
    /**
     * @brief      Put the rational number in a canonical form: the fraction is
     *             reduced and the sign (if any) is put on the numerator.
     */
    void reduce()
    {
        if (num != 0)
        {
            auto g = gcd(std::abs(num), std::abs(den));
            num /= g;
            den /= g;
        }
        if (den < 0)
        {
            num *= -1;
            den *= -1;
        }
    }

    /**
     * @brief      Return the greatest common denominator of two non-negative
     *             integers. Implements the Binary GCD Algorithm.
     *
     */
    static int gcd(unsigned int a, unsigned int b)
    {
        unsigned int d = 0;

        while (a % 2 == 0 && b % 2 == 0)
        {
            a /= 2;
            b /= 2;
            d += 1;
        }
        while (a != b)
        {
            if      (a % 2 == 0) a /= 2;
            else if (b % 2 == 0) b /= 2;
            else if (a > b) a = (a - b) / 2;
            else            b = (b - a) / 2;
        }
        return a * (1 << d);
    }

    int num = 0;
    int den = 1;
};




//=============================================================================
auto mara::make_rational(int integer)
{
    return rational_number_t(integer);
}

auto mara::make_rational(int numerator, int denominator)
{
    return rational_number_t(numerator, denominator);
}

auto mara::to_string(const rational_number_t& value)
{
    return std::to_string(value.get_numerator()) + " / " + std::to_string(value.get_denominator());
}

namespace mara
{
    inline rational_number_t operator+(int a, const rational_number_t& b) { return  b + a; }
    inline rational_number_t operator-(int a, const rational_number_t& b) { return -b + a; }
}
