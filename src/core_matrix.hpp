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
#include <type_traits>        // std::invoke_result_t
#include <functional>         // std::plus, std::minus, etc




//=============================================================================
namespace mara
{
    template<typename ValueType, std::size_t NumRows, std::size_t NumCols>
    struct matrix_t;

    template<typename ValueType, std::size_t NumRows, std::size_t NumCols>
    inline auto zero_matrix();

    template<typename ValueType, std::size_t Rank>
    inline auto identity_matrix();

    template<typename ValueType, typename... Args>
    inline auto diagonal_matrix(Args... args);

    template<typename ContainerType>
    inline auto row_vector(ContainerType container);

    template<typename ContainerType>
    inline auto column_vector(ContainerType container);

    template<typename T1, typename T2, std::size_t NumRows, std::size_t NumCols, std::size_t NumCols1Rows2>
    auto matrix_product(matrix_t<T1, NumRows, NumCols1Rows2> M1, matrix_t<T2, NumCols1Rows2, NumCols> M2);
}




/**
 * @brief      A class representing a matrix
 *
 * @tparam     ValueType  The value type of the elemnts
 * @tparam     NumRows    The number of rows in the matrix
 * @tparam     NumCols    The number of columns
 */
template<typename ValueType, std::size_t NumRows, std::size_t NumCols>
struct mara::matrix_t
{
public:

    //=========================================================================
    using value_type = ValueType;
    static constexpr std::size_t num_rows = NumRows;
    static constexpr std::size_t num_cols = NumCols;

    //=========================================================================
    const ValueType& operator()(std::size_t i, std::size_t j) const
    {
        return memory[i][j];
    }

    ValueType& operator()(std::size_t i, std::size_t j)
    {
        return memory[i][j];
    }

    bool operator==(const matrix_t& other) const
    {
        for (std::size_t i = 0; i < num_rows; ++i)
            for (std::size_t j = 0; j < num_cols; ++j)
                if (operator()(i, j) != other(i, j))
                    return false;
        return true;
    }

    bool operator!=(const matrix_t& other) const
    {
        for (std::size_t i = 0; i < num_rows; ++i)
            for (std::size_t j = 0; j < num_cols; ++j)
                if (operator()(i, j) != other(i, j))
                    return true;
        return false;
    }

    template<typename T2> auto operator+(const matrix_t<T2, NumRows, NumCols>& other) const { return binary_op(other, std::plus<>()); }
    template<typename T2> auto operator-(const matrix_t<T2, NumRows, NumCols>& other) const { return binary_op(other, std::minus<>()); }
    template<typename T2> auto operator*(const T2& scale) const { return binary_op(scale, std::multiplies<>()); }
    template<typename T2> auto operator/(const T2& scale) const { return binary_op(scale, std::divides<>()); }
    template<typename T2, std::size_t NumRows2, std::size_t NumCols2>
    auto operator*(const matrix_t<T2, NumRows2, NumCols2>& other) const
    {
        return matrix_product(*this, other);
    }

    //=========================================================================
    ValueType memory[NumRows][NumCols];

private:
    //=========================================================================
    template<typename Function>
    auto binary_op(const matrix_t& v, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = matrix_t();

        for (std::size_t i = 0; i < NumRows; ++i)
        {
            for (std::size_t j = 0; j < NumCols; ++j)
            {
                result(i, j) = fn(_(i, j), v(i, j));
            }
        }
        return result;
    }

    template<typename T, typename Function>
    auto binary_op(const T& a, Function&& fn) const
    {
        const auto& _ = *this;
        auto result = matrix_t<std::invoke_result_t<Function, ValueType, T>, NumRows, NumCols>();

        for (std::size_t i = 0; i < NumRows; ++i)
        {
            for (std::size_t j = 0; j < NumCols; ++j)
            {
                result(i, j) = fn(_(i, j), a);
            }
        }
        return result;
    }

    template<typename Function>
    auto unary_op(Function&& fn) const
    {
        const auto& _ = *this;
        auto result = matrix_t<std::invoke_result_t<Function, ValueType>, NumRows, NumCols>();

        for (std::size_t i = 0; i < NumRows; ++i)
        {
            for (std::size_t j = 0; j < NumCols; ++j)
            {
                result(i, j) = fn(_(i, j));
            }
        }
        return result;
    }
};




/**
 * @brief      Return a matrix of all zeros.
 *
 * @tparam     ValueType  The element type
 * @tparam     NumRows    The number of rows
 * @tparam     NumCols    The number of columns
 *
 * @return     The matrix
 */
template<typename ValueType, std::size_t NumRows, std::size_t NumCols>
auto mara::zero_matrix()
{
    auto result = matrix_t<ValueType, NumRows, NumCols>();

    for (std::size_t i = 0; i < NumRows; ++i)
        for (std::size_t j = 0; j < NumCols; ++j)
            result(i, j) = ValueType(0);

    return result;
}




/**
 * @brief      Return the identity matrix for the given value type and rank.
 *
 * @tparam     ValueType  { description }
 * @tparam     Rank       { description }
 *
 * @return     { description_of_the_return_value }
 */
template<typename ValueType, std::size_t Rank>
auto mara::identity_matrix()
{
    auto result = matrix_t<ValueType, Rank, Rank>();

    for (std::size_t i = 0; i < Rank; ++i)
        for (std::size_t j = 0; j < Rank; ++j)
            result(i, j) = ValueType(i == j);

    return result;
}




/**
 * @brief      Return a matrix with the given values on the diagonal.
 *
 * @param[in]  args       The values to go on the diagonal
 *
 * @tparam     ValueType  The element value type (must be given explicitly)
 * @tparam     Args       The args (must be castable to ValueType)
 *
 * @return     The diagonal matrix
 */
template<typename ValueType, typename... Args>
auto mara::diagonal_matrix(Args... args)
{
    constexpr std::size_t Rank = sizeof...(args);
    auto result = matrix_t<ValueType, Rank, Rank>();

    ValueType arg_list[Rank] = {ValueType(args)...};

    for (std::size_t i = 0; i < Rank; ++i)
        for (std::size_t j = 0; j < Rank; ++j)
            result(i, j) = i == j ? arg_list[i] : ValueType();

    return result;
}




/**
 * @brief      Return a 1 x N matrix from some type of fixed-length container.
 *             The element type is the same as that of the container.
 *
 * @param[in]  container      The container
 *
 * @tparam     ContainerType  The type of the container
 *
 * @return     The matrix
 */
template<typename ContainerType>
auto mara::row_vector(ContainerType container)
{
    using value_type = typename ContainerType::value_type;
    constexpr std::size_t Rank = container.size();

    auto result = matrix_t<value_type, 1, Rank>();

    for (std::size_t j = 0; j < Rank; ++j)
    {
        result(0, j) = container[j];
    }
    return result;
}




/**
 * @brief      Return a N x 1 matrix from some type of fixed-length container.
 *             The element type is the same as that of the container.
 *
 * @param[in]  container      The container
 *
 * @tparam     ContainerType  The type of the container
 *
 * @return     The matrix
 */
template<typename ContainerType>
auto mara::column_vector(ContainerType container)
{
    using value_type = typename ContainerType::value_type;
    constexpr std::size_t Rank = container.size();

    auto result = matrix_t<value_type, Rank, 1>();

    for (std::size_t i = 0; i < Rank; ++i)
    {
        result(i, 0) = container[i];
    }
    return result;
}




/**
 * @brief      Multiply two matrices of compatible value types and row and
 *             column number.
 *
 * @param[in]  M1             The first matrix
 * @param[in]  M2             The second matrix
 *
 * @tparam     T1             The value type of M1
 * @tparam     T2             The value type of M2
 * @tparam     NumRows        The number of rows in the resulting matrix, and in
 *                            M1
 * @tparam     NumCols        The number of columns in the result matrix, and in
 *                            M2
 * @tparam     NumCols1Rows2  The number of columns in M1 and rows in M2
 *
 * @return     The matrix product
 */
template<typename T1, typename T2, std::size_t NumRows, std::size_t NumCols, std::size_t NumCols1Rows2>
auto mara::matrix_product(matrix_t<T1, NumRows, NumCols1Rows2> M1, matrix_t<T2, NumCols1Rows2, NumCols> M2)
{
    using result_value_type = std::invoke_result_t<std::multiplies<>, T1, T2>;
    auto result = matrix_t<result_value_type, NumRows, NumCols>();

    for (std::size_t i = 0; i < NumRows; ++i)
    {
        for (std::size_t j = 0; j < NumCols; ++j)
        {
            auto element = result_value_type();

            for (std::size_t m = 0; m < NumCols1Rows2; ++m)
            {
                element += M1(i, m) * M2(m, j);
            }
            result(i, j) = element;
        }
    }
    return result;
}
