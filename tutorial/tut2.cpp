#include <iostream>
#include <cmath>
#include <cassert>
#include "core_ndarray.hpp"




/*
 * Turorial 2: array basics
 *
 * This shows how to start using Mara's n-dimensional arrays. The array library
 * is called ndarray-v2, and is maintained independently. There is more detail
 * on usage and implementation on github:
 *
 * https://github.com/jzrake/ndarray-v2
 */




/**
 * @brief      Create an array with 10 equally-spaced points, and loop over its
 *             values. Demonstrates:
 *
 *             - nd::linspace
 */
void create_a_simple_array_and_loop_over_it()
{
    auto A = nd::linspace(0.0, 1.0, 10); // Note: linspace includes both end-points

    for (auto a : A)
    {
        std::cout << a << std::endl;
    }

    // Or, if you also need the index you can do
    for (auto index : A.indexes())
    {
        std::cout << nd::to_string(index) << " " << A(index) << std::endl;
    }

    // You can index the array A like this:
    double value0_a = A(0);

    // In practice you'll rarely need to access array elements by index.
    // Instead, you'll generally apply functions to the whole array at once.
}




/**
 * @brief      Create two-dimensional arrays in a few different ways.
 *             Demonstrates:
 *
 *             - nd::zeros
 *             - nd::ones
 *             - nd::arange
 *             - nd::cartesian_product
 *             - nd::array_t::shape
 */
void create_simple_two_dimensional_arrays()
{
    // The nd::ones and nd::zeros functions:
    auto A = nd::zeros(10, 20);        // A is a 10 x 20 integer array whose elements are all 0.
    auto B = nd::ones<double>(10, 20); // B is a 10 x 20 double array whose elements are all 1.0.
    auto C = A + B * 2.5;              // C is a 10 x 20 double array whose elements are all 2.5.

    // You can index 2d arrays like this:
    std::cout << "c12 = " << C(1, 2) << std::endl;

    // You can try to add 2d arrays of different size, but you'll get an
    // exception at runtime:
    try {
        auto D = C + nd::zeros(10, 30);
    }
    catch (const std::exception& e)
    {
        std::cout << "Could not add C and D: " << e.what() << std::endl;
    }

    // You could try to add a 2d array to a 3d array, but you'd get a
    // compile-time error:

    // C + nd::zeros(10, 30, 60); // would not compile!

    // However, you can perform arithmetic between arrays and scalars. Just make
    // sure the scalar value comes *after* the array:
    auto D = C + 2.0; // (2.0 + C would not compile)

    // You can combine two (or more) 1d arrays to get get a 2d (or N-d)
    // cartesian product array:
    auto X = nd::linspace(0.0, 1.0, 10);
    auto Y = nd::arange(20);
    auto XY = nd::cartesian_product(X, Y);

    std::cout << "The shape of XY is " << nd::to_string(XY.shape()) << " (should be < 10 20 >)" << std::endl;

    // The elements of XY are 2-tuples. Since X has type double, and Y has type
    // int, the elements have type std::tuple<double, int>. To access the values
    // in a std::tuple, you have to use the std::get function:
    std::cout << "XY(1, 2) = " << std::get<0>(XY(1, 2)) << " " << std::get<1>(XY(1, 2)) << std::endl;
}




/**
 * @brief      Create new arrays by zipping arrays of the same shape together.
 *             Demonstrates:
 *
 *             - nd::zip
 *             - nd::unzip
 *             - nd::get
 *             - nd::all
 *             - equality comparison
 *             - pipe syntax
 */
void zip_arrays_together()
{
    auto X = nd::linspace(0.0, 1.0, 10);
    auto Y = nd::zeros<double>(10);
    auto Z = nd::arange(10);

    auto xyz = nd::zip(X, Y, Z); // xyz is a 1d array of 3-tuples

    // If you have an array of tuples, but you just wanted the elements at a
    // single tuple index, you can use nd::get like this:
    assert(nd::get<1>(xyz) == Y | nd::all());

    // Note that the equality operator applied to arrays returns an array of
    // bools, indicating the equality of the array elements. You check for the
    // equality of the whole array with the nd::all operator - it returns true
    // when all the elements of an array (casted to boolean) evaluate to true.

    // Above we also used the pipe syntax. This is short-hand for applying
    // functions to arrays - h(g(f(A))) means the same as A | f | g | h. If
    // you're used to member-function syntax, e.g. B = A.some_operator(), the
    // pipe does the same thing, B = A | some_operator, except it allows the
    // operators to be defined outside the array class. This means you can
    // logically extend the array class however you need, without having to
    // modify the class definition.

    // If you need to unpack an array of tuples into a tuple of arrays, you can
    // use nd::unzip:
    auto [x, y, z] = nd::unzip(xyz); // the assignment syntax on the left is a C++17 feature.
}




double gaussian(double x)
{
    return std::exp(-0.5 * x * x);
}




/**
 * @brief      Create new arrays by applying functions that operate
 *             element-wise. Demonstrates:
 *
 *             - nd::map
 */
void map_functions_over_arrays()
{
    auto x = nd::linspace(-1.0, 1.0, 100);

    // To create an array whose elements are "gaussian(xi) for each element xi
    // in x", we (unfortunately) cannot do `y = gaussian(x)` - because gaussian,
    // as defined above, is a function that takes doubles, not arrays of
    // doubles. Your first instinct would probably be to do something like this:

    // auto y = nd::zeros(10);
    // 
    // for (int i = 0; i < 10; ++i)
    // {
    //     y(i) = gaussian(x(i));
    // }

    // However, that would not compile! This is because arrays are read-only, or
    // "immutable". You can't assign values to their indexes. There's good
    // reason for this that will become clear as we go along (and there's one
    // exception to be covered later).

    // What we need to do is convert `gaussian` from a function that operates on
    // doubles to one that operates on arrays of doubles. We do this with the
    // `nd::map` function:

    auto gaussian_operating_on_arrays = nd::map(gaussian);
    auto y1 = gaussian_operating_on_arrays(x);

    // Using a lambda function and the pipe syntax, the whole sequence above can
    // be written compactly as:
    auto y2 = nd::linspace(-1.0, 1.0, 100) | nd::map([] (auto xi) { return std::exp(-0.5 * xi * xi); });

    assert(y1 == y2 | nd::all());
}




/**
 * @brief      Apply functions with multiple arguments to several arrays.
 *             Demonstrates:
 *
 *             - nd::zip
 *             - nd::map
 *             - nd::apply
 */
void map_functions_with_multiple_arguments_over_arrays()
{
    using namespace nd; // makes putting nd before everything unnecessary.

    // What if your function takes two variables, like this?
    auto gaussian2 = [] (double x, double sigma)
    {
        return std::exp(-0.5 * x * x / sigma / sigma);
    };

    // And you have an array of x's, and an array of sigma's...
    auto x = linspace(-1.0, 1.0, 100);
    auto s = linspace(1.0, 2.0, 100);

    // We can't use `map` on a function that takes two doubles to create a
    // function that takes two arrays. One solution is to zip the arrays
    // together, and then write a new function whose sole argument is a tuple,
    // like this:
    auto gaussian_operating_on_tuples = [] (std::tuple<double, double> t)
    {
        double x = std::get<0>(t);
        double s = std::get<1>(t);
        return std::exp(-0.5 * x * x / s / s);
    };

    // We could use it like this:
    auto y1 = zip(x, s) | map(gaussian_operating_on_tuples);

    // Since this pattern comes up a lot, there is a function, `apply`, that
    // writes the intermediate function for you. `apply` takes a function that
    // operates on N scalar values, and returns a function that operates on an
    // array of N-tuples. The whole sequence above may be written compactly like
    // this:
    auto y2 = zip(linspace(-1.0, 1.0, 100), linspace(1.0, 2.0, 100)) | apply(gaussian2);

    assert(y1 == y2 | all());
}




/**
 * @brief      Create memory-backed arrays by evaluating lazy arrays.
 *             Demonstrates:
 *
 *             - nd::to_shared
 */
void evaluate_lazy_arrays_to_memory_backed_arrays()
{
    // Mara's arrays are "lazy". This means that creating one does not allocate
    // any memory, or perform any computations. It just defines a rule for what
    // values should be returned when the array is indexed. In other words,
    // indexing an array really means "evaluating the rule that defines the
    // array". But you'll eventually need to store the array's elements in a
    // traditional data buffer, for example to output to the disk, or to cache
    // expensive operations that would otherwise be performed repeatedly. To do
    // this, you convert your array to a memory-backed array like so:
    auto A = nd::linspace(0.0, 1.0, 10000) | nd::to_shared();

    // You can use A in the same way as its lazy counterpart, but in addition
    // you can get a (const) pointer to A's data for efficient output to HDF5,
    // or sending via MPI. If the array has a member function to access its
    // data, it is a pointer to contiguous, row-major (C-style) memory.
    assert(A.data()[0] == 0.0);
}




//=============================================================================
int main(int argc, const char* argv[])
{
    create_a_simple_array_and_loop_over_it();
    create_simple_two_dimensional_arrays();
    zip_arrays_together();
    map_functions_over_arrays();
    map_functions_with_multiple_arguments_over_arrays();
    evaluate_lazy_arrays_to_memory_backed_arrays();
    return 0;
}
