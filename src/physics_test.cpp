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
#include "app_compile_opts.hpp"
#if MARA_COMPILE_SUBPROGRAM_TEST




#include <iostream>
#include "catch.hpp"
#include "physics_euler.hpp"
#define gamma_law_index (5. / 3)




//=============================================================================
TEST_CASE("Euler eigensystem is written correctly", "[mara::euler::primitive_t]")
{
    auto P = mara::euler::primitive_t()
    .with_gas_pressure(1.0)
    .with_mass_density(1.5)
    .with_velocity_1(0.2)
    .with_velocity_2(0.3)
    .with_velocity_3(0.4);
    auto A = P.flux_jacobian(gamma_law_index);
    auto L = P.eigenvalues(gamma_law_index);
    auto K = P.right_eigenvectors(gamma_law_index);
    auto Q = P.left_eigenvectors(gamma_law_index);

    auto A1 = K * L * Q;
    auto I = mara::identity_matrix<double, 5>();
    auto I1 = K * Q;

    for (std::size_t i = 0; i < 5; ++i)
    {
        for (std::size_t j = 0; j < 5; ++j)
        {
            INFO(i); INFO(j);
            CHECK(std::fabs((I - I1)(i, j)) < 1e-12);
            CHECK(std::fabs((A - A1)(i, j)) < 1e-12);
        }
    }
    REQUIRE(std::get<0>(P.eigensystem(gamma_law_index)) == P.eigenvalues(gamma_law_index));
}

TEST_CASE("Roe average states have the correct mathematical properties")
{
    auto Pl = mara::euler::primitive_t().with_gas_pressure(1.0).with_mass_density(1.5).with_velocity_2(0.2);
    auto Pr = mara::euler::primitive_t().with_gas_pressure(1.5).with_mass_density(1.0).with_velocity_3(0.5);
    auto Ul = Pl.to_conserved_density(gamma_law_index);
    auto Ur = Pr.to_conserved_density(gamma_law_index);

    REQUIRE(mara::euler::roe_average(Pl, Pr) == mara::euler::roe_average(Pr, Pl));

    auto B1 = mara::euler::roe_average(Pl, Pr).flux_jacobian(gamma_law_index) * mara::column_vector(Ur - Ul);
    auto B2 = Pr.flux(mara::unit_vector_t::on_axis_1(), gamma_law_index) - Pl.flux(mara::unit_vector_t::on_axis_1(), gamma_law_index);

    for (std::size_t i = 0; i < 5; ++i)
    {
        REQUIRE(B1(i, 0).value == Approx(B2[i].value));
    }
}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
