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




#include "core_catch.hpp"
#include "core_geometric.hpp"
#include "physics_euler.hpp"
#include "physics_iso2d.hpp"
#include "model_two_body.hpp"
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
    auto I = mara::identity_matrix<mara::unit_scalar<double>, 5>();
    auto I1 = K * Q;

    for (std::size_t i = 0; i < 5; ++i)
    {
        for (std::size_t j = 0; j < 5; ++j)
        {
            INFO(i); INFO(j);
            CHECK(std::fabs((I - I1)(i, j).value) < 1e-12);
            CHECK(std::fabs((A - A1)(i, j).value) < 1e-12);
        }
    }
    REQUIRE(std::get<0>(P.eigensystem(gamma_law_index)) == P.eigenvalues(gamma_law_index));
}

TEST_CASE("Roe average states have the correct mathematical properties")
{
    auto g = gamma_law_index;
    auto Pl = mara::euler::primitive_t().with_gas_pressure(1.0).with_mass_density(1.5).with_velocity_2(0.2);
    auto Pr = mara::euler::primitive_t().with_gas_pressure(1.5).with_mass_density(1.0).with_velocity_3(0.5);
    auto Ul = Pl.to_conserved_density(g);
    auto Ur = Pr.to_conserved_density(g);


    SECTION("The Roe average is symmetric")
    {
        REQUIRE(mara::euler::roe_average(Pl, Pr) == mara::euler::roe_average(Pr, Pl));
    }


    SECTION("The Roe average is homogeneous (e.g. Condition 3, Section 3.4 from Marti & Muller's review)")
    // http://www.mpa-garching.mpg.de/hydro/index.shtml
    {
        auto B1 = mara::euler::roe_average(Pl, Pr).flux_jacobian(g) * mara::column_vector(Ur - Ul);
        auto B2 = Pr.flux(mara::unit_vector_t::on_axis_1(), g) - Pl.flux(mara::unit_vector_t::on_axis_1(), g);

        for (std::size_t i = 0; i < 5; ++i)
        {
            REQUIRE(B1(i, 0).value == Approx(B2[i].value));
        }
    }
}

TEST_CASE("Isothermal 2d system", "[mara::iso2d::primitive_t]")
{
    auto P = mara::iso2d::primitive_t()
    .with_sigma(2.0)
    .with_velocity_x(0.5)
    .with_velocity_y(1.5);

    auto U = P.to_conserved_per_area();
    REQUIRE(P.velocity_x() == 0.5);
    REQUIRE(mara::get<0>(U).value == P.sigma());
    REQUIRE(mara::get<1>(U).value == P.sigma() * P.velocity_x());
    REQUIRE(mara::get<2>(U).value == P.sigma() * P.velocity_y());

    SECTION("HLLC gets zero contact speed for zero-velocity, equal pressure states")
    {
        auto Pl = mara::iso2d::primitive_t().with_sigma(1.0).with_velocity_x(0.0).with_velocity_y(0.0);
        auto Pr = Pl.with_sigma(2.0);
        auto al2 = 1.0;
        auto ar2 = 1.0 / 2.0;
        auto vars = mara::iso2d::compute_hllc_variables(Pl, Pr, al2, ar2, mara::unit_vector_t::on_axis_1());

        REQUIRE(Pl.gas_pressure(al2) == Approx(Pr.gas_pressure(ar2)));
        REQUIRE(vars.contact_speed() == 0.0);
    }
}

TEST_CASE("Two body gravity model works as expected", "[model_two_body]")
{
    auto dt = 1e-4;
    auto binary = mara::two_body_parameters_t{};
    auto state0 = mara::compute_two_body_state(binary, dt * 0.0);
    auto state1 = mara::compute_two_body_state(binary, dt * 0.5);
    auto state2 = mara::compute_two_body_state(binary, dt * 1.0);
    auto dx1 = state2.body1.position_x - state0.body1.position_x;
    auto dy1 = state2.body1.position_y - state0.body1.position_y;
    auto dx2 = state2.body2.position_x - state0.body2.position_x;
    auto dy2 = state2.body2.position_y - state0.body2.position_y;
    CHECK(dx1 / dt == Approx(state1.body1.velocity_x));
    CHECK(dy1 / dt == Approx(state1.body1.velocity_y));
    CHECK(dx2 / dt == Approx(state1.body2.velocity_x));
    CHECK(dy2 / dt == Approx(state1.body2.velocity_y));
    CHECK(state0.body1.position_x ==  0.5);
    CHECK(state0.body2.position_x == -0.5);
    CHECK(state0.body1.position_y ==  0.0);
    CHECK(state0.body2.position_y ==  0.0);
    CHECK(state0.body1.velocity_y > 0.0);
    CHECK(state0.body2.velocity_y < 0.0);
    CHECK(state2.body1.velocity_x < 0.0);
    CHECK(state2.body2.velocity_x > 0.0);
    CHECK(state2.body2.position_x * state2.body2.velocity_y - state2.body2.position_y * state2.body2.velocity_x > 0.0);
    CHECK(state0.body2.position_x * state0.body2.velocity_y - state0.body2.position_y * state0.body2.velocity_x ==
          state2.body2.position_x * state2.body2.velocity_y - state2.body2.position_y * state2.body2.velocity_x);

}

#endif // MARA_COMPILE_SUBPROGRAM_TEST
