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
#include "physics_mhd.hpp"
#include "physics_mhd_hlld.hpp"
#define gamma_law_index (5. / 3)

void test_hlld_jump_condtions( mara::mhd::primitive_t Pl, mara::mhd::primitive_t Pr, mara::unit_vector_t nhat, double gamma)
{
    auto vars = mara::mhd::compute_hlld_variables(Pl, Pr, nhat, gamma);
    auto pstar = vars.pstar;
    auto SM  = vars.SM;
    auto SL  = vars.SL;
    auto SR  = vars.SR;
    auto SLs = vars.SLstar;
    auto SRs = vars.SRstar;

    // CHECK WAVESPEEDS
    CHECK( SL  < SLs );
    CHECK( SLs < SM );
    CHECK( SM  < SRs );
    CHECK( SRs < SR );


    // CHECK SL jump condition
    // ====================================================================
    auto ULstar = vars.UL_star();
    auto dLstar  = ULstar[0].value;
    auto vxLstar = ULstar[1].value / dLstar;
    auto vyLstar = ULstar[2].value / dLstar;
    auto vzLstar = ULstar[3].value / dLstar;
    auto eLstar  = ULstar[4].value;
    auto bxLstar = ULstar[5].value;
    auto byLstar = ULstar[6].value;
    auto bzLstar = ULstar[7].value;
    auto bvL_star = vxLstar * bxLstar + vyLstar * byLstar + bzLstar * vzLstar;
    auto Ul = vars.UL();

    //Get b-along nhat
    auto balong = 0.5 * (Pl.bfield_along(nhat) + Pr.bfield_along(nhat));

    auto Fl = mara::mhd::conserved_to_flux(Ul, nhat, gamma);
    CHECK(SL * Ul[0].value - Fl[0].value == Approx(SL * dLstar - dLstar * SM));
    CHECK(SL * Ul[1].value - Fl[1].value == Approx(SL * dLstar * vxLstar - (dLstar * vxLstar * SM - balong * bxLstar + pstar * nhat[0])));
    CHECK(SL * Ul[2].value - Fl[2].value == Approx(SL * dLstar * vyLstar - (dLstar * vyLstar * SM - balong * byLstar + pstar * nhat[1])));
    CHECK(SL * Ul[3].value - Fl[3].value == Approx(SL * dLstar * vzLstar - (dLstar * vzLstar * SM - balong * bzLstar + pstar * nhat[2])));
    CHECK(SL * Ul[4].value - Fl[4].value == Approx(SL * eLstar - ( (eLstar + pstar)*SM - balong * bvL_star )));
    CHECK(SL * Ul[5].value - Fl[5].value == Approx(SL * bxLstar - (bxLstar * SM - balong * vxLstar)));
    CHECK(SL * Ul[6].value - Fl[6].value == Approx(SL * byLstar - (byLstar * SM - balong * vyLstar)));
    CHECK(SL * Ul[7].value - Fl[7].value == Approx(SL * bzLstar - (bzLstar * SM - balong * vzLstar)));

    // CHECK SR jump condition
    // ====================================================================
    auto URstar = vars.UR_star();
    auto dRstar  = URstar[0].value;
    auto vxRstar = URstar[1].value / dRstar;
    auto vyRstar = URstar[2].value / dRstar;
    auto vzRstar = URstar[3].value / dRstar;
    auto eRstar  = URstar[4].value;
    auto bxRstar = URstar[5].value;
    auto byRstar = URstar[6].value;
    auto bzRstar = URstar[7].value;
    auto bvR_star = vxRstar * bxRstar + vyRstar * byRstar + bzRstar * vzRstar;
    
    auto UR = vars.UR();
    auto FR = mara::mhd::conserved_to_flux(UR, nhat, gamma);
    CHECK(SR * UR[0].value - FR[0].value == Approx(SR * dRstar - dRstar * SM));
    CHECK(SR * UR[1].value - FR[1].value == Approx(SR * dRstar * vxRstar - (dRstar * vxRstar * SM - balong * bxRstar + pstar * nhat[0])));
    CHECK(SR * UR[2].value - FR[2].value == Approx(SR * dRstar * vyRstar - (dRstar * vyRstar * SM - balong * byRstar + pstar * nhat[1])));
    CHECK(SR * UR[3].value - FR[3].value == Approx(SR * dRstar * vzRstar - (dRstar * vzRstar * SM - balong * bzRstar + pstar * nhat[2])));
    CHECK(SR * UR[4].value - FR[4].value == Approx(SR * eRstar - ( (eRstar + pstar)*SM - balong * bvR_star )));
    CHECK(SR * UR[5].value - FR[5].value == Approx(SR * bxRstar - (bxRstar * SM - balong * vxRstar)));
    CHECK(SR * UR[6].value - FR[6].value == Approx(SR * byRstar - (byRstar * SM - balong * vyRstar)));
    CHECK(SR * UR[7].value - FR[7].value == Approx(SR * bzRstar - (bzRstar * SM - balong * vzRstar)));
    

    // CHECK SM jump condition
    // ====================================================================
    // Right
    auto URss  = vars.UR_starstar();
    auto dRss  = URss[0].value;
    auto vxRss = URss[1].value / dRss;
    auto vyRss = URss[2].value / dRss;
    auto vzRss = URss[3].value / dRss;
    auto bxRss = URss[5].value;
    auto byRss = URss[6].value;
    auto bzRss = URss[7].value;
    auto bvRss = vxRss * bxRss + vyRss * byRss + vzRss * bzRss;

    // Left
    auto ULss = vars.UL_starstar();
    auto dLss  = ULss[0].value;
    auto vxLss = ULss[1].value / dLss;
    auto vyLss = ULss[2].value / dLss;
    auto vzLss = ULss[3].value / dLss;
    auto bxLss = ULss[5].value;
    auto byLss = ULss[6].value;
    auto bzLss = ULss[7].value;
    auto bvLss = vxLss * bxLss + vyLss * byLss + vzLss * bzLss;


    // CHECK(vxLstar == vxRstar);
    // CHECK(vxLss == vxRss);
    // CHECK(bxLss == bxRss);

    CHECK(SM * dLstar * vxLss - (dLstar * vxLss * SM - balong * bxLss + pstar * nhat[0]) == Approx(SM * dRstar * vxRss - (dRstar * vxRss * SM - balong * bxRss + pstar * nhat[0])));
    CHECK(SM * dLstar * vyLss - (dLstar * vyLss * SM - balong * byLss + pstar * nhat[1]) == Approx(SM * dRstar * vyRss - (dRstar * vyRss * SM - balong * byRss + pstar * nhat[1])));
    CHECK(SM * dLstar * vzLss - (dLstar * vzLss * SM - balong * bzLss + pstar * nhat[2]) == Approx(SM * dRstar * vzRss - (dRstar * vzRss * SM - balong * bzRss + pstar * nhat[2])));
    CHECK(SM * bxLss - (bxLss * SM - balong * vxLss ) == Approx(SM * bxRss - (bxRss * SM - balong * vxRss )));
    CHECK(SM * byLss - (byLss * SM - balong * vyLss ) == Approx(SM * byRss - (byRss * SM - balong * vyRss )));
    CHECK(SM * bzLss - (bzLss * SM - balong * vzLss ) == Approx(SM * bzRss - (bzRss * SM - balong * vzRss )));
    CHECK(SM * ULss[4].value - ((ULss[4].value + pstar)*SM - balong * bvLss) == Approx(SM * URss[4].value - ((URss[4].value + pstar)*SM - balong * bvRss)));

}


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
    SECTION("U -> P and P -> U work correctly")
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
    }

    SECTION("Q -> P and P -> Q work correctly away from the origin")
    {
        auto x = mara::iso2d::location_2d_t{1.0, 2.0};
        auto P = mara::iso2d::primitive_t()
        .with_sigma(2.0)
        .with_velocity_x(0.5)
        .with_velocity_y(1.5);

        auto U = P.to_conserved_angmom_per_area(x);
        REQUIRE(mara::iso2d::recover_primitive(U, x).sigma() == P.sigma());
        REQUIRE(mara::iso2d::recover_primitive(U, x).velocity_x() == P.velocity_x());
        REQUIRE(mara::iso2d::recover_primitive(U, x).velocity_y() == P.velocity_y());
    }

    SECTION("Q -> P and P -> Q work correctly away from the origin")
    {
        auto x = mara::iso2d::location_2d_t{1e-8, 1e-8};
        auto P = mara::iso2d::primitive_t()
        .with_sigma(2.0)
        .with_velocity_x(0.5)
        .with_velocity_y(1.5);

        auto U = P.to_conserved_angmom_per_area(x);
        REQUIRE(mara::iso2d::recover_primitive(U, x).sigma() == P.sigma());
        REQUIRE(mara::iso2d::recover_primitive(U, x).velocity_x() == Approx(P.velocity_x()).epsilon(1e-14));
        REQUIRE(mara::iso2d::recover_primitive(U, x).velocity_y() == Approx(P.velocity_y()).epsilon(1e-14));
    }

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

TEST_CASE("MHD system", "[mara::mhd::primitive_t]")
{
    SECTION("Cons2Prim and Prim2Cons consistent")
    {
        auto P1 = mara::mhd::primitive_t()
        .with_mass_density(2.0)
        .with_velocity_1(0.5)
        .with_velocity_2(1.5)
        .with_velocity_3(1.0)
        .with_bfield_1(1.5)
        .with_bfield_2(1.0)
        .with_bfield_3(2.3);

        auto P2 = mara::mhd::recover_primitive( P1.to_conserved_density(1.4), 1.4, 0.0 );

        REQUIRE( P1==P2 );
    }
    SECTION("Check HLLD jump conditions")
    {       
        auto Pl = mara::mhd::primitive_t()
        .with_mass_density(1.7)
        .with_gas_pressure(0.1)
        .with_velocity_1(10.2)
        .with_velocity_2(204.2)
        .with_velocity_3(-9.1)
        .with_bfield_1(1.2)
        .with_bfield_2(-2.4)
        .with_bfield_3(-9.1);

        auto Pr = mara::mhd::primitive_t()
        .with_mass_density(200.2)
        .with_gas_pressure(1000)
        .with_velocity_1(-101.34)
        .with_velocity_2(-7.3)
        .with_velocity_3(12.0)
        .with_bfield_1(-90.2)
        .with_bfield_2(68.1)
        .with_bfield_3(160.3);

        auto gamma = 1.66667;
        auto n_x = mara::unit_vector_t::on_axis(0);
        auto n_y = mara::unit_vector_t::on_axis(1);
        auto n_z = mara::unit_vector_t::on_axis(2);

        test_hlld_jump_condtions(Pl.with_bfield_1(100.2), Pr.with_bfield_1(100.2), n_x, gamma);
        test_hlld_jump_condtions(Pl.with_bfield_2(100.2), Pr.with_bfield_2(100.2), n_y, gamma);
        test_hlld_jump_condtions(Pl.with_bfield_3(100.2), Pr.with_bfield_3(100.2), n_z, gamma);
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
