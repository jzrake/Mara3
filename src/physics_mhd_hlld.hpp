/**
 ==============================================================================
 Copyright 2019, Christopher Tiede

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
#include <cmath>
#include "core_geometric.hpp"
#include "core_matrix.hpp"
#include "core_dimensional.hpp"
#include "core_tuple.hpp"
#include "physics_mhd.hpp"

int sgn(double d)
{ 
	auto eps = 1e-8;
	return d<-eps ? -1 : d>eps; 
}



struct mara::mhd::riemann_hlld_variables_t 
{
	double gamma_law_index;

	mara::unit_vector_t nhat;
	mara::mhd::primitive_t PL;
	mara::mhd::primitive_t PR;

	mara::arithmetic_sequence_t<double, 3> v_para_l;
	mara::arithmetic_sequence_t<double, 3> v_para_r;
	mara::arithmetic_sequence_t<double, 3> v_perp_l;
	mara::arithmetic_sequence_t<double, 3> v_perp_r;

	mara::arithmetic_sequence_t<double, 3> b_para_l;
	mara::arithmetic_sequence_t<double, 3> b_para_r;
	mara::arithmetic_sequence_t<double, 3> b_perp_l;
	mara::arithmetic_sequence_t<double, 3> b_perp_r;

	double dl;  //density
	double dr;
	double ul;  //velocity along nhat
	double ur;
	double pl;  //pressure
	double pr;
	double b_along;

	double SR;
	double SL;
	double SLstar;
	double SRstar;
	double SM;

	double pstar;


	auto UL() const { return PL.to_conserved_density(gamma_law_index); }
	auto UR() const { return PR.to_conserved_density(gamma_law_index); }
	auto FL() const { return PL.flux(nhat, gamma_law_index); }
	auto FR() const { return PR.flux(nhat, gamma_law_index); }

	mara::mhd::conserved_density_t UL_star(void) const
	{
		auto beta = (SM - ul) / ( dl * (SL - ul) * (SL - SM) - b_along * b_along );

		auto num   = dl * (SL - ul) * (SL - ul) - b_along * b_along;
		auto denom = dl * (SL - ul) * (SL - SM) - b_along * b_along;
		auto zeta  = num / denom;

		auto vx_star = SM * nhat[0] + v_perp_l[0] - beta * b_along * b_perp_l[0];
		auto vy_star = SM * nhat[1] + v_perp_l[1] - beta * b_along * b_perp_l[1];
		auto vz_star = SM * nhat[2] + v_perp_l[2] - beta * b_along * b_perp_l[2];

		//IS IT RIGHT TO NOT CHANGE B_PARALLEL???
		auto bx_star = zeta * b_perp_l[0] + b_para_l[0];
		auto by_star = zeta * b_perp_l[1] + b_para_l[1];
		auto bz_star = zeta * b_perp_l[2] + b_para_l[2];

		auto e       = (PL.to_conserved_density(gamma_law_index))[4].value;
		auto bv      =  PL.bfield_dot_velocity();
		auto bv_star =  vx_star * bx_star + vy_star * by_star + vz_star * bz_star;
		auto e_star  = ((SL - ul) * e - pl * ul + pstar * SM + b_along * (bv - bv_star)) / (SL - SM);

		return mara::mhd::conserved_density_t{
			dl * (SL - ul) / (SL - SM),
			dl * vx_star,
			dl * vy_star,
			dl * vz_star,
			e_star ,
			bx_star,
			by_star,
			bz_star
		};
	}

	mara::mhd::conserved_density_t UR_star(void) const
	{
		auto beta = (SM - ur) / ( dr * (SR - ur) * (SR - SM) - b_along );

		auto num   = dr * (SR - ur) * (SR - ur) - b_along * b_along;
		auto denom = dr * (SR - ur) * (SR - SM) - b_along * b_along;
		auto zeta  = num / denom;

		auto vx_star = SM * nhat[0] + v_perp_r[0] - beta * b_along * b_perp_r[0];
		auto vy_star = SM * nhat[1] + v_perp_r[1] - beta * b_along * b_perp_r[1];
		auto vz_star = SM * nhat[2] + v_perp_r[2] - beta * b_along * b_perp_r[2];

		//IS IT RIGHT TO NOT CHANGE B_PARALLEL???
		auto bx_star = zeta * b_perp_r[0] + b_para_r[0];
		auto by_star = zeta * b_perp_r[1] + b_para_r[1];
		auto bz_star = zeta * b_perp_r[2] + b_para_r[2];

		auto e       = (PR.to_conserved_density(gamma_law_index))[4].value;
		auto bv      =  PR.bfield_dot_velocity();
		auto bv_star =  vx_star * bx_star + vy_star * by_star + vz_star * bz_star;
		auto e_star  = ((SR - ur) * e - pr * ur + pstar * SM + b_along * (bv - bv_star)) / (SR - SM);

		return mara::mhd::conserved_density_t{
			dr * ( SR - ur ) / (SR - SM),
			dr * vx_star,
			dr * vy_star,
			dr * vz_star,
			e_star ,
			bx_star,
			by_star,
			bz_star
		};
	}

	mara::mhd::conserved_density_t UL_starstar() const
	{
		auto b_sign  = sgn(b_along);

		if( b_sign != 0 ){
			auto Ul_star = UL_star();
			auto Ur_star = UR_star();
			auto d_star  = Ul_star[0].value;
			auto vx_star = Ul_star[1].value;
			auto vy_star = Ul_star[2].value;
			auto vz_star = Ul_star[3].value;
			auto bx_star = Ul_star[5].value;
			auto by_star = Ul_star[6].value;
			auto bz_star = Ul_star[7].value;

			auto bv_star = vx_star * bx_star + vy_star * by_star + vz_star * bz_star;

			auto v_starstar  = get_v_starstar(Ul_star, Ur_star, b_sign);
			auto b_starstar  = get_b_starstar(Ul_star, Ur_star, b_sign);
			auto bv_starstar = v_starstar[0] * b_starstar[0] + v_starstar[1] * b_starstar[1] + v_starstar[2] * b_starstar[2];
			auto e_starstar  = Ul_star[4].value - std::sqrt(d_star) * (bv_star - bv_starstar) * b_sign;

			return mara::mhd::conserved_density_t{
				d_star,
				d_star * v_starstar[0] * (!nhat[0]),
				d_star * v_starstar[1] * (!nhat[1]),
				d_star * v_starstar[2] * (!nhat[2]),
				e_starstar,
				b_starstar[0] * (!nhat[0]),
				b_starstar[1] * (!nhat[1]),
				b_starstar[2] * (!nhat[2])
			};
		}
		else{
			return UL_star();
		}
	}

	mara::mhd::conserved_density_t UR_starstar() const
	{
		auto b_sign  = sgn(b_along);

		if( b_along !=0 ){
			auto Ul_star = UL_star();
			auto Ur_star = UR_star();
			auto d_star  = Ur_star[0].value;
			auto vx_star = Ur_star[1].value;
			auto vy_star = Ur_star[2].value;
			auto vz_star = Ur_star[3].value;
			auto bx_star = Ur_star[5].value;
			auto by_star = Ur_star[6].value;
			auto bz_star = Ur_star[7].value;

			auto bv_star = vx_star * bx_star + vy_star * by_star + vz_star * bz_star;

			auto v_starstar  = get_v_starstar(Ul_star, Ur_star, b_sign);
			auto b_starstar  = get_b_starstar(Ul_star, Ur_star, b_sign);
			auto bv_starstar = v_starstar[0] * b_starstar[0] + v_starstar[1] * b_starstar[1] + v_starstar[2] * b_starstar[2];
			auto e_starstar  = Ur_star[4].value + std::sqrt(d_star) * (bv_star - bv_starstar) * b_sign;

			return mara::mhd::conserved_density_t{
				d_star,
				d_star * v_starstar[0] * (!nhat[0]),
				d_star * v_starstar[1] * (!nhat[1]),
				d_star * v_starstar[2] * (!nhat[2]),
				e_starstar,
				b_starstar[0] * (!nhat[0]),
				b_starstar[1] * (!nhat[1]),
				b_starstar[2] * (!nhat[2])
			};
		}
		else{
			return UR_star();
		}
	}

	mara::arithmetic_sequence_t<double,3> get_v_starstar(const mara::mhd::conserved_density_t& UL_star, const mara::mhd::conserved_density_t& UR_star, int b_sign) const
	{
		auto dl_star  = UL_star[0].value;
		auto dr_star  = UR_star[1].value;
		auto ul_star  = UL_star[1].value / dl_star;
		auto ur_star  = UR_star[1].value / dr_star;
		auto vl_star  = UL_star[2].value / dl_star;
		auto vr_star  = UR_star[2].value / dr_star;
		auto wl_star  = UL_star[3].value / dl_star;
		auto wr_star  = UR_star[3].value / dr_star;
		auto bxl_star = UL_star[5].value;
		auto bxr_star = UR_star[5].value;
		auto byl_star = UL_star[6].value;
		auto byr_star = UR_star[6].value;
		auto bzl_star = UL_star[7].value;
		auto bzr_star = UR_star[7].value;

		auto rt_dl_star = std::sqrt(dl_star);
		auto rt_dr_star = std::sqrt(dr_star);
		
		auto eta        =  rt_dl_star + rt_dr_star;
		auto u_starstar = (rt_dl_star * ul_star + rt_dr_star * ur_star + (bxr_star - bxl_star) * b_sign) / eta;
		auto v_starstar = (rt_dl_star * vl_star + rt_dr_star * vr_star + (byr_star - byl_star) * b_sign) / eta;
		auto w_starstar = (rt_dl_star * wl_star + rt_dr_star * wr_star + (bzr_star - bzl_star) * b_sign) / eta;

		return mara::arithmetic_sequence_t<double, 3>{u_starstar, v_starstar, w_starstar};
	}

	mara::arithmetic_sequence_t<double,3> get_b_starstar(const conserved_density_t& UL_star, const conserved_density_t& UR_star, int b_sign) const
	{
		auto dl_star  = UL_star[0].value;
		auto dr_star  = UR_star[1].value;
		auto ul_star  = UL_star[1].value / dl_star;
		auto ur_star  = UR_star[1].value / dr_star;
		auto vl_star  = UL_star[2].value / dl_star;
		auto vr_star  = UR_star[2].value / dr_star;
		auto wl_star  = UL_star[3].value / dl_star;
		auto wr_star  = UR_star[3].value / dr_star;
		auto bxl_star = UL_star[5].value;
		auto bxr_star = UR_star[5].value;
		auto byl_star = UL_star[6].value;
		auto byr_star = UR_star[6].value;
		auto bzl_star = UL_star[7].value;
		auto bzr_star = UR_star[7].value;

		auto rt_dl_star = std::sqrt(dl_star);
		auto rt_dr_star = std::sqrt(dr_star);
		
		auto eta         = rt_dl_star + rt_dr_star;
		auto bx_starstar = rt_dl_star * bxr_star + rt_dr_star * bxl_star + rt_dl_star * rt_dr_star * (ur_star - ul_star) * b_sign;
		auto by_starstar = rt_dl_star * byr_star + rt_dr_star * byl_star + rt_dl_star * rt_dr_star * (vr_star - vl_star) * b_sign;
		auto bz_starstar = rt_dl_star * bzr_star + rt_dr_star * bzl_star + rt_dl_star * rt_dr_star * (wr_star - wl_star) * b_sign;

		return mara::arithmetic_sequence_t<double, 3>{bx_starstar/eta, by_starstar/eta, bz_starstar/eta};
	}

	mara::mhd::flux_vector_t interface_flux() const
	{
		if      (0.0     <= SL                  ) return FL();
        else if ( SL     <= 0.0 && 0.0 <= SLstar) return FL() + (UL_star() - UL()) * make_velocity(SL);
        else if ( SLstar <= 0.0 && 0.0 <= SM    ) return FL() - (UL_star() + UL()) * make_velocity(SL) + (UL_starstar() - UL_star()) * make_velocity(SL);
        else if ( SM     <= 0.0 && 0.0 <= SRstar) return FR() - (UR_star() + UR()) * make_velocity(SR) + (UR_starstar() - UR_star()) * make_velocity(SR);
        else if ( SRstar <= 0.0 && 0.0 <= SR    ) return FR() + (UR_star() - UR()) * make_velocity(SR);
        else if ( SR     <= 0.0                 ) return FR();
        throw std::invalid_argument("riemann_hlld_variables_t::interface_flux");
	}

};


/**
 * @brief     Get fluxes from an array of conserved values
 * 
 */
inline mara::mhd::flux_vector_t mara::mhd::conserved_to_flux(
	const mara::mhd::conserved_density_t& U,
	const mara::unit_vector_t& nhat,
	double gamma_law_index)
{
	auto   temp_floor = 0.0;
	return mara::mhd::recover_primitive(U, gamma_law_index, temp_floor).flux(nhat, U);
}


/**
 * @brief     Compute quantities for hlld riemann solver
 * 
 */
inline mara::mhd::riemann_hlld_variables_t mara::mhd::compute_hlld_variables(
	const mara::mhd::primitive_t& Pl,
	const mara::mhd::primitive_t& Pr,
	const mara::unit_vector_t& nhat,
	double gamma_law_index)
{
	// Left and Right prims for calculations
	// ========================================================================
    auto dr  = Pr.mass_density();
    auto dl  = Pl.mass_density();
    auto ur  = Pr.velocity_along(nhat);
    auto ul  = Pl.velocity_along(nhat);
    auto pr  = Pr.gas_pressure();
    auto pl  = Pl.gas_pressure();

    // Force bfield in direction of nhat to be the same on left and right
	// ========================================================================
    auto b_along = 0.5 * ( Pr.bfield_along(nhat) + Pl.bfield_along(nhat) );


    // Left and Right parallel and perpendicular velocity and field vectors
	// ========================================================================
    auto v_para_l = nhat * ul;
    auto v_para_r = nhat * ur;
    auto b_para_l = nhat * b_along;
    auto b_para_r = nhat * b_along;
    auto v_perp_l = Pl.velocity() - v_para_l;
    auto v_perp_r = Pr.velocity() - v_para_r;
    auto b_perp_l = Pl.bfield()   - b_para_l;
    auto b_perp_r = Pr.bfield()   - b_para_r;

    // Get fast waves and the outermost signal speeds
	// ========================================================================
    auto FWl = Pl.fast_wave_speeds(nhat, gamma_law_index ); 
    auto FWr = Pr.fast_wave_speeds(nhat, gamma_law_index );
    auto SL  = std::min(FWl.m, FWr.m).value;
    auto SR  = std::max(FWl.p, FWr.p).value;


	// Get signal speed associated with the entropy wave
	// ========================================================================
    auto SM_top = (SR - ur) * dr * ur - (SL - ul) * dl * ul - pr + pl; 
    auto SM_bot = (SR - ur) * dr - (SL - ul) * dl;
    auto SM     = SM_top / SM_bot;

    // Get signal speed associated with internal alfven waves
	// ========================================================================
    auto dstar_l = dl * (SL - ul) / (SL - SM);
    auto dstar_r = dr * (SR - ur) / (SR - SM);
    auto SLstar = SM - std::abs(b_along) / dstar_l;
    auto SRstar = SM - std::abs(b_along) / dstar_r;

    // Get the average-total-pressure in Riemann fan
	// ========================================================================
    auto pstar_one = (SR - ur) * dr * pl - (SL - ul) * dl * pr;
    auto pstar_two = (SR - ur) * (SL - ul) * (ur - ul) * dl * dr;
    auto pstar_bot = (SR - ur) * dr - (SL - ul) * dl;
    auto pstar     = (pstar_one + pstar_two) / pstar_bot;


    auto r     = riemann_hlld_variables_t();
    r.nhat     = nhat;
    r.PL       = Pl;
    r.PR       = Pr;
    r.v_para_l = v_para_l;
    r.v_para_r = v_para_r;
    r.v_perp_l = v_perp_l;
    r.v_perp_r = v_perp_r;
    r.b_para_l = b_para_l;
    r.b_para_r = b_para_r;
    r.b_perp_l = b_perp_l;
    r.b_perp_r = b_perp_r;
    r.dl       = dl;
    r.dr       = dr;
    r.ul       = ul;
    r.ur       = ur;
    r.pr       = pr;
    r.pl       = pl;
    r.pstar    = pstar;
    r.b_along  = b_along;
    r.SL       = SL;
    r.SR       = SR;
    r.SLstar   = SLstar;
    r.SRstar   = SRstar;
    r.SM       = SM;
    r.gamma_law_index = gamma_law_index;
	return r;
}

/**
 * @brief     Calculate hlld riemann solution from left and right 
 *            primitive states
 * 
 */
inline mara::mhd::flux_vector_t mara::mhd::riemann_hlld(
		const primitive_t& Pl,
		const primitive_t& Pr,
		const unit_vector_t& nhat,
		double gamma_law_index)
{
	return compute_hlld_variables(Pl, Pr, nhat, gamma_law_index).interface_flux();
}







//=============================================================================