//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file  dustfluids_roe_solver.cpp
//! \brief Roe's linearized Riemann solver.
//!
//!Computes 1D fluxes using Roe's linearization.  When Roe's method fails because of
//!negative density in the intermediate states, LLF fluxes are used instead (only density,
//!not pressure, is checked in this version).
//!
//!REFERENCES:
//!- P. Roe, "Approximate Riemann solvers, parameter vectors, and difference schemes",
//!  JCP, 43, 357 (1981).

// C headers

// C++ headers
#include <algorithm>  // max()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "dustfluids.hpp"

namespace {
// prototype for function to compute Roe fluxes from eigenmatrices
inline void RoeFlux(const int irho, const Real df_prim_roe[], const Real df_du[],
      const Real df_prim_li[], const Real cs, Real dust_flux[], Real df_ev[], int &llf_flag);
} // namespace

//----------------------------------------------------------------------------------------
//! \fn void DustFluids::RoeRiemannSolverDustFluids
//  \brief The Roe Riemann solver for dust fluids (spatially isothermal)

void DustFluids::RoeRiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &prim_df_l,
        AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {
  Real df_prim_li[(NDUSTVAR)], df_prim_ri[(NDUSTVAR)], df_prim_roe[(NDUSTVAR)];
  Real df_fl[(NDUSTVAR)],      df_fr[(NDUSTVAR)],      df_flxi[(NDUSTVAR)];
	Real df_ev[(NDUSTVAR)],      df_du[(NDUSTVAR)];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int idust = n;
    int irho  = 4*idust;
    int ivx   = (IVX + ((index-IVX))%3)   + irho;
    int ivy   = (IVX + ((index-IVX)+1)%3) + irho;
    int ivz   = (IVX + ((index-IVX)+2)%3) + irho;

#pragma omp simd private(df_prim_li, df_prim_ri, df_prim_roe, df_flxi, df_fl, df_fr, df_ev, df_du)
		for (int i=il; i<=iu; ++i) {
      const Real &cs = cs_dustfluids_array(idust, k, j, i);

			//--- Step 1.  Load L/R states into local variables
			df_prim_li[irho] = prim_df_l(irho, i);
			df_prim_li[ivx]  = prim_df_l(ivx,  i);
			df_prim_li[ivy]  = prim_df_l(ivy,  i);
			df_prim_li[ivz]  = prim_df_l(ivz,  i);

			df_prim_ri[irho] = prim_df_r(irho, i);
			df_prim_ri[ivx]  = prim_df_r(ivx,  i);
			df_prim_ri[ivy]  = prim_df_r(ivy,  i);
			df_prim_ri[ivz]  = prim_df_r(ivz,  i);

			//--- Step 2.  Compute Roe-averaged data from left- and right-states
			Real sqrtdl  = std::sqrt(df_prim_li[irho]);
			Real sqrtdr  = std::sqrt(df_prim_ri[irho]);
			Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

			df_prim_roe[irho] = sqrtdl*sqrtdr;
			df_prim_roe[ivx]  = (sqrtdl*df_prim_li[ivx] + sqrtdr*df_prim_ri[ivx])*isdlpdr;
			df_prim_roe[ivy]  = (sqrtdl*df_prim_li[ivy] + sqrtdr*df_prim_ri[ivy])*isdlpdr;
			df_prim_roe[ivz]  = (sqrtdl*df_prim_li[ivz] + sqrtdr*df_prim_ri[ivz])*isdlpdr;

			//--- Step 3.  Compute L/R fluxes
			Real mxl = df_prim_li[irho]*df_prim_li[ivx];
			Real mxr = df_prim_ri[irho]*df_prim_ri[ivx];

			df_fl[irho] = mxl;
			df_fr[irho] = mxr;

			df_fl[ivx] = mxl*df_prim_li[ivx];
			df_fr[ivx] = mxr*df_prim_ri[ivx];

			df_fl[ivy] = mxl*df_prim_li[ivy];
			df_fr[ivy] = mxr*df_prim_ri[ivy];

			df_fl[ivz] = mxl*df_prim_li[ivz];
			df_fr[ivz] = mxr*df_prim_ri[ivz];

			df_fl[ivx] += (cs*cs)*df_prim_li[irho];
			df_fr[ivx] += (cs*cs)*df_prim_ri[irho];

			//--- Step 4.  Compute Roe fluxes.
			df_du[irho] = df_prim_ri[irho] - df_prim_li[irho];
			df_du[ivx]  = df_prim_ri[irho] * df_prim_ri[ivx] - df_prim_li[irho] * df_prim_li[ivx];
			df_du[ivy]  = df_prim_ri[irho] * df_prim_ri[ivy] - df_prim_li[irho] * df_prim_li[ivy];
			df_du[ivz]  = df_prim_ri[irho] * df_prim_ri[ivz] - df_prim_li[irho] * df_prim_li[ivz];

			df_flxi[irho] = 0.5*(df_fl[irho] + df_fr[irho]);
			df_flxi[ivx]  = 0.5*(df_fl[ivx]  + df_fr[ivx]);
			df_flxi[ivy]  = 0.5*(df_fl[ivy]  + df_fr[ivy]);
			df_flxi[ivz]  = 0.5*(df_fl[ivz]  + df_fr[ivz]);

			int llf_flag = 0;
			RoeFlux(irho, df_prim_roe, df_du, df_prim_li, cs, df_flxi, df_ev, llf_flag);

			//--- Step 5.  Overwrite with upwind flux if flow is supersonic
			if (df_ev[0] >= 0.0) {
				df_flxi[irho] = df_fl[irho];
				df_flxi[ivx]  = df_fl[ivx];
				df_flxi[ivy]  = df_fl[ivy];
				df_flxi[ivz]  = df_fl[ivz];
			}
			if (df_ev[NDUSTVAR-1] <= 0.0) {
				df_flxi[irho] = df_fr[irho];
				df_flxi[ivx]  = df_fr[ivx];
				df_flxi[ivy]  = df_fr[ivy];
				df_flxi[ivz]  = df_fr[ivz];
			}

			////--- Step 6. Overwrite with LLF flux if any of intermediate states are negative
			//if (llf_flag != 0) {
				//Real cl = pmy_block->peos->SoundSpeed(df_prim_li);
				//Real cr = pmy_block->peos->SoundSpeed(df_prim_ri);
				//Real a  = 0.5*std::max( (std::abs(df_prim_li[ivx]) + cl), (std::abs(df_prim_ri[ivx]) + cr) );

				//df_flxi[irho] = 0.5*(df_fl[irho] + df_fr[irho]) - a*df_du[irho];
				//df_flxi[ivx]    = 0.5*(df_fl[ivx]    + df_fr[ivx])    - a*df_du[ivx];
				//df_flxi[ivy]    = 0.5*(df_fl[ivy]    + df_fr[ivy])    - a*df_du[ivy];
				//df_flxi[ivz]    = 0.5*(df_fl[ivz]    + df_fr[ivz])    - a*df_du[ivz];
			//}

			//--- Step 7. Store results into 3D array of fluxes
			dust_flux(irho, k, j, i) = df_flxi[irho];
			dust_flux(ivx,  k, j, i) = df_flxi[ivx];
			dust_flux(ivy,  k, j, i) = df_flxi[ivy];
			dust_flux(ivz,  k, j, i) = df_flxi[ivz];
		}
  }
  return;
}

namespace {
//----------------------------------------------------------------------------------------
//! \fn RoeFlux()
//  \brief Computes Roe fluxes for the conserved variables, that is
//            F[n] = 0.5*(F_l + F_r) - SUM_m(coeff[m]*rem[n][m])
//  where     coeff[n] = 0.5*df_ev[n]*SUM_m(dU[m]*lem[n][m])
//  and the rem[n][m] and lem[n][m] are matrices of the L- and R-eigenvectors of Roe's
//  matrix "A". Also returns the eigenvalues through the argument list.
//
// INPUT:
//   df_prim_roe: vector of Roe averaged primitive variables
//   df_du:       Ur - Ul, difference in L/R-states in conserved variables
//   df_prim_li:  Wl, left state in primitive variables
//   dust_flux:   (F_l + F_r)/2
//
// OUTPUT:
//   dust_flux: final Roe flux
//   df_ev:     vector of eingenvalues
//   llf_flag:  flag set to 1 if d<0 in any intermediate state
//
//  The order of the components in the input vectors should be:
//     (rho_id,ivx,ivy,ivz)
//
// REFERENCES:
// - J. Stone, T. Gardiner, P. Teuben, J. Haprim_df_ley, & J. Simon "Athena: A new code for
//   astrophysical MHD", ApJS, (2008), Appendix A.  Equation numbers refer to this paper.
#pragma omp declare simd simdlen(SIMD_WIDTH) notinbranch
inline void RoeFlux(const int rho_id, const Real df_prim_roe[], const Real df_du[],
                  const Real df_prim_li[], const Real cs,
                  Real dust_flux[], Real df_ev[], int &llf_flag) {
	Real a[(4)];
	Real coeff[(4)];

  int v1_id = rho_id + 1;
  int v2_id = rho_id + 2;
  int v3_id = rho_id + 3;

  Real d  = df_prim_roe[rho_id];
  Real v1 = df_prim_roe[v1_id];
  Real v2 = df_prim_roe[v2_id];
  Real v3 = df_prim_roe[v3_id];

//--- Adiabatic hydrodynamics
  // Compute eigenvalues (eq. B6)
  df_ev[rho_id] = v1 - cs;
  df_ev[v1_id]  = v1;
  df_ev[v2_id]  = v1;
  df_ev[v3_id]  = v1 + cs;

  // Compute projection of dU onto L-eigenvectors using matrix elements from eq. B7
  a[0]  = df_du[rho_id]*(0.5 + 0.5*v1/cs);
  a[0] -= df_du[v1_id]*0.5/cs;

  a[1]  = df_du[rho_id]*(-v2);
  a[1] += df_du[v2_id];

  a[2]  = df_du[rho_id]*(-v3);
  a[2] += df_du[v3_id];

  a[3]  = df_du[rho_id]*(0.5 - 0.5*v1/cs);
  a[3] += df_du[v1_id]*0.5/cs;

  coeff[0] = -0.5 * std::abs(df_ev[rho_id]) * a[0];
  coeff[1] = -0.5 * std::abs(df_ev[v1_id])  * a[1];
  coeff[2] = -0.5 * std::abs(df_ev[v2_id])  * a[2];
  coeff[3] = -0.5 * std::abs(df_ev[v3_id])  * a[3];

  // compute density in intermediate states and check that it is positive, set flag
  // This requires computing the [0][*] components of the right-eigenmatrix
  Real dens = df_prim_li[rho_id] + a[0];  // rem[0][0]=1, so don't bother to compute or store
  if (dens < 0.0) llf_flag = 1;

  dens += a[3];  // rem[0][3]=1, so don't bother to compute or store
  if (dens < 0.0) llf_flag = 1;

  // Now multiply projection with R-eigenvectors from eq. B3 and SUM into output fluxes
  dust_flux[rho_id] += coeff[0];
  dust_flux[rho_id] += coeff[3];

  dust_flux[v1_id] += coeff[0]*(v1 - cs);
  dust_flux[v1_id] += coeff[3]*(v1 + cs);

  dust_flux[v2_id] += coeff[0]*v2;
  dust_flux[v2_id] += coeff[1];
  dust_flux[v2_id] += coeff[3]*v2;

  dust_flux[v3_id] += coeff[0]*v3;
  dust_flux[v3_id] += coeff[2];
  dust_flux[v3_id] += coeff[3]*v3;

  return;
}
} // namespace
