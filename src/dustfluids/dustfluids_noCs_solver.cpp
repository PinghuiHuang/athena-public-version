//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_hlle_solver.cpp
//  \brief spatially isothermal HLLE Riemann solver for dust fludis, no dust sound speed
//
//  Computes 1D df_fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This df_flux is
//  very diffusive, especially for contacts, and so it is not recommended for use in
//  applications.  However, as shown by Einfeldt et al.(1991), it is positively
//  conservative (cannot return negative densities or pressure), so it is a useful
//  option when other approximate solvers fail and/or when extra dissipation is needed.
//
// REFERENCES:
// - E.F. Toro, "Riemann Solvers and numerical methods for df_fluid dynamics", 2nd ed.,
//   Springer-Verlag, Berlin, (1999) chpt. 10.
// - Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
// - A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//   schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "../hydro/hydro.hpp"
#include "dustfluids.hpp"

//----------------------------------------------------------------------------------------
//! \fn void DustFluids::HLLE_RiemannSolver_DustFluids
//  \brief The HLLE Riemann solver for Dust Fluids (no dust sound speed)

void DustFluids::HLLENoCsRiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &df_prim_l,
                          AthenaArray<Real> &df_prim_r, AthenaArray<Real> &dust_flux) {

  Real df_prim_li[(num_dust_var)], df_prim_ri[(num_dust_var)], df_prim_roe[(num_dust_var)];
  Real df_fl[(num_dust_var)],      df_fr[(num_dust_var)],      df_flxi[(num_dust_var)];

  for (int n=0; n<NDUSTFLUIDS; n++) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    int ivx     = (IVX + ((index-IVX))%3)   + rho_id;
    int ivy     = (IVX + ((index-IVX)+1)%3) + rho_id;
    int ivz     = (IVX + ((index-IVX)+2)%3) + rho_id;
#pragma omp simd private(df_prim_li, df_prim_ri, df_prim_roe, df_fl, df_fr, df_flxi)
    for (int i=il; i<=iu; ++i) {
      const Real &nu_d   = nu_dustfluids_array(dust_id, k, j, i);

      df_prim_li[rho_id] = df_prim_l(rho_id, i);
      df_prim_li[ivx]    = df_prim_l(ivx,    i);
      df_prim_li[ivy]    = df_prim_l(ivy,    i);
      df_prim_li[ivz]    = df_prim_l(ivz,    i);

      df_prim_ri[rho_id] = df_prim_r(rho_id, i);
      df_prim_ri[ivx]    = df_prim_r(ivx,    i);
      df_prim_ri[ivy]    = df_prim_r(ivy,    i);
      df_prim_ri[ivz]    = df_prim_r(ivz,    i);

      //Compute middle state estimates with PVRS (Toro 10.5.2)
      //Real al, ar, el, er;
      Real sqrtdl  = std::sqrt(df_prim_li[rho_id]);
      Real sqrtdr  = std::sqrt(df_prim_ri[rho_id]);
      Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

      df_prim_roe[rho_id] = sqrtdl*sqrtdr;
      df_prim_roe[ivx]    = (sqrtdl*df_prim_li[ivx] + sqrtdr*df_prim_ri[ivx])*isdlpdr;
      //Compute the max/min wave speeds based on L/R and Roe-averaged values
      Real al = std::min(df_prim_roe[ivx],df_prim_li[ivx]);
      Real ar = std::max(df_prim_roe[ivx],df_prim_ri[ivx]);

      Real bp = ar > 0.0 ? ar : 0.0;
      Real bm = al < 0.0 ? al : 0.0;

      //Compute L/R df_fluxes along lines bm/bp: F_L - (S_L)U_L; F_R - (S_R)U_R
      Real vxl = df_prim_li[ivx] - bm;
      Real vxr = df_prim_ri[ivx] - bp;

      df_fl[rho_id] = vxl * df_prim_li[rho_id];
      df_fr[rho_id] = vxr * df_prim_ri[rho_id];

      df_fl[ivx]    = df_prim_li[ivx] * df_fl[rho_id];
      df_fr[ivx]    = df_prim_ri[ivx] * df_fr[rho_id];

      df_fl[ivy]    = df_prim_li[ivy] * df_fl[rho_id];
      df_fr[ivy]    = df_prim_ri[ivy] * df_fr[rho_id];

      df_fl[ivz]    = df_prim_li[ivz] * df_fl[rho_id];
      df_fr[ivz]    = df_prim_ri[ivz] * df_fr[rho_id];

      //Compute the HLLE df_flux at interface.
      Real tmp  = 0.0;
      if (bp != bm) tmp = 0.5*(bp + bm)/(bp - bm);

      df_flxi[rho_id] = 0.5*(df_fl[rho_id] + df_fr[rho_id]) + (df_fl[rho_id] - df_fr[rho_id])*tmp;
      df_flxi[ivx]    = 0.5*(df_fl[ivx]+df_fr[ivx]) + (df_fl[ivx]-df_fr[ivx])*tmp;
      df_flxi[ivy]    = 0.5*(df_fl[ivy]+df_fr[ivy]) + (df_fl[ivy]-df_fr[ivy])*tmp;
      df_flxi[ivz]    = 0.5*(df_fl[ivz]+df_fr[ivz]) + (df_fl[ivz]-df_fr[ivz])*tmp;

      dust_flux(rho_id,k,j,i) = df_flxi[rho_id];
      dust_flux(ivx,k,j,i)    = df_flxi[ivx];
      dust_flux(ivy,k,j,i)    = df_flxi[ivy];
      dust_flux(ivz,k,j,i)    = df_flxi[ivz];
    }
  }

  return;
}
