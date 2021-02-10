//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_noCs_solver.cpp
//! \brief HLLE Riemann solver for dust fludis (no dust sound speed)
//!
//! Computes 1D fluxes using the Harten-Lax-van Leer (HLL) Riemann solver.  This flux is
//! very diffusive, especially for contacts, and so it is not recommended for use in
//! applications.  However, as shown by Einfeldt et al.(1991), it is positively
//! conservative (cannot return negative densities or pressure), so it is a useful
//! option when other approximate solvers fail and/or when extra dissipation is needed.
//!
//!REFERENCES:
//!- E.F. Toro, "Riemann Solvers and numerical methods for fluid dynamics", 2nd ed.,
//!  Springer-Verlag, Berlin, (1999) chpt. 10.
//!- Einfeldt et al., "On Godunov-type methods near low densities", JCP, 92, 273 (1991)
//!- A. Harten, P. D. Lax and B. van Leer, "On upstream differencing and Godunov-type
//!  schemes for hyperbolic conservation laws", SIAM Review 25, 35-61 (1983).

// C headers

// C++ headers
#include <algorithm>  // max(), min()
#include <cmath>      // sqrt()

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../eos/eos.hpp"
#include "dustfluids.hpp"

//----------------------------------------------------------------------------------------
//! \fn void DustFluids::RiemannSolver_DustFluids
//! \brief The HLLE Riemann solver for Dust Fluids (no dust sound speed)

void DustFluids::RiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
                          const int index, AthenaArray<Real> &prim_df_l,
                          AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux) {

  Real df_prim_li[(NDUSTVARS)], df_prim_ri[(NDUSTVARS)], df_prim_roe[(NDUSTVARS)];

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int idust = n;
    int irho  = 4*idust;
    int ivx   = (IVX + ((index-IVX))%3)   + irho;
    int ivy   = (IVX + ((index-IVX)+1)%3) + irho;
    int ivz   = (IVX + ((index-IVX)+2)%3) + irho;
#pragma omp simd private(df_prim_li, df_prim_ri, df_prim_roe)
    for (int i=il; i<=iu; i++) {
      df_prim_li[irho] = prim_df_l(irho, i);
      df_prim_li[ivx]  = prim_df_l(ivx,  i);
      df_prim_li[ivy]  = prim_df_l(ivy,  i);
      df_prim_li[ivz]  = prim_df_l(ivz,  i);

      df_prim_ri[irho] = prim_df_r(irho, i);
      df_prim_ri[ivx]  = prim_df_r(ivx,  i);
      df_prim_ri[ivy]  = prim_df_r(ivy,  i);
      df_prim_ri[ivz]  = prim_df_r(ivz,  i);

      Real sqrtdl  = std::sqrt(df_prim_li[irho]);
      Real sqrtdr  = std::sqrt(df_prim_ri[irho]);
      Real isdlpdr = 1.0/(sqrtdl + sqrtdr);

      df_prim_roe[irho] = sqrtdl*sqrtdr;
      df_prim_roe[ivx]  = (sqrtdl*df_prim_li[ivx] + sqrtdr*df_prim_ri[ivx])*isdlpdr;

      Real negative_li = df_prim_li[ivx] < 0.0 ? 1.0 : 0.0;
      Real positive_ri = df_prim_ri[ivx] > 0.0 ? 1.0 : 0.0;
      Real temp        = 1.0 - negative_li * positive_ri;

      Real fra_roe = (df_prim_roe[ivx] > 0.0) ? 1.0 : 0.0;
      if (df_prim_roe[ivx] == 0.0) fra_roe = 0.5;

      dust_flux(irho, k, j, i) = temp*(fra_roe*df_prim_li[irho]*df_prim_li[ivx] +
                                 (1.0-fra_roe)*df_prim_ri[irho]*df_prim_ri[ivx]);

      dust_flux(ivx, k, j, i)  = temp*(fra_roe*df_prim_li[ivx]*dust_flux(irho, k, j, i) +
                                 (1.0-fra_roe)*df_prim_ri[ivx]*dust_flux(irho, k, j, i));

      dust_flux(ivy, k, j, i)  = temp*(fra_roe*df_prim_li[ivy]*dust_flux(irho, k, j, i) +
                                 (1.0-fra_roe)*df_prim_ri[ivy]*dust_flux(irho, k, j, i));

      dust_flux(ivz, k, j, i)  = temp*(fra_roe*df_prim_li[ivz]*dust_flux(irho, k, j, i) +
                                 (1.0-fra_roe)*df_prim_ri[ivz]*dust_flux(irho, k, j, i));
    }
  }
  return;
}
