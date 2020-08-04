//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//  \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C++ headers
#include <algorithm>   // min,max
#include <limits>

// Athena++ headers
#include "../../defs.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif

// Van Leer flux limiter
Real DustFluidsDiffusion::Van_leer_limiter(const Real a, const Real b){
  Real c = a * b;
  return c > 0.0? (2.0*c)/(a+b): 0.0;
}

void DustFluidsDiffusion::DustFluidsMomentumDiffusiveFlux(const AthenaArray<Real> &prim_df,
            const AthenaArray<Real> &w, AthenaArray<Real> *df_diff_flux) {
  DustFluids *pdf           = pmb_->pdustfluids;
  const bool f2             = pmb_->pmy_mesh->f2;
  const bool f3             = pmb_->pmy_mesh->f3;
  //Coordinates *pco                  = pmb_->pcoord;
  const int num_dust_var    = 4*NDUSTFLUIDS;
  // rho_id: concentration diffusive flux. v1_id, v2_id, v3_id: momentum diffusive flux
  AthenaArray<Real> &x1flux = df_diff_flux[X1DIR];
  AthenaArray<Real> &x2flux = df_diff_flux[X2DIR];
  AthenaArray<Real> &x3flux = df_diff_flux[X3DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) {
      if (!f3) // 2D
        jl = js - 1, ju = je + 1, kl = ks, ku = ke;
      else // 3D
        jl = js - 1, ju = je + 1, kl = ks - 1, ku = ke + 1;
    }
  }

  // i-direction loop
  for (int n=0; n<num_dust_var; n+=4) {
    int dust_id = n/4;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          // v_xi * F_rho_x1, The diffusion of the i-momentum in the x1 direction
          // Upwind Scheme depends on the sign of F_rho_x1
          // F_rho_x1 = x1flux(rho_id,k,j,i) is the concentration diffusive flux in the x1 direction
          if (x1flux(rho_id,k,j,i) >= 0.0) {
            x1flux(v1_id,k,j,i)         += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1));
            if (f2) x1flux(v2_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i));
            if (f3) x1flux(v3_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i));
          }
          else {
            x1flux(v1_id,k,j,i)         += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1));
            if (f2) x1flux(v2_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i));
            if (f3) x1flux(v3_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i));
          }

          // v_x1 * F_rho_i, The advection of the i-diffusive flux in x1 direction
          // Upwind Scheme depends on the sign of v_x1
          // We need to interpolate the F_rho_i to cell centers
          if (prim_df(v1_id,k,j,i) >= 0.0) {
            x1flux(v1_id,k,j,i)         += prim_df(v1_id,k,j,i) * Van_leer_limiter(x1flux(rho_id,k,j,i), x1flux(rho_id,k,j,i-1));
            if (f2) x1flux(v1_id,k,j,i) += prim_df(v1_id,k,j,i) * Van_leer_limiter(x2flux(rho_id,k,j,i), x2flux(rho_id,k,j-1,i));
            if (f3) x1flux(v1_id,k,j,i) += prim_df(v1_id,k,j,i) * Van_leer_limiter(x3flux(rho_id,k,j,i), x3flux(rho_id,k-1,j,i));
          }
          else {
            x1flux(v1_id,k,j,i)         += prim_df(v1_id,k,j,i) * Van_leer_limiter(x1flux(rho_id,k,j,i), x1flux(rho_id,k,j,i+1));
            if (f2) x1flux(v1_id,k,j,i) += prim_df(v1_id,k,j,i) * Van_leer_limiter(x2flux(rho_id,k,j,i), x2flux(rho_id,k,j+1,i));
            if (f3) x1flux(v1_id,k,j,i) += prim_df(v1_id,k,j,i) * Van_leer_limiter(x3flux(rho_id,k,j,i), x3flux(rho_id,k+1,j,i));
          }

        }
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (!f3) // 2D
      il = is - 1, iu = ie + 1, kl = ks, ku = ke;
    else // 3D
      il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  }

  if (f2) { // 2D or 3D
    for (int n=0; n<num_dust_var; n+=4) {
      int dust_id = n/4;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=kl; k<=ku; ++k) {
        for (int j=js; j<=je+1; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            // v_xi * F_rho_x2, The diffusion of the i-momentum in the x2 direction
            // Upwind Scheme depends on the sign of F_rho_x2
            // F_rho_x2 = x2flux(rho_id,k,j,i) is the concentration diffusive flux in the x2 direction
            if (x2flux(rho_id,k,j,i) >= 0.0){
              x2flux(v1_id,k,j,i)         += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1));
              x2flux(v2_id,k,j,i)         += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i));
              if (f3) x2flux(v3_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i));
            }
            else {
              x2flux(v1_id,k,j,i)         += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1));
              x2flux(v2_id,k,j,i)         += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i));
              if (f3) x2flux(v3_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i));
            }

            // v_x2 * F_rho_i, The advection of the i-diffusive flux in x2 direction
            // Upwind Scheme depends on the sign of v_x2
            // We need to interpolate the F_rho_i to cell centers
            if (prim_df(v2_id,k,j,i) >= 0.0) {
              x2flux(v2_id,k,j,i)         += prim_df(v2_id,k,j,i) * Van_leer_limiter(x1flux(rho_id,k,j,i), x1flux(rho_id,k,j,i-1));
              x2flux(v2_id,k,j,i)         += prim_df(v2_id,k,j,i) * Van_leer_limiter(x2flux(rho_id,k,j,i), x2flux(rho_id,k,j-1,i));
              if (f3) x2flux(v2_id,k,j,i) += prim_df(v2_id,k,j,i) * Van_leer_limiter(x3flux(rho_id,k,j,i), x3flux(rho_id,k-1,j,i));
            }
            else {
              x2flux(v2_id,k,j,i)         += prim_df(v2_id,k,j,i) * Van_leer_limiter(x1flux(rho_id,k,j,i), x1flux(rho_id,k,j,i+1));
              x2flux(v2_id,k,j,i)         += prim_df(v2_id,k,j,i) * Van_leer_limiter(x2flux(rho_id,k,j,i), x2flux(rho_id,k,j+1,i));
              if (f3) x2flux(v2_id,k,j,i) += prim_df(v2_id,k,j,i) * Van_leer_limiter(x3flux(rho_id,k,j,i), x3flux(rho_id,k+1,j,i));
            }

          }
        }
      }
    }
  } // zero flux for 1D

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  if (MAGNETIC_FIELDS_ENABLED) {
    if (f2) // 2D or 3D
      il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
    else // 1D
      il = is - 1, iu = ie + 1;
  }

  if (f3) { // 3D
    for (int n=0; n<num_dust_var; n+=4) {
      int dust_id = n/4;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=ks; k<=ke+1; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            // v_xi * F_rho_x3, The diffusion of the i-momentum in the x3 direction
            // Upwind Scheme depends on the sign of F_rho_x3
            // F_rho_x3 = x3flux(rho_id,k,j,i) is the concentration diffusive flux in the x3 direction
            if (x3flux(rho_id,k,j,i) >= 0.0) {
              x3flux(v1_id,k,j,i)         += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1));
              if (f2) x3flux(v2_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i));
              x3flux(v3_id,k,j,i)         += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i));
            }
            else {
              x3flux(v1_id,k,j,i)         += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1));
              if (f2) x3flux(v2_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i));
              x3flux(v3_id,k,j,i)         += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i));
            }

            // v_x3 * F_rho_i, The advection of the i-diffusive flux in x3 direction
            // Upwind Scheme depends on the sign of v_x3
            // We need to interpolate the F_rho_i to cell centers
            if (prim_df(v3_id,k,j,i) >= 0.0) {
              x3flux(v3_id,k,j,i)         += prim_df(v3_id,k,j,i) * Van_leer_limiter(x1flux(rho_id,k,j,i), x1flux(rho_id,k,j,i-1));
              if (f2) x3flux(v3_id,k,j,i) += prim_df(v3_id,k,j,i) * Van_leer_limiter(x2flux(rho_id,k,j,i), x2flux(rho_id,k,j-1,i));
              x3flux(v3_id,k,j,i)         += prim_df(v3_id,k,j,i) * Van_leer_limiter(x3flux(rho_id,k,j,i), x3flux(rho_id,k-1,j,i));
            }
            else {
              x3flux(v3_id,k,j,i)         += prim_df(v3_id,k,j,i) * Van_leer_limiter(x1flux(rho_id,k,j,i), x1flux(rho_id,k,j,i+1));
              if (f2) x3flux(v3_id,k,j,i) += prim_df(v3_id,k,j,i) * Van_leer_limiter(x2flux(rho_id,k,j,i), x2flux(rho_id,k,j+1,i));
              x3flux(v3_id,k,j,i)         += prim_df(v3_id,k,j,i) * Van_leer_limiter(x3flux(rho_id,k,j,i), x3flux(rho_id,k+1,j,i));
            }

          }
        }
      }
    }
  } // zero flux for 1D/2D
  return;
}
