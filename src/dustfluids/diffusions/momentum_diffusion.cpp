//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//  \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C headers

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
  DustFluids *pdf = pmb_->pdustfluids;
  const bool f2          = pmb_->pmy_mesh->f2;
  const bool f3          = pmb_->pmy_mesh->f3;
  //Coordinates *pco       = pmb_->pcoord;
  const int num_dust_var = 4*NDUSTFLUIDS;
  AthenaArray<Real> &x1flux = df_diff_flux[X1DIR];
  int il, iu, jl, ju, kl, ku;
  int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  Real nu_i_m, nu_i_p;         // Face center, _m: 0.5*[nu(i-1) + nu(i)], _p: 0.5*[nu(i+1) + nu(i)]
  Real nu_j_m, nu_j_p;         // Face center, _m: 0.5*[nu(j-1) + nu(j)], _p: 0.5*[nu(j+1) + nu(j)]
  Real nu_k_m, nu_k_p;         // Face center, _m: 0.5*[nu(k-1) + nu(k)], _p: 0.5*[nu(k+1) + nu(k)]
  Real rho_i_m, rho_i_p;       // Face center, _m: 0.5*[gas_rho(i-1) + gas_rho(i)], _p: 0.5*[gas_rho(i+1) + gas_rho(i)]
  Real rho_j_m, rho_j_p;       // Face center, _m: 0.5*[gas_rho(j-1) + gas_rho(j)], _p: 0.5*[gas_rho(j+1) + gas_rho(j)]
  Real rho_k_m, rho_k_p;       // Face center, _m: 0.5*[gas_rho(k-1) + gas_rho(k)], _p: 0.5*[gas_rho(k+1) + gas_rho(k)]
  Real df_d11, df_d21, df_d31; // df_dij = ((rho_d/rho_g)[i]-(rho_d/rho_g)[i-1])/dx1v(j-1)
  Real df_d12, df_d22, df_d32;
  Real df_d13, df_d23, df_d33;
  int dust_id, rho_id, v1_id, v2_id, v3_id;
  //Real nu_rhog_d11, nu_rhog_d22, nu_rhog_d33;

  // i-direction
  jl = js, ju = je, kl = ks, ku = ke; // We ignore the changes on index caused by magnetic field

  // i-direction loop
  for (int n=0; n<num_dust_var; n+=4) {
    dust_id = n/4;
    rho_id  = 4*dust_id;
    v1_id   = rho_id + 1;
    v2_id   = rho_id + 2;
    v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=is; i<=ie+1; ++i) {
          // Calculate some parameters on x1 direction
          nu_i_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j,i-1) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          nu_i_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j,i+1) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          rho_i_m = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));
          rho_i_p = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i+1));

          // df_d11 = D(rho_d/rho_g)_x1/D(x1)
          df_d11 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j,i-1)/w(IDN,k,j,i-1))/pco_->dx1v(i-1);


          if (df_d11 > 0) { // Diffusion along three direction, Upwind Scheme depends on df_d11
            x1flux(v1_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1)); //v1 diffusion
            x1flux(v2_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i)); //v2 diffusion
            if (f3)
              x1flux(v3_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i)); //v3 diffusion
          }
          else {
            x1flux(v1_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1)); //v1 diffusion
            x1flux(v2_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i)); //v2 diffusion
            if (f3)
              x1flux(v3_id,k,j,i) += x1flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i)); //v3 diffusion
          }

          if (prim_df(v1_id,k,j,i) > 0) //Upwind Scheme depends on v1
            x1flux(v1_id,k,j,i) -= nu_i_m*rho_i_m*df_d11*0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1));
          else
            x1flux(v1_id,k,j,i) -= nu_i_p*rho_i_p*df_d11*0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1));

          if (f2){ //v2 diffusion if the simulation is 2D, Calculate some parameters on x2 direction
            nu_j_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j-1,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            nu_j_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j+1,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            rho_j_m = 0.5*(w(IDN,k,j,i) + w(IDN,k,j-1,i));
            rho_j_p = 0.5*(w(IDN,k,j,i) + w(IDN,k,j+1,i));
            // df_d21 = D(rho_d/rho_g)_x2/D(x1)
            df_d21 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i)-prim_df(rho_id,k,j-1,i)/w(IDN,k,j-1,i))/pco_->dx1v(i-1);

            if (prim_df(v1_id,k,j,i) > 0) //Upwind Scheme depends on v1
              x1flux(v1_id,k,j,i) -= nu_j_m*rho_j_m*df_d21 * Van_leer_limiter(prim_df(v1_id,k,j,i), prim_df(v1_id,k,j,i-1));
            else
              x1flux(v1_id,k,j,i) -= nu_j_p*rho_j_p*df_d21 * Van_leer_limiter(prim_df(v1_id,k,j,i), prim_df(v1_id,k,j,i+1));
          }

          if (f3){ // v3 diffusion if the simulation is 3D
            // Calculate some parameters on x3 direction
            nu_k_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k-1,j,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            nu_k_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k+1,j,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            rho_k_m = 0.5*(w(IDN,k,j,i) + w(IDN,k-1,j,i));
            rho_k_p = 0.5*(w(IDN,k,j,i) + w(IDN,k+1,j,i));
            // df_d31 = D(rho_d/rho_g)_x3/D(x1)
            df_d31 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k-1,j,i)/w(IDN,k-1,j,i))/pco_->dx1v(i-1);

            if (prim_df(v1_id,k,j,i) > 0) //Upwind Scheme depends on v1
              x1flux(v1_id,k,j,i) -= nu_k_m*rho_k_m*df_d31 *
              Van_leer_limiter(prim_df(v3_id,k,j,i), prim_df(v1_id,k,j,i-1));
            else
              x1flux(v1_id,k,j,i) -= nu_k_p*rho_k_p*df_d31 *
              Van_leer_limiter(prim_df(v1_id,k,j,i), prim_df(v1_id,k,j,i+1));
          }

        }
      }
    }
  }

  // j-direction
  il = is, iu = ie, kl = ks, ku = ke;
  //if (MAGNETIC_FIELDS_ENABLED) {
      //if (!f3)// 2D
        //il = is - 1, iu = ie + 1, kl = ks, ku = ke;
      //else // 3D
        //il = is - 1, iu = ie + 1, kl = ks - 1, ku = ke + 1;
  //}

  if (f2) { // 2D or 3D
  AthenaArray<Real> &x2flux = df_diff_flux[X2DIR];
  for (int n=0; n<num_dust_var; n+=4) {
    dust_id = n/4;
    rho_id  = 4*dust_id;
    v1_id   = rho_id + 1;
    v2_id   = rho_id + 2;
    v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      for (int j=js; j<=je+1; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // Calculate some parameters on x2 direction
          nu_j_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j-1,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          nu_j_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j+1,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          rho_j_m = 0.5*(w(IDN,k,j,i) + w(IDN,k,j-1,i));
          rho_j_p = 0.5*(w(IDN,k,j,i) + w(IDN,k,j+1,i));
           //df_d22 = D(rho_d/rho_g)_x2/D(x2)
          df_d22 = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j-1,i)/w(IDN,k,j-1,i))/pco_->h2v(i)/pco_->dx2v(j-1);

          // Add the diffusion term on the X1 direction flux
          // nu_rhog_d22 = nu_d*rho_g*D(rho_d/rho_g)/D(x2)
          //nu_rhog_d22 = df_d22 > 0 ? nu_j_m*rho_j_m*df_d22 : nu_j_p*rho_j_p*df_d22;
          //nu_rhog_d22 = nu_j_m*rho_j_m*df_d22;
          //x2flux(rho_id,k,j,i) -= nu_rhog_d22;  // rho_d diffusion along x2 direction

          if (df_d22 > 0 ){ // Diffusion along three direction, Upwind Scheme depends on df_d22
            x2flux(v1_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1));
            x2flux(v2_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i));
            if (f3)
              x2flux(v3_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i));
          }
          else {
            x2flux(v1_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1));
            x2flux(v2_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i));
            if (f3)
              x2flux(v3_id,k,j,i) += x2flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i));
          }

          // Calculate some parameters on x1 direction
          nu_i_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j,i-1) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          nu_i_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j,i+1) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          rho_i_m = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));
          rho_i_p = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i+1));
          df_d12  = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j,i-1)/w(IDN,k,j,i-1))/pco_->h2v(i)/pco_->dx2v(j-1);

          if (prim_df(v2_id,k,j,i) > 0) //v2 diffusion, Upwind Scheme depends on v2
            x2flux(v2_id,k,j,i) -= nu_i_m*rho_i_m*df_d12 *
            Van_leer_limiter(prim_df(v2_id,k,j,i), prim_df(v2_id,k,j-1,i));
          else
            x2flux(v2_id,k,j,i) -= nu_i_p*rho_i_p*df_d12 *
            Van_leer_limiter(prim_df(v2_id,k,j,i), prim_df(v2_id,k,j+1,i));

          if (prim_df(v2_id,k,j,i) > 0) //v2 diffusion, Upwind Scheme depends on v2
            x2flux(v2_id,k,j,i) -= nu_i_m*rho_i_m*df_d22 *
            0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i));
          else
            x2flux(v2_id,k,j,i) -= nu_i_p*rho_i_p*df_d22 *
            0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i));

          // v3 diffusion if the simulation is 3D
          if (f3){
            // Calculate some parameters on x3 direction
            nu_k_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k-1,j,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            nu_k_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k+1,j,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            rho_k_m = 0.5*(w(IDN,k,j,i) + w(IDN,k-1,j,i));
            rho_k_p = 0.5*(w(IDN,k,j,i) + w(IDN,k+1,j,i));
            df_d32  = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k-1,j,i)/w(IDN,k-1,j,i))
              /pco_->h2v(i)/pco_->dx2v(j-1);

            if (prim_df(v2_id,k,j,i) > 0) //v2 diffusion, Upwind Scheme depends on v2
              x2flux(v2_id,k,j,i) -= nu_k_m*rho_k_m*df_d32 *
              Van_leer_limiter(prim_df(v2_id,k,j,i), prim_df(v2_id,k,j+1,i));
            else
              x2flux(v2_id,k,j,i) -= nu_k_p*rho_k_p*df_d32 *
              Van_leer_limiter(prim_df(v2_id,k,j,i), prim_df(v2_id,k,j+1,i));
          }

          }
        }
      }
    }
  } // zero flux for 1D

  // k-direction
  il = is, iu = ie, jl = js, ju = je;
  //if (MAGNETIC_FIELDS_ENABLED) {
      //if (f2)// 2D or 3D
        //il = is - 1, iu = ie + 1, jl = js - 1, ju = je + 1;
      //else // 1D
        //il = is - 1, iu = ie + 1;
  //}

  if (f3) { // 3D
  AthenaArray<Real> &x3flux = df_diff_flux[X3DIR];
  for (int n=0; n<num_dust_var; n+=4) {
    dust_id = n/4;
    rho_id  = 4*dust_id;
    v1_id   = rho_id + 1;
    v2_id   = rho_id + 2;
    v3_id   = rho_id + 3;
    for (int k=ks; k<=ke+1; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          // Calculate some parameters on x3 direction
          nu_k_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k-1,j,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          nu_k_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k+1,j,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          rho_k_m = 0.5*(w(IDN,k,j,i) + w(IDN,k-1,j,i));
          rho_k_p = 0.5*(w(IDN,k,j,i) + w(IDN,k+1,j,i));
          df_d33  = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k-1,j,i)/w(IDN,k-1,j,i))/pco_->dx3v(k-1)/pco_->h31v(i)/pco_->h32v(j);

          // Add the diffusion term on the X1 direction flux
          // nu_rhog_d33 = nu_d*rho_g*D(rho_d/rho_g)/D(x3)
          //nu_rhog_d33 = df_d33 > 0 ? nu_k_m*rho_k_m*df_d33 : nu_k_p*rho_k_p*df_d33;
          //nu_rhog_d33 = nu_k_m*rho_k_m*df_d33;
          //x3flux(rho_id,k,j,i) -= nu_rhog_d33; // rho_d diffusion along x3 direction

          if (df_d33 > 0) { // Diffusion along three direction, Upwind Scheme depends on df_d33
            x3flux(v1_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i-1));
            x3flux(v2_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j-1,i));
            x3flux(v3_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i));
          }
          else {
            x3flux(v1_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v1_id,k,j,i) + prim_df(v1_id,k,j,i+1));
            x3flux(v2_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v2_id,k,j,i) + prim_df(v2_id,k,j+1,i));
            x3flux(v3_id,k,j,i) += x3flux(rho_id,k,j,i) * 0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i));
          }

          // Calculate some parameters on x1 direction
          nu_i_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j,i-1) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          nu_i_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j,i+1) + pdf->nu_dustfluids_array(dust_id,k,j,i));
          rho_i_m = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i-1));
          rho_i_p = 0.5*(w(IDN,k,j,i) + w(IDN,k,j,i+1));
          df_d13  = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j,i-1)/w(IDN,k,j,i-1))
            /pco_->dx3v(k-1)/pco_->h31v(i)/pco_->h32v(j);

          if (prim_df(v3_id,k,j,i) > 0) //v3 diffusion, Upwind Scheme depends on v3
            x3flux(v3_id,k,j,i) -= nu_i_m*rho_i_m*df_d13 *
            Van_leer_limiter(prim_df(v3_id,k,j,i), prim_df(v3_id,k-1,j,i));
          else
            x3flux(v3_id,k,j,i) -= nu_i_p*rho_i_p*df_d13 *
            Van_leer_limiter(prim_df(v3_id,k,j,i), prim_df(v3_id,k+1,j,i));

          if (f2){ //v2 diffusion
            // Calculate some parameters on x2 direction
            nu_j_m  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j-1,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            nu_j_p  = 0.5*(pdf->nu_dustfluids_array(dust_id,k,j+1,i) + pdf->nu_dustfluids_array(dust_id,k,j,i));
            rho_j_m = 0.5*(w(IDN,k,j,i) + w(IDN,k,j-1,i));
            rho_j_p = 0.5*(w(IDN,k,j,i) + w(IDN,k,j+1,i));
            df_d23  = (prim_df(rho_id,k,j,i)/w(IDN,k,j,i) - prim_df(rho_id,k,j-1,i)/w(IDN,k,j-1,i))
              /pco_->dx3v(k-1)/pco_->h31v(i)/pco_->h32v(j);

            if (prim_df(v3_id,k,j,i) > 0) //v3 diffusion, Upwind Scheme depends on v3
              x3flux(v3_id,k,j,i) -= nu_j_m*rho_j_m*df_d23 *
              Van_leer_limiter(prim_df(v3_id,k,j,i), prim_df(v3_id,k-1,j,i));
            else
              x3flux(v3_id,k,j,i) -= nu_j_p*rho_j_p*df_d23 *
              Van_leer_limiter(prim_df(v3_id,k,j,i), prim_df(v3_id,k+1,j,i));
          }

          if (prim_df(v3_id,k,j,i) > 0) //v3 diffusion, Upwind Scheme depends on v3
            x3flux(v3_id,k,j,i) -= nu_k_m*rho_k_m*df_d33 *
            0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k-1,j,i));
          else
            x3flux(v3_id,k,j,i) -= nu_k_p*rho_k_p*df_d33 *
            0.5*(prim_df(v3_id,k,j,i) + prim_df(v3_id,k+1,j,i));

          }
        }
      }
    }
  } // zero flux for 1D/2D
  return;
}

