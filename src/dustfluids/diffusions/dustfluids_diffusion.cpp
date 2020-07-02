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
#include <string>
#include <cstring>    // strcmp
#include <sstream>

// Athena++ headers
#include "../../defs.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_diffusion.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


DustFluidsDiffusion::DustFluidsDiffusion(DustFluids *pdf, ParameterInput *pin) :
  r0_{pin->GetOrAddReal("problem", "r0", 1.0)},
  eddy_timescale_r0{pin->GetOrAddReal("dust", "eddy_time", 1.0)}, // Omega_K^-1 for disk problem
  Momentum_Diffusion_Flag_{pin->GetOrAddBoolean("dust", "Momentum_Diffusion_Flag", false)},
  dustfluids_diffusion_defined(true),
  pmy_dustfluids_(pdf), pmb_(pmy_dustfluids_->pmy_block), pco_(pmb_->pcoord) {
  const int num_dust_var = 4*NDUSTFLUIDS;
  int nc1 = pmb_->ncells1, nc2 = pmb_->ncells2, nc3 = pmb_->ncells3;
  //int is = pmb_->is; int js = pmb_->js; int ks = pmb_->ks;
  //int ie = pmb_->ie; int je = pmb_->je; int ke = pmb_->ke;
  //int il, iu, jl, ju, kl, ku;
  //jl = js, ju = je, kl = ks, ku = ke;

  //if (pmb_->block_size.nx2 > 1) {
    //if (pmb_->block_size.nx3 == 1) // 2D
      //jl = js-1, ju = je+1, kl = ks, ku = ke;
    //else // 3D
      //jl = js-1, ju = je+1, kl = ks-1, ku = ke+1;
  //}
  //il = is, iu = ie+1;

  if (dustfluids_diffusion_defined) {
    dustfluids_diffusion_flux[X1DIR].NewAthenaArray(num_dust_var, nc3, nc2, nc1+1);
    dustfluids_diffusion_flux[X2DIR].NewAthenaArray(num_dust_var, nc3, nc2+1, nc1);
    dustfluids_diffusion_flux[X3DIR].NewAthenaArray(num_dust_var, nc3+1, nc2, nc1);

    dx1_.NewAthenaArray(nc1);
    dx2_.NewAthenaArray(nc1);
    dx3_.NewAthenaArray(nc1);
    diff_tot_.NewAthenaArray(nc1);

    x1area_.NewAthenaArray(nc1+1);
    x2area_.NewAthenaArray(nc1);
    x3area_.NewAthenaArray(nc1);
    x2area_p1_.NewAthenaArray(nc1);
    x3area_p1_.NewAthenaArray(nc1);
    vol_.NewAthenaArray(nc1);
  }

}


void DustFluidsDiffusion::CalcDustFluidsDiffusionFlux(const AthenaArray<Real> &prim_df,
    const AthenaArray<Real> &cons_df) {
  DustFluids *pdf = pmy_dustfluids_;
  Hydro *phyd     = pmb_->phydro;

  if (dustfluids_diffusion_defined) {
    // Set the diffusion flux of dust fluids as zero
    ClearDustFluidsFlux(dustfluids_diffusion_flux);

    DustFluidsConcentrationDiffusiveFlux(prim_df, phyd->w, dustfluids_diffusion_flux);
    if (Momentum_Diffusion_Flag_)
      DustFluidsMomentumDiffusiveFlux(prim_df, phyd->w, dustfluids_diffusion_flux);
  }
  return;
}


void DustFluidsDiffusion::GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k) {
  if (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) {
    rad = pco->x1v(i);
    phi = pco->x2v(j);
    z   = pco->x3v(k);
  } else if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
    rad = std::abs(pco->x1v(i)*std::sin(pco->x2v(j)));
    phi = pco->x3v(i);
    z   = pco->x1v(i)*std::cos(pco->x2v(j));
  }
  return;
}


void DustFluidsDiffusion::AddDustFluidsDiffusionFlux(AthenaArray<Real> *flux_diff,
                    AthenaArray<Real> *flux_df) {
  // flux_diff: diffusion flux, flux_df: total flux
  int size1 = flux_df[X1DIR].GetSize();
#pragma omp simd
  for (int i=0; i<size1; ++i)
    flux_df[X1DIR](i) += flux_diff[X1DIR](i);

  if (pmb_->pmy_mesh->f2) {
    int size2 = flux_df[X2DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size2; ++i)
      flux_df[X2DIR](i) += flux_diff[X2DIR](i);
  }

  if (pmb_->pmy_mesh->f3) {
    int size3 = flux_df[X3DIR].GetSize();
#pragma omp simd
    for (int i=0; i<size3; ++i)
      flux_df[X3DIR](i) += flux_diff[X3DIR](i);
  }

  return;
}


//----------------------------------------------------------------------------------------
//! \fn void DustFluidsDiffusion::ClearDustFluidsFlux
//  \brief Reset diffusion flux back to zeros

// TODO(felker): completely general operation for any 1x AthenaArray<Real> flux[3]. Move
// from this class.

void DustFluidsDiffusion::ClearDustFluidsFlux(AthenaArray<Real> *flux_diff) {
  flux_diff[X1DIR].ZeroClear();
  flux_diff[X2DIR].ZeroClear();
  flux_diff[X3DIR].ZeroClear();
  return;
}


Real DustFluidsDiffusion::NewDiffusionDt() {
  Real real_max = std::numeric_limits<Real>::max();
  DustFluids *pdf = pmy_dustfluids_;
  const bool f2 = pmb_->pmy_mesh->f2;
  const bool f3 = pmb_->pmy_mesh->f3;
  int il = pmb_->is - NGHOST; int jl = pmb_->js; int kl = pmb_->ks;
  int iu = pmb_->ie + NGHOST; int ju = pmb_->je; int ku = pmb_->ke;
  int dust_id, rho_id, v1_id, v2_id, v3_id;
  Real fac;
  if (f3)
    fac = 1.0/6.0;
  else if (f2)
    fac = 0.25;
  else
    fac = 0.5;

  Real dt_df_diff = real_max;

  AthenaArray<Real> &diff_t = diff_tot_;
  AthenaArray<Real> &len = dx1_, &dx2 = dx2_, &dx3 = dx3_;

  const int num_dust_var = 4*NDUSTFLUIDS;
  for (int n = 0; n<num_dust_var; n+=4){
    dust_id = n/4;
    rho_id  = 4*dust_id;
    v1_id   = rho_id + 1;
    v2_id   = rho_id + 2;
    v3_id   = rho_id + 3;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          diff_t(i) = 0.0;
        }
#pragma omp simd
        for (int i=il; i<=iu; ++i) diff_t(i) += pdf->nu_dustfluids_array(dust_id,k,j,i);
        pco_->CenterWidth1(k, j, il, iu, len);
        pco_->CenterWidth2(k, j, il, iu, dx2);
        pco_->CenterWidth3(k, j, il, iu, dx3);
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          len(i) = (f2) ? std::min(len(i), dx2(i)) : len(i);
          len(i) = (f3) ? std::min(len(i), dx3(i)) : len(i);
        }
        if (dustfluids_diffusion_defined) {
          for (int i=il; i<=iu; ++i)
            dt_df_diff = std::min(dt_df_diff, static_cast<Real>(
                SQR(len(i))*fac/(diff_t(i) + TINY_NUMBER)));
        }
      }
    }
  }
  return dt_df_diff;
}


void DustFluidsDiffusion::User_Defined_StoppingTime(const int kl, const int ku, const int jl, const int ju,
            const int il, const int iu, const AthenaArray<Real> particle_density,
            const AthenaArray<Real> &w, AthenaArray<Real> &stopping_time){
  DustFluids *pdf  = pmy_dustfluids_;
  for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
    int &dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) { //TODO check the index
          Real &st_time  = stopping_time(dust_id,k,j,i);
          const Real &wd = w(IDN,k,j,i);
          st_time        = particle_density(dust_id)/wd;
        }
      }
    }
  }
  return;
}


void DustFluidsDiffusion::User_Defined_DustDiffusivity(const AthenaArray<Real> &nu_gas,
            const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
            const AthenaArray<Real> &stopping_time,
            AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs){

  Hydro *phyd        = pmb_->phydro;
  HydroDiffusion &hd = phyd->hdif;

  if (hd.nu_alpha == 0.0 && hd.nu_iso == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in the defination of gas viscosity." << std::endl
          << "The viscosity of gas (nu_alpha or nu_iso) must be set." << std::endl;
      ATHENA_ERROR(msg); // No viscoisity in gas
      return;
  }

  bool alpha_disk_model = ((hd.nu_alpha > 0.0) && (std::strcmp(PROBLEM_GENERATOR, "disk") == 0));

  if (alpha_disk_model){
    for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) { //TODO check the index
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real eddy_time     = eddy_timescale_r0*pow(rad/r0_,1.5); // The eddy time is Omega^-1, scaled with r^-1.5
            Real Stokes_number = stopping_time(dust_id,k,j,i)/eddy_time;
            diffusivity        = nu_gas(HydroDiffusion::DiffProcess::alpha,k,j,i)/(1.0 + SQR(Stokes_number));
            Real &soundspeed   = dust_cs(dust_id,k,j,i);
            soundspeed         = std::sqrt(diffusivity/eddy_time);
          }
        }
      }
    }
  }
  else if ((hd.nu_iso > 0.0) && (std::strcmp(PROBLEM_GENERATOR, "disk") == 0)){ // if nu_iso >0 and disk problem used
    for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) { //TODO check the index
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real eddy_time     = eddy_timescale_r0*pow(rad/r0_,1.5); // The eddy time is Omega^-1, scaled with r^-1.5
            Real Stokes_number = stopping_time(dust_id,k,j,i)/eddy_time;
            diffusivity        = nu_gas(HydroDiffusion::DiffProcess::iso,k,j,i)/(1.0 + SQR(Stokes_number));
            Real &soundspeed   = dust_cs(dust_id,k,j,i);
            soundspeed         = std::sqrt(diffusivity/eddy_time);
          }
        }
      }
    }
  }
  else {
    for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) { //TODO check the index
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real eddy_time     = eddy_timescale_r0; // In other problems, fixed the eddy time as constant
            Real Stokes_number = stopping_time(dust_id,k,j,i)/eddy_time;
            diffusivity        = nu_gas(HydroDiffusion::DiffProcess::iso,k,j,i)/(1.0 + SQR(Stokes_number));
            Real &soundspeed   = dust_cs(dust_id,k,j,i);
            soundspeed         = std::sqrt(diffusivity/eddy_time);
          }
        }
      }
    }
  }
  return;
}


void DustFluidsDiffusion::ConstStoppingTime(const int kl, const int ku, const int jl, const int ju,
              const int il, const int iu, AthenaArray<Real> &stopping_time){
  for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
    int &dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) { //TODO check the index
          Real rad, phi, z;
          GetCylCoord(pco_, rad, phi, z, i, j, k);
          Real &st_time  = stopping_time(dust_id,k,j,i);
          st_time        = pmy_dustfluids_->const_stopping_time_(dust_id);
        }
      }
    }
  }
  return;
}


void DustFluidsDiffusion::ConstDustDiffusivity(const AthenaArray<Real> &nu_gas,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &stopping_time,
  AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs){
  for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
    int &dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) { //TODO check the index
          Real rad, phi, z;
          GetCylCoord(pco_, rad, phi, z, i, j, k);
          Real &diffusivity = dust_diffusivity(dust_id,k,j,i);
          diffusivity       = pmy_dustfluids_->const_nu_dust_(dust_id);
          Real &soundspeed  = dust_cs(dust_id,k,j,i);
          soundspeed        = std::sqrt(diffusivity/eddy_timescale_r0);
        }
      }
    }
  }
  return;
}
