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

class Hydro;
class HydroDiffusion;

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


void DustFluidsDiffusion::ConstantDustDiffusivity(const AthenaArray<Real> &nu_gas,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &stopping_time,
  AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs){
  for (int n=0; n<NDUSTFLUIDS; n++) {
    int &dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
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

void DustFluidsDiffusion::ZeroDustDiffusivity(const AthenaArray<Real> &nu_gas,
  const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
  const AthenaArray<Real> &stopping_time,
  AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs){

  dust_diffusivity.ZeroClear();
  dust_cs.ZeroClear();

  return;
}


void DustFluidsDiffusion::UserDefinedDustDiffusivity(const AthenaArray<Real> &nu_gas,
            const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
            const AthenaArray<Real> &stopping_time,
            AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs){

  Hydro *phyd        = pmb_->phydro;
  HydroDiffusion &hd = phyd->hdif;

  if (hd.nu_alpha == 0.0 && hd.nu_iso == 0.0 && hd.nu_aniso == 0) {
    std::stringstream msg;
    msg << "### FATAL ERROR in the defination of gas viscosity." << std::endl
        << "The viscosity of gas (nu_alpha or nu_iso or nu_aniso) must be set." << std::endl;
    ATHENA_ERROR(msg); // Error if gas is inviscid
    return;
  }

  // Disk problems and Shakura & Sunyaev (1973) viscoisty profile are used.
  bool alpha_disk_model = ((hd.nu_alpha > 0.0) && (std::strcmp(PROBLEM_GENERATOR, "disk") == 0));

  if (alpha_disk_model){
    for (int n=0; n<NDUSTFLUIDS; n++) {
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real eddy_time     = eddy_timescale_r0*pow(rad/r0_,1.5); // The eddy time is Omega_K^-1, scaled with r^1.5
            Real Stokes_number = stopping_time(dust_id,k,j,i)/eddy_time;
            diffusivity        = nu_gas(HydroDiffusion::DiffProcess::alpha,k,j,i)/(1.0 + SQR(Stokes_number)); // Youdin et al. 2007
            Real &soundspeed   = dust_cs(dust_id,k,j,i);
            soundspeed         = std::sqrt(diffusivity/eddy_time);
          }
        }
      }
    }
  }
  else if ((hd.nu_iso > 0.0) && (std::strcmp(PROBLEM_GENERATOR, "disk") == 0)){ // if nu_iso >0 and disk problem used
    for (int n=0; n<NDUSTFLUIDS; n++) {
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real eddy_time     = eddy_timescale_r0*pow(rad/r0_,1.5); // The eddy time is Omega^-1, scaled with r^-1.5
            Real Stokes_number = stopping_time(dust_id,k,j,i)/eddy_time;
            diffusivity        = nu_gas(HydroDiffusion::DiffProcess::iso,k,j,i)/(1.0 + SQR(Stokes_number)); // Youdin et al. 2007
            Real &soundspeed   = dust_cs(dust_id,k,j,i);
            soundspeed         = std::sqrt(diffusivity/eddy_time);
          }
        }
      }
    }
  }
  else if ((hd.nu_iso > 0.0)){
    for (int n=0; n<NDUSTFLUIDS; n++) {
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real &eddy_time    = eddy_timescale_r0; // In other problems, fix the eddy time as constant
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
    for (int n=0; n<NDUSTFLUIDS; n++) {
      int &dust_id = n;
      for (int k=kl; k<=ku; ++k) {
        for (int j=jl; j<=ju; ++j) {
#pragma omp simd
          for (int i=il; i<=iu; ++i) {
            Real rad, phi, z;
            GetCylCoord(pco_, rad, phi, z, i, j, k);
            Real &diffusivity  = dust_diffusivity(dust_id,k,j,i);
            Real &eddy_time    = eddy_timescale_r0; // In other problems, fix the eddy time as constant
            Real Stokes_number = stopping_time(dust_id,k,j,i)/eddy_time;
            diffusivity        = nu_gas(HydroDiffusion::DiffProcess::aniso,k,j,i)/(1.0 + SQR(Stokes_number));
            Real &soundspeed   = dust_cs(dust_id,k,j,i);
            soundspeed         = std::sqrt(diffusivity/eddy_time);
          }
        }
      }
    }
  }
  return;
}
