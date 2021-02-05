//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids.cpp
//! \brief implementation of functions in class DustFluids

// C headers

// C++ headers
#include <algorithm>
#include <string>
#include <vector>
#include <cstring>    // strcmp
#include <sstream>
#include <stdexcept>  // runtime_error

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../coordinates/coordinates.hpp"
#include "../eos/eos.hpp"
#include "../mesh/mesh.hpp"
#include "../reconstruct/reconstruction.hpp"
#include "dustfluids.hpp"
#include "diffusions/dustfluids_diffusion.hpp"
#include "drags/dust_gas_drag.hpp"
#include "srcterms/dustfluids_srcterms.hpp"

class DustFluidsDiffusion;
class DustGasDrag;
class DustFluidsSourceTerms;

//! constructor, initializes data structures and parameters
DustFluids::DustFluids(MeshBlock *pmb, ParameterInput *pin)  :
  pmy_block(pmb), pco_(pmb->pcoord),
  df_cons(NDUSTVAR,    pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_cons1(NDUSTVAR,   pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_cons_bs(NDUSTVAR, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_cons_as(NDUSTVAR, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_prim(NDUSTVAR,    pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_prim1(NDUSTVAR,   pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_prim_n(NDUSTVAR,  pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_flux{{NDUSTVAR,   pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
            {NDUSTVAR, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
            (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
            {NDUSTVAR, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
            (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}},
  coarse_df_cons_(NDUSTVAR, pmb->ncc3, pmb->ncc2, pmb->ncc1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  coarse_df_prim_(NDUSTVAR, pmb->ncc3, pmb->ncc2, pmb->ncc1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  stopping_time_array(NDUSTFLUIDS,   pmb->ncells3, pmb->ncells2, pmb->ncells1),
  stopping_time_array_n(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  nu_dustfluids_array(NDUSTFLUIDS,   pmb->ncells3, pmb->ncells2, pmb->ncells1),
  nu_dustfluids_array_n(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  cs_dustfluids_array(NDUSTFLUIDS,   pmb->ncells3, pmb->ncells2, pmb->ncells1),
  cs_dustfluids_array_n(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  dfbvar(pmb, &df_cons, &coarse_df_cons_, df_flux, DustFluidsBoundaryQuantity::cons_df),
  internal_density(NDUSTFLUIDS),    // normalized dust internal density, used in user defined stopping time
  const_stopping_time(NDUSTFLUIDS), // const stopping time, used in constant stopping time
  const_nu_dust(NDUSTFLUIDS),       // const dust diffusivity, used in constant diffusivity
  dfdrag(this, pin),
  dfdif(this,  pin),
  dfsrc(this,  pin) {

  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;

  Mesh *pm = pmy_block->pmy_mesh;
  pmb->RegisterMeshBlockData(df_cons);

  ConstStoppingTime_Flag = pin->GetBoolean("dust",      "Const_StoppingTime_Flag");
  SoundSpeed_Flag        = pin->GetOrAddBoolean("dust", "Dust_SoundSpeed_Flag", false);

  // If dust is inviscid, then sound speed flag is set as false
  if (!(dfdif.Diffusion_Flag))
    SoundSpeed_Flag = false;

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    // read the dust internal density, stopping time, nu_dust
    if (ConstStoppingTime_Flag)
      const_stopping_time(n) = pin->GetReal("dust", "stopping_time_" + std::to_string(n+1));
    else
      internal_density(n) = pin->GetReal("dust", "internal_density_" + std::to_string(n+1));

    if (dfdif.dustfluids_diffusion_defined) {
      if (dfdif.ConstNu_Flag)
        const_nu_dust(n) = pin->GetReal("dust", "nu_dust_" + std::to_string(n+1));
    }
  }

  // Allocate optional dustfluids variable memory registers for time-integrator
  if (pmb->precon->xorder == 4) {
    // fourth-order cell-centered approximations
    df_cons_cc.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);
    df_prim_cc.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);
  }

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");

  if (integrator == "ssprk5_4" || STS_ENABLED)
    // future extension may add "int nregister" to Hydro class
    df_cons2.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);

  // If STS RKL2, allocate additional memory registers
  if (STS_ENABLED) {
    std::string sts_integrator = pin->GetOrAddString("time", "sts_integrator", "rkl1");
    if (sts_integrator == "rkl2") {
      df_cons0.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);
      df_cons_fl_div.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);
    }
  }

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&df_cons, &coarse_df_cons_);
  }

  // enroll DustFluidsBoundaryVariable object
  dfbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&dfbvar);
  pmb->pbval->bvars_main_int.push_back(&dfbvar);

  if (STS_ENABLED) {
    if (dfdif.dustfluids_diffusion_defined) {
      pmb->pbval->bvars_sts.push_back(&dfbvar);
    }
  }


  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  //dx_df_prim_.NewAthenaArray(nc1);
  df_prim_l_.NewAthenaArray(NDUSTVAR,  nc1);
  df_prim_r_.NewAthenaArray(NDUSTVAR,  nc1);
  df_prim_lb_.NewAthenaArray(NDUSTVAR, nc1);
  x1face_area_.NewAthenaArray(nc1+1);

  if (pm->f2) {
    x2face_area_.NewAthenaArray(nc1);
    x2face_area_p1_.NewAthenaArray(nc1);
  }
  if (pm->f3) {
    x3face_area_.NewAthenaArray(nc1);
    x3face_area_p1_.NewAthenaArray(nc1);
  }

  cell_volume_.NewAthenaArray(nc1);
  dflx_.NewAthenaArray(NDUSTVAR, nc1);

  // fourth-order integration scheme
  if (pmb->precon->xorder == 4) {
    // 4D scratch arrays
    df_prim_l3d_.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);
    df_prim_r3d_.NewAthenaArray(NDUSTVAR, nc3, nc2, nc1);
    scr1_nkji_.NewAthenaArray(NDUSTVAR,   nc3, nc2, nc1);
    scr2_nkji_.NewAthenaArray(NDUSTVAR,   nc3, nc2, nc1);
    // store all face-centered mass fluxes (all 3x coordinate directions) from Hydro:

    // 1D scratch arrays
    laplacian_l_df_fc_.NewAthenaArray(nc1);
    laplacian_r_df_fc_.NewAthenaArray(nc1);
  }
}


void DustFluids::ConstantStoppingTime(const int kl, const int ku, const int jl, const int ju,
              const int il, const int iu, AthenaArray<Real> &stopping_time){
  for (int n=0; n<NDUSTFLUIDS; ++n) { // Calculate the stopping time array and the dust diffusivity array
    int dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &st_time = stopping_time(dust_id, k, j, i);
          st_time       = const_stopping_time(dust_id);
        }
      }
    }
  }
  return;
}


void DustFluids::UserDefinedStoppingTime(const int kl, const int ku, const int jl, const int ju,
            const int il, const int iu, const AthenaArray<Real> internal_density,
            const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time){
  //Real rad, phi, z;
  //Real sqrt_gm0 = std::sqrt(dfsrc.gm_);
  Real inv_Omega = 1.0/dfsrc.Omega_0_;

  for (int n=0; n<NDUSTFLUIDS; ++n) {
    int dust_id = n;
    int rho_id  = 4*dust_id;
    //Real inv_internal = 1.0/internal_density(dust_id);
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) {
          Real &st_time = stopping_time(dust_id, k, j, i);

          // The Stopping time is inversely proportional to the density of gas
          //const Real &gas_rho = w(IDN, k, j, i);
          //st_time = internal_density(dust_id)/gas_rho;

          // Dusty Shock Test
          //const Real &dust_rho = prim_df(rho_id, k, j, i);
          //st_time = dust_rho*inv_internal;

          // Constant Stokes number in disk problems
          //if ( (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0) ||
                //std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0) {
            //dfdif.GetCylCoord(pco_, rad, phi, z, i, j, k);
            //st_time = internal_density(dust_id)*std::pow(rad, 1.5)*sqrt_gm0;
          //}

          // NSH equilibrium Test && Streaming Instability Test
          st_time = internal_density(dust_id)*inv_Omega;
        }
      }
    }
  }
  return;
}


void DustFluids::SetDustFluidsProperties(const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust) {
  int il = pmy_block->is - NGHOST; int jl = pmy_block->js; int kl = pmy_block->ks;
  int iu = pmy_block->ie + NGHOST; int ju = pmy_block->je; int ku = pmy_block->ke;
  Hydro          *phyd = pmy_block->phydro;
  HydroDiffusion &hd   = phyd->hdif;

  if (pmy_block->block_size.nx2 > 1) {
    jl -= NGHOST; ju += NGHOST;
  }

  if (pmy_block->block_size.nx3 > 1) {
    kl -= NGHOST; ku += NGHOST;
  }

  if ( ConstStoppingTime_Flag )
    ConstantStoppingTime(kl, ku, jl, ju, il, iu, stopping_time);
  else
    UserDefinedStoppingTime(kl, ku, jl, ju, il, iu, internal_density,
        w, prim_df, stopping_time);

  if ( dfdif.dustfluids_diffusion_defined ) {
    if ( dfdif.ConstNu_Flag )
      dfdif.ConstantDustDiffusivity(hd.nu, kl, ku, jl, ju, il, iu,
        stopping_time, nu_dust, cs_dust);
    else
      dfdif.UserDefinedDustDiffusivity(hd.nu, kl, ku, jl, ju, il, iu,
        stopping_time, nu_dust, cs_dust);
  }
  else { // If the dust fluids are inviscid, then set the diffusivities and sound speed as zeros.
    dfdif.ZeroDustDiffusivity(hd.nu, kl, ku, jl, ju, il, iu,
      stopping_time, nu_dust, cs_dust);
  }

  return;
}
