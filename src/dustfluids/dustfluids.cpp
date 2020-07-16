//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids.cpp
//  \brief implementation of functions in class DustFluids

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

// constructor, initializes data structures and parameters
DustFluids::DustFluids(MeshBlock *pmb, ParameterInput *pin)  :
  pmy_block(pmb), pco_(pmb->pcoord),
  df_cons(4*NDUSTFLUIDS,  pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_cons1(4*NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_prim(4*NDUSTFLUIDS,  pmb->ncells3, pmb->ncells2, pmb->ncells1),
  df_flux{{4*NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1+1},
            {4*NDUSTFLUIDS, pmb->ncells3, pmb->ncells2+1, pmb->ncells1,
            (pmb->pmy_mesh->f2 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)},
            {4*NDUSTFLUIDS, pmb->ncells3+1, pmb->ncells2, pmb->ncells1,
            (pmb->pmy_mesh->f3 ? AthenaArray<Real>::DataStatus::allocated :
            AthenaArray<Real>::DataStatus::empty)}},
  coarse_df_cons_(4*NDUSTFLUIDS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  coarse_df_prim_(4*NDUSTFLUIDS, pmb->ncc3, pmb->ncc2, pmb->ncc1,
            (pmb->pmy_mesh->multilevel ? AthenaArray<Real>::DataStatus::allocated :
             AthenaArray<Real>::DataStatus::empty)),
  stopping_time_array(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  nu_dustfluids_array(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  cs_dustfluids_array(NDUSTFLUIDS, pmb->ncells3, pmb->ncells2, pmb->ncells1),
  dfbvar(pmb, &df_cons, &coarse_df_cons_, df_flux),
  particle_density_(NDUSTFLUIDS),    // normalized particle internal density, used in user defined stopping time
  const_stopping_time_(NDUSTFLUIDS), // const stopping time, used in constant stopping time
  const_nu_dust_(NDUSTFLUIDS),       // const dust diffusivity, used in constant diffusivity
  dfdrag(this, pin),
  dfdif(this,  pin),
  dfsrc(this,  pin) {
  const int num_dust_var = 4*NDUSTFLUIDS;
  int nc1 = pmb->ncells1, nc2 = pmb->ncells2, nc3 = pmb->ncells3;
  Mesh *pm = pmy_block->pmy_mesh;
  pmb->RegisterMeshBlockData(df_cons);

  ConstStoppingTime_Flag = pin->GetBoolean("dust", "Const_StoppingTime_Flag");
  SoundSpeed_Flag        = pin->GetOrAddBoolean("dust", "Dust_SoundSpeed_Flag", false);

  for (int n=0; n<NDUSTFLUIDS; n++){
    // read the dust internal density, stopping time, nu_dust
    std::string particle_string = "particle_density_";
    std::string st_time_string  = "stopping_time_";
    std::string nu_string       = "nu_dust_";

    if (ConstStoppingTime_Flag)
      const_stopping_time_(n) = pin->GetReal("dust", st_time_string  + std::to_string(n+1));
    else
      particle_density_(n) = pin->GetReal("dust", particle_string + std::to_string(n+1));

    if (dfdif.dustfluids_diffusion_defined) {
      if (dfdif.ConstNu_Flag)
        const_nu_dust_(n) = pin->GetReal("dust", nu_string + std::to_string(n+1));
    }
  }

  // Allocate optional dustfluids variable memory registers for time-integrator
  if (pmb->precon->xorder == 4) {
    // fourth-order cell-centered approximations
    df_cons_cc.NewAthenaArray(num_dust_var, nc3, nc2, nc1);
    df_prim_cc.NewAthenaArray(num_dust_var, nc3, nc2, nc1);
  }

  // If user-requested time integrator is type 3S*, allocate additional memory registers
  std::string integrator = pin->GetOrAddString("time", "integrator", "vl2");
  if (integrator == "ssprk5_4" || STS_ENABLED)
    // future extension may add "int nregister" to Hydro class
    df_cons2.NewAthenaArray(num_dust_var, nc3, nc2, nc1);

  // "Enroll" in SMR/AMR by adding to vector of pointers in MeshRefinement class
  if (pm->multilevel) {
    refinement_idx = pmy_block->pmr->AddToRefinement(&df_cons, &coarse_df_cons_);
  }

  // enroll CellCenteredBoundaryVariable object
  dfbvar.bvar_index = pmb->pbval->bvars.size();
  pmb->pbval->bvars.push_back(&dfbvar);
  pmb->pbval->bvars_main_int.push_back(&dfbvar);

  // Allocate memory for scratch arrays
  dt1_.NewAthenaArray(nc1);
  dt2_.NewAthenaArray(nc1);
  dt3_.NewAthenaArray(nc1);
  //dx_df_prim_.NewAthenaArray(nc1);
  df_prim_l_.NewAthenaArray(num_dust_var,  nc1);
  df_prim_r_.NewAthenaArray(num_dust_var,  nc1);
  df_prim_lb_.NewAthenaArray(num_dust_var, nc1);
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
  dflx_.NewAthenaArray(num_dust_var, nc1);

  // fourth-order integration scheme
  if (pmb->precon->xorder == 4) {
    // 4D scratch arrays
    df_prim_l3d_.NewAthenaArray(num_dust_var, nc3, nc2, nc1);
    df_prim_r3d_.NewAthenaArray(num_dust_var, nc3, nc2, nc1);
    scr1_nkji_.NewAthenaArray(num_dust_var,   nc3, nc2, nc1);
    scr2_nkji_.NewAthenaArray(num_dust_var,   nc3, nc2, nc1);
    // store all face-centered mass fluxes (all 3x coordinate directions) from Hydro:

    // 1D scratch arrays
    laplacian_l_df_fc_.NewAthenaArray(nc1);
    laplacian_r_df_fc_.NewAthenaArray(nc1);
  }
}

void DustFluids::Constant_StoppingTime(const int kl, const int ku, const int jl, const int ju,
              const int il, const int iu, AthenaArray<Real> &stopping_time){
  for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
    int &dust_id = n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) { //TODO check the index
          Real rad, phi, z;
          dfdif.GetCylCoord(pco_, rad, phi, z, i, j, k);
          Real &st_time  = stopping_time(dust_id,k,j,i);
          st_time        = const_stopping_time_(dust_id);
        }
      }
    }
  }
  return;
}

void DustFluids::UserDefined_StoppingTime(const int kl, const int ku, const int jl, const int ju,
            const int il, const int iu, const AthenaArray<Real> particle_density,
            const AthenaArray<Real> &w, AthenaArray<Real> &stopping_time){
  for (int n=0; n<NDUSTFLUIDS; n++) { // Calculate the stopping time array and the dust diffusivity array
    int &dust_id = n;
    int rho_id   = 4*n;
    for (int k=kl; k<=ku; ++k) {
      for (int j=jl; j<=ju; ++j) {
#pragma omp simd
        for (int i=il; i<=iu; ++i) { //TODO check the index
          Real &dust_den = df_prim(rho_id,k,j,i);
          Real &st_time  = stopping_time(dust_id,k,j,i);
          const Real &wd = w(IDN,k,j,i);
          //// The stopping time is in inversely proportion to the density of gas, see Takeuchi & Lin, 2001
          st_time        = particle_density(dust_id)/wd;
          //st_time = dust_den/particle_density(dust_id);
        }
      }
    }
  }
  return;
}


void DustFluids::SetDustFluidsProperties(){
  int il = pmy_block->is - NGHOST; int jl = pmy_block->js; int kl = pmy_block->ks;
  int iu = pmy_block->ie + NGHOST; int ju = pmy_block->je; int ku = pmy_block->ke;
  Hydro *phyd        = pmy_block->phydro;
  HydroDiffusion &hd = phyd->hdif;

  if (pmy_block->block_size.nx2 > 1) {
    jl -= NGHOST; ju += NGHOST;
  }

  if (pmy_block->block_size.nx3 > 1) {
    kl -= NGHOST; ku += NGHOST;
  }

  if ( ConstStoppingTime_Flag )
    Constant_StoppingTime(kl, ku, jl, ju, il, iu, stopping_time_array);
  else
    UserDefined_StoppingTime(kl, ku, jl, ju, il, iu, particle_density_,
        phyd->w, stopping_time_array);

  if ( dfdif.dustfluids_diffusion_defined ) {
    if ( dfdif.ConstNu_Flag )
      dfdif.Constant_DustDiffusivity(hd.nu, kl, ku, jl, ju, il, iu,
        stopping_time_array, nu_dustfluids_array, cs_dustfluids_array);
    else
      dfdif.UserDefined_DustDiffusivity(hd.nu, kl, ku, jl, ju, il, iu,
        stopping_time_array, nu_dustfluids_array, cs_dustfluids_array);
  }

  return;
}
