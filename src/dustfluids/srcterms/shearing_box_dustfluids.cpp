//======================================================================================
// Athena++ astrophysical MHD code
// Copyright (C) 2014 James M. Stone  <jmstone@princeton.edu>
//
// This program is free software: you can redistribute and/or modify it under the terms
// of the GNU General Public License (GPL) as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
// PARTICULAR PURPOSE.  See the GNU General Public License for more details.
//
// You should have received a copy of GNU GPL in the file LICENSE included in the code
//======================================================================================
//! \file shearing_box.cpp
//  \brief Adds source terms due to local shearing box approximation
//======================================================================================

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_srcterms.hpp"

class DustFluids;
class ParameterInput;

//--------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::ShearingBoxSourceTerms_DustFluids(const Real dt,
//  const AthenaArray<Real> *df_flux, const AthenaArray<Real> &df_prim, AthenaArray<Real>
//  &df_cons)
//  \brief Shearing Box source terms
//
//  Detailed description starts here.
//  We add shearing box source term via operator splitting method. The source terms are
//  added after the fluxes are computed in each step of the integration (in
//  FluxDivergence) to give predictions of the conservative variables for either the
//  next step or the final update.

void DustFluidsSourceTerms::ShearingBoxSourceTermsDustFluids(const Real dt,
            const AthenaArray<Real> *flux_df, const AthenaArray<Real> &prim_df,
                                              AthenaArray<Real> &cons_df) {
  if (Omega_0_==0.0 || qshear_==0.0 ) {
    std::cout << "[ShearingBoxSourceTerms]: Omega_0 or qshear not stated " << std::endl;
    return;
  }
  MeshBlock *pmb  = pmy_dustfluids_->pmy_block;

  // 1) S_M = -rho*grad(Phi); S_E = -rho*v*grad(Phi)
  //    dM1/dt = 2q\rho\Omega^2 x
  //    dE /dt = 2q\Omega^2 (\rho v_x)
  // 2) Coriolis forces:
  //    dM1/dt = 2\Omega(\rho v_y)
  //    dM2/dt = -2\Omega(\rho v_x)
  if (pmb->block_size.nx3 > 1 || ShBoxCoord_ == 1) {
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int k=pmb->ks; k<=pmb->ke; ++k) {
        for (int j=pmb->js; j<=pmb->je; ++j) {
          for (int i=pmb->is; i<=pmb->ie; ++i) {
            const Real &rho_dust = prim_df(rho_id, k, j, i);
            const Real &v1_dust  = prim_df(v1_id,  k, j, i);
            const Real &v2_dust  = prim_df(v2_id,  k, j, i);
            const Real &v3_dust  = prim_df(v3_id,  k, j, i);
            cons_df(v1_id,k,j,i) += dt*(2.0*qshear_*Omega_0_*Omega_0_*rho_dust*pmb->pcoord->x1v(i)
                                   +2.0*Omega_0_*rho_dust*v2_dust);
            cons_df(v2_id,k,j,i) -= dt*2.0*Omega_0_*rho_dust*v1_dust;
          }
        }
      }
    }
  } else if (pmb->block_size.nx3 == 1 && ShBoxCoord_ == 2) {
    int ks = pmb->ks;
    for (int n=0; n<NDUSTFLUIDS; ++n) {
      int dust_id = n;
      int rho_id  = 4*dust_id;
      int v1_id   = rho_id + 1;
      int v2_id   = rho_id + 2;
      int v3_id   = rho_id + 3;
      for (int j=pmb->js; j<=pmb->je; ++j) {
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          const Real &rho_dust = prim_df(rho_id, ks, j, i);
          const Real &v1_dust  = prim_df(v1_id,  ks, j, i);
          const Real &v2_dust  = prim_df(v2_id,  ks, j, i);
          const Real &v3_dust  = prim_df(v3_id,  ks, j, i);
          cons_df(v1_id,ks,j,i) += dt*(2.0*qshear_*Omega_0_*Omega_0_*rho_dust*pmb->pcoord->x1v(i)
                                   +2.0*Omega_0_*rho_dust*v3_dust);
          cons_df(v3_id,ks,j,i) -= dt*2.0*Omega_0_*rho_dust*v1_dust;
        }
      }
    }
  } else {
    std::cout << "[ShearingBoxSourceTerms]: not compatible to 1D !!" << std::endl;
    return;
  }

  return;
}
