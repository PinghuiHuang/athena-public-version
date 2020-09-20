//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  \brief Adds source terms due to point mass AT ORIGIN

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

//----------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::PointMass_DustFluids
//  \brief Adds source terms due to point mass AT ORIGIN

void DustFluidsSourceTerms::PointMassDustFluids(const Real dt, const AthenaArray<Real> *flux_df,
                                 const AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df) {
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;
  for (int n=0;n<num_dust_var;n+=4) {
    int dust_id = n/4;
    int rho_id  = 4*dust_id;
    int v1_id   = rho_id + 1;
    int v2_id   = rho_id + 2;
    int v3_id   = rho_id + 3;
    for (int k=pmb->ks; k<=pmb->ke; ++k) {
      for (int j=pmb->js; j<=pmb->je; ++j) {
#pragma omp simd
        for (int i=pmb->is; i<=pmb->ie; ++i) {
          Real den = prim_df(rho_id,k,j,i);
          Real src = dt*den*pmb->pcoord->coord_src1_i_(i)*gm_/pmb->pcoord->x1v(i);
          cons_df(v1_id,k,j,i) -= src;
        }
      }
    }
  }
  return;
}
