//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_dustfluids.cpp
//  \brief implements boundary functions for DustFluids variables and utilities to manage
//  primitive/conservative variable relationship in a derived class of the
//  CellCenteredBoundaryVariable base class.

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../dustfluids/dustfluids.hpp"
#include "../../../mesh/mesh.hpp"
#include "bvals_dustfluids.hpp"

//----------------------------------------------------------------------------------------
//! \class DustFluidsBoundaryFunctions

DustFluidsBoundaryVariable::DustFluidsBoundaryVariable(
    MeshBlock *pmb, AthenaArray<Real> *var_dustfluids, AthenaArray<Real> *coarse_var,
    AthenaArray<Real> *var_flux,
    DustFluidsBoundaryQuantity dustfluids_type) :
    CellCenteredBoundaryVariable(pmb, var_dustfluids, coarse_var, var_flux),
    dustfluids_type_(dustfluids_type) {
  flip_across_pole_ = flip_across_pole_dustfluids;
}

//----------------------------------------------------------------------------------------
//! \fn void DustFluidsBoundaryVariable::SelectCoarseBuffer(DustFluidsBoundaryQuantity type)
//  \brief

void DustFluidsBoundaryVariable::SelectCoarseBuffer(DustFluidsBoundaryQuantity dustfluids_type) {
  if (pmy_mesh_->multilevel) {
    switch (dustfluids_type) {
      case (DustFluidsBoundaryQuantity::cons_df): {
        coarse_buf = &(pmy_block_->pdustfluids->coarse_df_cons_);
        break;
      }
      case (DustFluidsBoundaryQuantity::prim_df): {
        coarse_buf = &(pmy_block_->pdustfluids->coarse_df_prim_);
        break;
      }
    }
  }
  dustfluids_type_ = dustfluids_type;
  return;
}

// TODO(felker): make general (but restricted) setter fns in CellCentered and FaceCentered
void DustFluidsBoundaryVariable::SwapDustFluidsQuantity(AthenaArray<Real> &var_dustfluids,
                                              DustFluidsBoundaryQuantity dustfluids_type) {
  var_cc = &var_dustfluids;
  SelectCoarseBuffer(dustfluids_type);
  return;
}
