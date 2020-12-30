//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_shear_dustfluids.cpp
//  \brief functions that apply shearing box BCs for dustfluids variables
//========================================================================================

// C headers

// C++ headers
#include <algorithm>  // min
#include <cmath>
#include <cstdlib>
#include <cstring>    // memcpy
#include <iomanip>
#include <iostream>   // endl
#include <sstream>    // stringstream
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../../../coordinates/coordinates.hpp"
#include "../../../eos/eos.hpp"
#include "../../../field/field.hpp"
#include "../../../globals.hpp"
#include "../../../dustfluids/dustfluids.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../parameter_input.hpp"
#include "../../../utils/buffer_utils.hpp"
#include "../../bvals.hpp"
#include "../../bvals_interfaces.hpp"

// MPI header
#ifdef MPI_PARALLEL
#include <mpi.h>
#endif

//--------------------------------------------------------------------------------------
//! \fn void DustFluidsBoundaryVariable::AddDustFluidsShearForInit()
//  \brief Send shearing box boundary buffers for dustfluids variables

void DustFluidsBoundaryVariable::AddDustFluidsShearForInit() {
  MeshBlock *pmb = pmy_block_;
  Mesh *pmesh = pmb->pmy_mesh;
  AthenaArray<Real> &var = *var_cc;

  int jl = pmb->js - NGHOST;
  int ju = pmb->je + NGHOST;
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmesh->mesh_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  Real qomL = pbval_->qomL_;

  int sign[2]{1, -1};
  int ib[2]{pmb->is - NGHOST, pmb->ie + 1};

  // could call modified ShearQuantities(src=shear_cc_, dst=var, upper), by first loading
  // shear_cc_=var for rho_id, v2_id so that order of v2_id update to var doesn't matter.
  // Would need to reassign src=shear_cc_ to updated dst=var for v2_id after? Is it used?
  for (int upper=0; upper<2; upper++) {
    if (pbval_->is_shear[upper]) {
      // step 1. -- add shear to the periodic boundary values
      for (int n=0; n<NDUSTFLUIDS; ++n) {
				int dust_id = n;
				int rho_id  = 4*dust_id;
				int v1_id   = rho_id + 1;
				int v2_id   = rho_id + 2;
				int v3_id   = rho_id + 3;
				for (int k=kl; k<=ku; k++) {
					for (int j=jl; j<=ju; j++) {
						for (int i=0; i<NGHOST; i++) {
							int ii = ib[upper] + i;
							// add shear to conservative
							shear_cc_[upper](v2_id, k, j, i) = var(v2_id, k, j, ii)
																						+ sign[upper]*qomL*var(rho_id, k, j, ii);
							var(v2_id, k, j, ii) = shear_cc_[upper](v2_id, k, j, i);
						}
					}
				}
			}
		}  // if boundary is shearing
	}  // loop over inner/outer boundaries
  return;
}
// --------------------------------------------------------------------------------------
// ! \fn void DustFluidsBoundaryVariable::ShearQuantities(AthenaArray<Real> &shear_cc_,
//                                                   bool upper)
//  \brief Apply shear to DustFluids x2 momentum and energy

void DustFluidsBoundaryVariable::ShearQuantities(AthenaArray<Real> &shear_cc_, bool upper) {
  MeshBlock *pmb = pmy_block_;
  Mesh *pmesh = pmb->pmy_mesh;
  AthenaArray<Real> &var = *var_cc;
  int js = pmb->js;
  int je = pmb->je;

  int jl = js - NGHOST;
  int ju = je + NGHOST;
  int kl = pmb->ks;
  int ku = pmb->ke;
  if (pmesh->mesh_size.nx3 > 1) {
    kl -= NGHOST;
    ku += NGHOST;
  }

  Real qomL = pbval_->qomL_;
  int sign[2]{1, -1};
  int ib[2]{pmb->is - NGHOST, pmb->ie + 1};

	for (int n=0; n<NDUSTFLUIDS; ++n) {
		int dust_id = n;
		int rho_id  = 4*dust_id;
		int v1_id   = rho_id + 1;
		int v2_id   = rho_id + 2;
		int v3_id   = rho_id + 3;
		for (int k=kl; k<=ku; k++) {
			for (int j=jl; j<=ju; j++) {
				for (int i=0; i<NGHOST; i++) {
					int ii = ib[upper] + i;
					shear_cc_(v2_id,k,j,i) += + sign[upper]*qomL*var(rho_id,k,j,ii);
				}
			}
		}
	}
  return;
}
