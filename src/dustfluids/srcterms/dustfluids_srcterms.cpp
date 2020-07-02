//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//  \brief Class to implement source terms in the dustfluids equations

// C headers

// C++ headers
#include <cstring>    // strcmp
#include <iostream>
#include <sstream>
#include <stdexcept>  // runtime_error
#include <string>     // c_str()

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../../mesh/mesh.hpp"
#include "../../parameter_input.hpp"
#include "../dustfluids.hpp"
#include "dustfluids_srcterms.hpp"


class DustFluids;
class ParameterInput;


//----------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::AddDustFluidsSourceTerms
//  \brief Adds source terms to conserved variables

DustFluidsSourceTerms::DustFluidsSourceTerms(DustFluids *pdf, ParameterInput *pin) {
  pmy_dustfluids_                = pdf;
  dustfluids_sourceterms_defined = false;

  // read point mass or constant acceleration parameters from input block

  // set the point source only when the coordinate is spherical or 2D
  // cylindrical.
  gm_ = pin->GetOrAddReal("problem","GM",0.0);
  if (gm_ != 0.0) {
    if (std::strcmp(COORDINATE_SYSTEM, "spherical_polar") == 0
        || (std::strcmp(COORDINATE_SYSTEM, "cylindrical") == 0
            && pdf->pmy_block->block_size.nx3==1)) {
      dustfluids_sourceterms_defined = true;
    } else {
      std::stringstream msg;
      msg << "### FATAL ERROR in DustFluidsSourceTerms constructor" << std::endl
          << "The point mass gravity works only in spherical polar coordinates"
          << "or in 2D cylindrical coordinates." << std::endl
          << "Check <problem> GM parameter in the input file." << std::endl;
      ATHENA_ERROR(msg);
    }
  }

  // read shearing box parameters from input block
  Omega_0_    = pin->GetOrAddReal("problem",    "Omega0",     0.0);
  qshear_     = pin->GetOrAddReal("problem",    "qshear",     0.0);
  ShBoxCoord_ = pin->GetOrAddInteger("problem", "shboxcoord", 1);

  if ((Omega_0_ !=0.0) && (qshear_ != 0.0)) dustfluids_sourceterms_defined = true;

  //if (SELF_GRAVITY_ENABLED) dustfluids_sourceterms_defined = true;

  //UserSourceTerm = pmy_dustfluids_->pmy_block->pmy_mesh->UserSourceTerm_;
  //if (UserSourceTerm != nullptr)
    //dustfluids_sourceterms_defined = true;
}

//----------------------------------------------------------------------------------------
//! \fn void DustFluidsSourceTerms::AddSourceTerms_Dustfluids
//  \brief Adds source terms to conserved variables

void DustFluidsSourceTerms::AddDustFluidsSourceTerms(const Real time, const Real dt,
                     const AthenaArray<Real> *flux_df, const AthenaArray<Real> &prim_df,
                     AthenaArray<Real> &cons_df) {
  MeshBlock *pmb = pmy_dustfluids_->pmy_block;

  // accleration due to point mass (MUST BE AT ORIGIN)
  if (gm_ != 0.0)
    PointMass_DustFluids(dt, flux_df, prim_df, cons_df);

  // Add new source terms here
  //if (SELF_GRAVITY_ENABLED) SelfGravity(dt, flux, prim, cons);

  // shearing box source terms: tidal and Coriolis forces
  if ((Omega_0_ !=0.0) && (qshear_ != 0.0))
    ShearingBoxSourceTerms_DustFluids(dt, flux_df, prim_df, cons_df);
  // MyNewSourceTerms()

  ////  user-defined source terms
  //if (UserSourceTerm != nullptr)
    //UserSourceTerm(pmb, time, dt, prim_df, cons_df);

  return;
}
