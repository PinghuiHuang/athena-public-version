#ifndef DUSTFLUIDS_SRCTERMS_HPP_
#define DUSTFLUIDS_SRCTERMS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_srcterms.hpp
//  \brief defines class DustFluidsSourceTerms
//  Contains data and functions that implement physical (not coordinate) source terms

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../dustfluids.hpp"

// Forward declarations
class DustFluids;
class ParameterInput;

//! \class DustFluidsSourceTerms
//  \brief data and functions for physical source terms in the dustfluids
class DustFluidsSourceTerms {
  public:
    DustFluidsSourceTerms(DustFluids * pdf, ParameterInput *pin);

    // accessors
    Real GetGM() const {return gm_;}

    // data
    bool dustfluids_sourceterms_defined;

    // functions
    void AddDustFluidsSourceTerms(const Real time, const Real dt,
        const AthenaArray<Real> *flux_df, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &cons_df);
    // shearing box src terms
    void ShearingBoxSourceTerms_DustFluids(const Real dt, const AthenaArray<Real> *flux_df,
                                  const AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);

    void PointMass_DustFluids(const Real dt, const AthenaArray<Real> *flux_df,
                              const AthenaArray<Real> &prim_df, AthenaArray<Real> &cons_df);

  //void EnrollSrcTermFunction(SrcTermFunc my_func);
  //SrcTermFunc UserSourceTerm;

  private:
    DustFluids *pmy_dustfluids_;  // ptr to DustFluids containing this DustFluidsSourceTerms
    Real gm_;                     // GM for point mass MUST BE LOCATED AT ORIGIN
    Real Omega_0_, qshear_;       // Orbital freq and shear rate in shearing box
    int  ShBoxCoord_;             // ShearCoordinate type: 1=xy (default), 2=xz
};
#endif // DUSTFLUIDS_SRCTERMS_HPP_
