#ifndef BVALS_CC_DUSTFLUIDS_BVALS_HPP_
#define BVALS_CC_DUSTFLUIDS_BVALS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file bvals_dustfluids.hpp
//  \brief

// C headers

// C++ headers

// Athena++ headers
#include "../../../athena.hpp"
#include "../../../athena_arrays.hpp"
#include "../bvals_cc.hpp"

//----------------------------------------------------------------------------------------
//! \class CellCenteredBoundaryVariable
//  \brief

class DustFluidsBoundaryVariable : public CellCenteredBoundaryVariable {
 public:
  DustFluidsBoundaryVariable(MeshBlock *pmb,
                        AthenaArray<Real> *var_dustfluids, AthenaArray<Real> *coarse_var,
                        AthenaArray<Real> *var_flux,
                        DustFluidsBoundaryQuantity dustfluids_type);
                                                // AthenaArray<Real> &prim);
  virtual ~DustFluidsBoundaryVariable() = default;

  // switch between DustFluids class members "u" and "w" (or "u" and "u1", ...)
  void SwapDustFluidsQuantity(AthenaArray<Real> &var_dustfluids, DustFluidsBoundaryQuantity dustfluids_type);
  void SelectCoarseBuffer(DustFluidsBoundaryQuantity dustfluids_type);

  void AddDustFluidsShearForInit();
  void ShearQuantities(AthenaArray<Real> &shear_cc_, bool upper) override;

  // BoundaryPhysics: need to flip sign of velocity vectors for Reflect*()
  void ReflectInnerX1(Real time, Real dt,
                      int il, int jl, int ju, int kl, int ku, int ngh) override;
  void ReflectOuterX1(Real time, Real dt,
                      int iu, int jl, int ju, int kl, int ku, int ngh) override;
  void ReflectInnerX2(Real time, Real dt,
                      int il, int iu, int jl, int kl, int ku, int ngh) override;
  void ReflectOuterX2(Real time, Real dt,
                      int il, int iu, int ju, int kl, int ku, int ngh) override;
  void ReflectInnerX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int kl, int ngh) override;
  void ReflectOuterX3(Real time, Real dt,
                      int il, int iu, int jl, int ju, int ku, int ngh) override;
  //protected:
 private:
  // DustFluids is a unique cell-centered variable because of the relationship between
  // DustFluidsBoundaryQuantity::cons u and DustFluidsBoundaryQuantity::prim w.
  DustFluidsBoundaryQuantity dustfluids_type_;
};

#endif // BVALS_CC_DUSTFLUIDS_BVALS_HPP_
