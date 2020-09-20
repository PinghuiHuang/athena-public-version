#ifndef DRAG_DUSTGAS_HPP_
#define DRAG_DUSTGAS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_srcterms.hpp
//  \brief defines class DustGasDrag
//  Contains data and functions that implement physical (not coordinate) drag terms

// C headers

// C++ headers
#include <cstring>    // strcmp
#include <sstream>

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../dustfluids.hpp"

// Forward declarations
class DustFluids;
class ParameterInput;

//! \class DustGasDrag
//  \brief data and functions for drags between dust and gas
class DustGasDrag {
  public:
    DustGasDrag(DustFluids *pdf, ParameterInput *pin);

    // data
    bool DustFeedback_Flag; // true or false, the flag of dust feedback term

    // functions
    void AerodynamicDrag(MeshBlock *pmb, const int stage, const Real dt,
        const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //// Implicit Schemes
    //void SingleDustNoFeedbackImplicit(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //void SingleDustFeedbackImplicit(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //void MultipleDustNoFeedbackImplicit(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //void MultipleDustFeedbackImplicit(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //// Semi-Implicit Schemes
    //void SingleDustNoFeedbackSemiImplicit(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //void SingleDustFeedbackSemiImplicit(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    // 2-order implicit drag scheme, depends on the time integrator
    void VL2ImplicitFeedback(MeshBlock *pmb, const int stage,
        const Real dt, const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //void VL2ImplicitNoFeedback(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    void RK2ImplicitFeedback(MeshBlock *pmb, const int stage,
        const Real dt, const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    //void RK2ImplicitNoFeedback(MeshBlock *pmb, const int stage,
        //const Real dt, const AthenaArray<Real> &stopping_time,
        //const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        //const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    // Functions worked on calculating drag-matrixes
    // Matrix Addition
    void Addition(const AthenaArray<Real> &a_matrix, const Real b_num,
                  const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix);

    void Addition(AthenaArray<Real> &a_matrix, const Real b_num, const AthenaArray<Real> &b_matrix);

    void Addition(const Real a_num, const Real b_num,
                  const AthenaArray<Real> &b_matrix, AthenaArray<Real> &c_matrix);

    void Addition(const Real a_num, const Real b_num, AthenaArray<Real> &b_matrix);

    // Matrix Multiplication
    void Multiplication(const AthenaArray<Real> &a_matrix, const AthenaArray<Real> &b_matrix,
                                          AthenaArray<Real> &c_matrix);

    void Multiplication(const Real a_num, const AthenaArray<Real> &b_matrix,
                                          AthenaArray<Real> &c_matrix);

    void Multiplication(const Real a_num, AthenaArray<Real> &b_matrix);

    // LU decompose on the drags matrix
    void LUdecompose(const AthenaArray<Real> &a_matrix);

    void SolveLinearEquation(AthenaArray<Real> &b_vector, AthenaArray<Real> &x_vector);

    void SolveMultipleLinearEquation(AthenaArray<Real> &b_matrix, AthenaArray<Real> &x_matrix);

    // Calculate the inverse of matrix
    void Inverse(AthenaArray<Real> &a_matrix, AthenaArray<Real> &a_inv_matrix);

    // calculate the determinant of drags matrix
    Real Determinant();

  private:
    static const int num_species  = NDUSTFLUIDS + 1; // gas and n dust fluids
    static const int num_dust_var = 4*NDUSTFLUIDS;   // Number of dust variables (rho, v1, v2, v3)*4
    std::string integrator;                          // Time Integrator
    DustFluids  *pmy_dustfluids_;                    // ptr to DustFluids containing this DustGasDrag
    MeshBlock   *pmb_;                               // ptr to meshblock containing this DustGasDrag
    Coordinates *pco_;                               // ptr to coordinates class
    //Real        hydro_gamma_;                      // The adiabatic index of gas

    // data for LU decomposition
    AthenaArray<Real> drags_matrix; // The matrix of drags between dust and gas
    AthenaArray<Real> scale_vector; // scale_vector stores the implicit scaling of each row
    AthenaArray<Real> aref_matrix;
    AthenaArray<Real> lu_matrix;    // Stores the decomposition.
    AthenaArray<int>  indx_array;   // Stores the permutation.
    Real det;                       // The determinant of the matrix of drags
};
#endif // DRAG_DUSTGAS_HPP_
