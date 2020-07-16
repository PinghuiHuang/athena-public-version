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
    bool DustFeedback_Flag;           // true or false, the flag of dust feedback term

    // functions
    void Aerodynamics_Drag(MeshBlock *pmb, const Real dt, const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    void SingleDust_NoFeedback(MeshBlock *pmb, const Real dt,
        const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    void SingleDust_Feedback(MeshBlock *pmb, const Real dt,
        const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    void MultipleDust_NoFeedback(MeshBlock *pmb, const Real dt,
        const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        const AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    void MultipleDust_Feedback(MeshBlock *pmb, const Real dt,
        const AthenaArray<Real> &stopping_time,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
        AthenaArray<Real> &u, AthenaArray<Real> &cons_df);

    // The functions on calculating the drags-matrix
    // LU decompose on the drags matrix
    void LUdecompose(const AthenaArray<Real> &a_matrix);

    void SolveLinearEquation(AthenaArray<Real> &b_vector, AthenaArray<Real> &x_vector);

    void SolveMultipleLinearEquation(AthenaArray<Real> &b_matrix, AthenaArray<Real> &x_matrix);

    // Calculate the inverse of drags matrix
    AthenaArray<Real> InverseMatrix(AthenaArray<Real> &a_matrix);

    // calculate the determinant of drags matrix
    Real Determinant();

    // Iterative improve the precision of solve linear equations
    void IterativeImprove(AthenaArray<Real> &b_vector, AthenaArray<Real> &x_vector);

  private:
    const int num_species = NDUSTFLUIDS + 1; // gas and n dust fluids
    DustFluids  *pmy_dustfluids_;            // ptr to DustFluids containing this DustGasDrag
    MeshBlock   *pmb_;                       // ptr to meshblock containing this DustGasDrag
    Coordinates *pco_;                       // ptr to coordinates class
    Real        hydro_gamma_;                // The adiabatic index of gas

    // data for LU decomposition
    AthenaArray<Real> drags_matrix; // The matrix of drags between dust and gas
    AthenaArray<Real> scale_vector; // scale_vector stores the implicit scaling of each row
    AthenaArray<Real> aref_matrix;
    AthenaArray<Real> lu_matrix;    // Stores the decomposition.
    AthenaArray<int>  indx_array;   // Stores the permutation.
    Real det;                       // The determinant of the matrix of drags
};
#endif // DRAG_DUSTGAS_HPP_
