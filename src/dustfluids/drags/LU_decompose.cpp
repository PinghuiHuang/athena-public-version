//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids_diffusion.cpp
//  \brief Compute dustfluids fluxes corresponding to diffusion processes.

// C headers

// C++ headers
#include <algorithm>   // min,max
#include <limits>
#include <cstring>    // strcmp
#include <sstream>

// Athena++ headers
#include "../../defs.hpp"
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../../coordinates/coordinates.hpp"
#include "../dustfluids.hpp"
#include "dust_gas_drag.hpp"

// OpenMP header
#ifdef OPENMP_PARALLEL
#include <omp.h>
#endif


//! \class DustGasDrag
//  \brief data and functions for drags between dust and gas
// The LUdecompose algorithms come from "Nurmerical Recipes", 3ed, Charpter 2.3, William H. Press et al. 2007

void DustGasDrag::LUdecompose(const AthenaArray<Real> &a_matrix)
{
  int i, j, k, imax;
  Real biggest_num, temp;
  det = 1.0; //No row interchanges yet.
  AthenaArray<Real> scale_vector(num_species); // scale_vector stores the implicit scaling of each row
  //lu_matrix.NewAthenaArray(num_species, num_species), Stores the decomposition.

  indx_array.ZeroClear();
  lu_matrix.ZeroClear();

  lu_matrix = a_matrix;

  for (i = 0; i<num_species; i++) { // Loop over rows to get the implicit scaling information
    biggest_num = 0.0;
#pragma omp simd
    for (j = 0; j<num_species; j++)
      if ((temp = std::abs(lu_matrix(i,j))) > biggest_num) biggest_num = temp;
    if (biggest_num == 0.0) {
      std::stringstream msg;
      msg << "### FATAL ERROR in Singular matrix in LU decomposition" << std::endl;
      ATHENA_ERROR(msg); // No nonzero largest element.
    }
    scale_vector(i) = 1.0/biggest_num; // Save the scaling.
  }

  for (k = 0; k<num_species; k++) {         // This is the outermost kij loop
    biggest_num = 0.0;                      // Initialize for the search for largest pivot element.
    for (i = k; i<num_species; i++) {
      temp = scale_vector(i)*std::abs(lu_matrix(i,k));
      if (temp > biggest_num) {             // Is the figure of merit for the pivot better than the best so far?
        biggest_num = temp;
        imax = i;
      }
    }
    if (k != imax) {                        // Interchange Rows
#pragma omp simd
      for (j = 0; j<num_species; j++) {
        temp              = lu_matrix(imax,j);
        lu_matrix(imax,j) = lu_matrix(k,j);
        lu_matrix(k,j)    = temp;
      }
      det = -det;                               // change the parity of det
      scale_vector(imax) = scale_vector(k);     // Also interchange the scale factor
    }
    indx_array(k) = imax;
    if (lu_matrix(k,k) == 0.0)
      lu_matrix(k,k) = TINY_NUMBER;

    for (i = k+1; i<num_species; i++) {         // Divide by the pivot element
      temp = lu_matrix(i,k) /= lu_matrix(k,k);
#pragma omp simd
      for (j = k+1; j<num_species; j++)         // Innermost loop: reduce remaining submatrix.
        lu_matrix(i,j) -= temp*lu_matrix(k,j);
    }
  }
  return;
}

void DustGasDrag::SolveLinearEquation(AthenaArray<Real> &b_vector, AthenaArray<Real> &x_vector)
{
  int i, ii = 0, ip, j;
  Real sum;

  if (b_vector.GetDim1() != num_species || x_vector.GetDim1() != num_species){
    std::stringstream msg;
    msg << "### FATAL ERROR in DustGasDrag::SolveLinearEquation bad sizes" << std::endl;
    ATHENA_ERROR(msg);
  }

  for (i = 0; i<num_species; i++)
    x_vector(i) = b_vector(i);

  for (i = 0; i<num_species; i++) { // When ii is set to a positive value,
    ip           = indx_array(i);   // it will become the index of the first nonvanishing element of b.
    sum          = x_vector(ip);    // We now do the forward substitution
    x_vector(ip) = x_vector(i);     // The only new wrinkle is to unscramble the permutation
    if (ii != 0)
      for (j = ii-1; j<i; j++) sum -= lu_matrix(i,j)*x_vector(j);
    else if (sum != 0.0)            // A nonzero element was encountered, so from now on we
      ii = i+1;                     // will have to do the sums in the loop above.
    x_vector(i)=sum;
  }

  for (i = num_species-1; i>=0; i--) { // Now we do the backsubstitution,
    sum = x_vector(i);
#pragma omp simd
    for (j = i+1; j<num_species; j++) sum -= lu_matrix(i,j)*x_vector(j);
    x_vector(i) = sum/lu_matrix(i,i); // Store a component of the solution vector X
  }
  return;
}

void DustGasDrag::SolveMultipleLinearEquation(AthenaArray<Real> &b_matrix, AthenaArray<Real> &x_matrix)
{
  int i,j,m = b_matrix.GetDim2();
  if (b_matrix.GetDim1() != num_species || x_matrix.GetDim1() != num_species
      || b_matrix.GetDim2() != x_matrix.GetDim2()) {
    std::stringstream msg;
    msg << "### FATAL ERROR in DustGasDrag::SolveMultipleLinearEquation bad sizes" << std::endl;
    ATHENA_ERROR(msg);
  }

  AthenaArray<Real> xx(num_species);
  for (j = 0; j<m; j++) {  // Copy and solve each column in turn.
    for (i = 0; i<num_species; i++) xx(i) = b_matrix(i,j);
    SolveLinearEquation(xx,xx);
    for (i = 0; i<num_species; i++) x_matrix(i,j) = xx(i);
  }
  return;
}

AthenaArray<Real> DustGasDrag::InverseMatrix(AthenaArray<Real> &a_matrix)
{ //Using the stored LU decomposition, return in ainv the matrix inverse A^-1.
  int i,j;
  //ainv.resize(num_species,num_species);
  AthenaArray<Real> a_inv_matrix = a_matrix;
  for (i = 0; i<num_species; i++) {
    for (j = 0; j<num_species; j++) a_matrix(i,j) = 0.;
    a_matrix(i,i) = 1.;
  }
  SolveMultipleLinearEquation(a_matrix, a_inv_matrix);
  return a_inv_matrix;
}

Real DustGasDrag::Determinant()
{ // Using the stored LU decomposition, return the determinant of the matrix A
  Real dd = det;
  for (int i = 0; i<num_species; i++) dd *= lu_matrix(i,i);
  return dd;
}

void DustGasDrag::IterativeImprove(AthenaArray<Real> &b_vector, AthenaArray<Real> &x_vector)
{
  int i,j;
  AthenaArray<Real> r_vector(num_species);
  //AthenaArray<Real> aref_matrix(num_species, num_species);
  //aref_matrix.ZeroClear();
  aref_matrix = drags_matrix;

  for (i = 0; i<num_species; i++) {
    long double sdp = -b_vector(i);
#pragma omp simd
    for (j = 0; j<num_species; j++)
      sdp += (long double)aref_matrix(i,j) * (long double)x_vector(j);
    r_vector(i) = sdp;
  }
  SolveLinearEquation(r_vector,r_vector);
  for (i = 0; i<num_species; i++) x_vector(i) -= r_vector(i);
  return;
}
