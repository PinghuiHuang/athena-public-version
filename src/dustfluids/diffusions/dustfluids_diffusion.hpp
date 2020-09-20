#ifndef DUSTFLUIDS_DIFFUSION_HPP_
#define DUSTFLUIDS_DIFFUSION_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file DustFluids_diffusion.hpp
//  \brief defines class DustFluidsDiffusion
//  Contains data and functions that implement the diffusion processes

// C headers

// C++ headers

// Athena++ headers
#include "../../athena.hpp"
#include "../../athena_arrays.hpp"
#include "../dustfluids.hpp"

// Forward declarations
class DustFluids;
class ParameterInput;
class Coordinates;


//! \class DustFluidsDiffusion
//  \brief data and functions for physical diffusion processes in the DustFluids

class DustFluidsDiffusion {
  public:
    DustFluidsDiffusion(DustFluids *pdf, ParameterInput *pin);

    // true or false, the bool value of the dust diffusion
    bool dustfluids_diffusion_defined; // true or false
    bool ConstNu_Flag;                // true or false, the flag of using the constant diffusivity of dust
    bool Momentum_Diffusion_Flag;     // true or false, the flag of momentum diffusion of dust fluids due to concentration diffusion.

    // The flux tensor of dust fluids caused by diffusions
    AthenaArray<Real> dustfluids_diffusion_flux[3];

    // functions
    // Calculate the diffusion flux
    void CalcDustFluidsDiffusionFlux(const AthenaArray<Real> &prim_df,
        const AthenaArray<Real> &cons_df);

    // Add the diffusion flux on df_flux
    void AddDustFluidsDiffusionFlux(AthenaArray<Real> *flux_diff,
        AthenaArray<Real> *flux_df);

    // reset the diffusion flux of dust as zero.
    void ClearDustFluidsFlux(AthenaArray<Real> *flux_diff);

    // calculate the new parabolic dt, make sure it won't conflict the CFL condition
    Real NewDiffusionDt();

    // Other functions
    Real VanLeerLimiter(const Real a, const Real b); // Van Leer Flux Limiter on the momentum diffusion

    // Transfer the coordinate into cylindrical, used in disk problem
    void GetCylCoord(Coordinates *pco, Real &rad, Real &phi, Real &z, int i, int j, int k);

    // Diffusivity
    // Calculate the dust diffusivity varied with the gas surface density and gas viscosity
    void UserDefinedDustDiffusivity(const AthenaArray<Real> &nu_gas,
      const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs);

    // Set the constant dust diffusivity
    void ConstantDustDiffusivity(const AthenaArray<Real> &nu_gas,
      const int kl, const int ku, const int jl, const int ju, const int il, const int iu,
      const AthenaArray<Real> &stopping_time,
      AthenaArray<Real> &dust_diffusivity, AthenaArray<Real> &dust_cs);

    // Concentration and Momentum diffusivity
    void DustFluidsConcentrationDiffusiveFlux(const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &w, AthenaArray<Real> *df_diff_flux);

    void DustFluidsMomentumDiffusiveFlux(const AthenaArray<Real> &prim_df,
      const AthenaArray<Real> &w, AthenaArray<Real> *df_flx);

  private:
    static const int num_dust_var = 4*NDUSTFLUIDS;  // Number of dust variables (rho, v1, v2, v3)*4
    DustFluids  *pmy_dustfluids_;                   // ptr to DustFluids containing this DustFluidsDiffusion
    MeshBlock   *pmb_;                              // ptr to meshblock containing this DustFluidsDiffusion
    Coordinates *pco_;                              // ptr to coordinates class
    //AthenaArray<Real> x1area_, x2area_, x2area_p1_, x3area_, x3area_p1_;
    //AthenaArray<Real> vol_;
    AthenaArray<Real> dx1_, dx2_, dx3_; // scratch arrays used in NewTimeStep
    AthenaArray<Real> diff_tot_;

    Real eddy_timescale_r0;      // The eddy timescale (turn over time of eddy) at r0
    Real r0_;                    // The length unit of radial direction in disk problem

    // functions pointer to calculate spatial dependent coefficients
    //DustFluidsDiffusionCoeffFunc CalcDustFluidsDiffusivityCoeff_;

};
#endif // DUSTFLUIDS_DIFFUSION_HPP_
