#ifndef DUSTFLUIDS_HPP_
#define DUSTFLUIDS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dustfluids.hpp
//! \brief definitions for DustFluid class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/dustfluids/bvals_dustfluids.hpp"
#include "../hydro/hydro_diffusion/hydro_diffusion.hpp"
#include "diffusions/dustfluids_diffusion.hpp"
#include "drags/dust_gas_drag.hpp"
#include "srcterms/dustfluids_srcterms.hpp"

class MeshBlock;
class ParameterInput;
class Hydro;
class DustFluidsSourceTerms;
class DustFluidsDiffusion;
class DustGasDrag;

//! \class DustFluids
//! \brief dust fluids data and functions

class DustFluids {
  friend class EquationOfState;
  friend class Hydro;

  public:
    DustFluids(MeshBlock *pmb, ParameterInput *pin);

    MeshBlock* pmy_block;
    // Leaving as ctor parameter in case of run-time "ndustfluids" option

    // public data:
    // "conservative vars" = density, momentums of dust
    AthenaArray<Real> df_cons, df_cons1, df_cons2; // time-integrator memory register #1
    AthenaArray<Real> df_cons0, df_cons_fl_div;    // rkl2 STS memory registers;
    AthenaArray<Real> df_cons_bs, df_cons_as;      // time-integrator memory register before and after explicit sources terms

    // "primitive vars" = density, velocities of dust
    AthenaArray<Real> df_prim, df_prim1, df_prim_n;  // time-integrator memory register #3
    AthenaArray<Real> df_flux[3];                    // face-averaged flux vector

    // storage for mesh refinement, SMR/AMR
    AthenaArray<Real> coarse_df_cons_, coarse_df_prim_; // coarse df_cons and coarse df_prim, used in mesh refinement
    int refinement_idx{-1};                             // vector of pointers in MeshRefinement class

    AthenaArray<Real> stopping_time_array;      // Arrays of stopping time of dust
    AthenaArray<Real> nu_dustfluids_array;      // Arrays of dust diffusivity array, nu_d
    AthenaArray<Real> cs_dustfluids_array;      // Arrays of sound speed of dust, cs_d^2 = nu_d/T_eddy

    AthenaArray<Real> stopping_time_array_n;    // Arrays of stopping time of dust at stage n
    AthenaArray<Real> nu_dustfluids_array_n;    // Arrays of dust diffusivity array, nu_d at stage n
    AthenaArray<Real> cs_dustfluids_array_n;    // Arrays of sound speed of dust, cs_d^2 = nu_d/T_eddy at stage n

    // fourth-order intermediate quantities
    AthenaArray<Real> df_cons_cc, df_prim_cc;   // cell-centered approximations
    // (only needed for 4th order EOS evaluations that have explicit dependence on species
    // concentration)

    DustFluidsBoundaryVariable dfbvar;  // Dust Fluids boundary variables Object (Cell-Centered)
    DustGasDrag                dfdrag;  // Object used in calculating the dust-gas drags
    DustFluidsDiffusion        dfdif;   // Object used in calculating the diffusions of dust
    DustFluidsSourceTerms      dfsrc;   // Object used in calculating the source terms of dust

    bool ConstStoppingTime_Flag;           // true or false, the flag of using the constant stopping time of dust
    bool SoundSpeed_Flag;                  // true or false, turn on the sound speed of dust fluids

    Real internal_density[NDUSTFLUIDS];    // normalized dust internal density, used in user defined stopping time
    Real const_stopping_time[NDUSTFLUIDS]; // Constant stopping time
    Real const_nu_dust[NDUSTFLUIDS];       // Constant concentration diffusivity of dust


    // Public functions:
    // Stopping time
    // Calculate the user defined stopping time, varied with the properties of gas and dust
    void UserDefinedStoppingTime(const int kl, const int ku, const int jl, const int ju,
        const int il, const int iu, const Real *internal_density,
        const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df, AthenaArray<Real> &stopping_time);

    // Set the constant stopping time of dust
    void ConstantStoppingTime(const int kl, const int ku, const int jl, const int ju,
        const int il, const int iu, AthenaArray<Real> &stopping_time);

    // Calculate dust fluids flux
    void AddDustFluidsFluxDivergence(const Real wght, AthenaArray<Real> &cons_df); // Add flux divergence
    void AddDustFluidsFluxDivergence_STS(const Real wght, int stage, AthenaArray<Real> &cons_df_out,
                                           AthenaArray<Real> &cons_df_fl_div_out); // Add flux divergence
    void CalculateDustFluidsFluxes(AthenaArray<Real> &prim_df, const int order);   // Calculate fluxes of dust fluids
    void CalculateDustFluidsFluxes_STS(); // Calculate fluxes of dust fluids in super time step

    // Riemann Solvers for dust fluids
    // HLLE solver without sound speed of dust
    void HLLENoCsRiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &prim_df_l,
        AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux);

    // HLLE solver with sound speed of dust
    void HLLERiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &prim_df_l,
        AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux);

    // Roe solver of dust
    void RoeRiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &prim_df_l,
        AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux);

    // Riemann solver of dust
    void RiemannSolverDustFluids(const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &prim_df_l,
        AthenaArray<Real> &prim_df_r, AthenaArray<Real> &dust_flux);

    // Computes the new timestep of advection of dust in a meshblock
    Real NewAdvectionDt();

    // Set up stopping time, diffusivity and sound speed of dust fluids
    void SetDustFluidsProperties(const AthenaArray<Real> &w, const AthenaArray<Real> &prim_df,
    AthenaArray<Real> &stopping_time, AthenaArray<Real> &nu_dust, AthenaArray<Real> &cs_dust);


  private:
    // ptr to coordinates class
    Coordinates *pco_;
    // scratch space used to compute fluxes
    // 2D scratch arrays
    AthenaArray<Real> dt1_, dt2_, dt3_;                     // scratch arrays used in NewAdvectionDt
    AthenaArray<Real> df_prim_l_, df_prim_r_, df_prim_lb_;  // left and right states in reconstruction

    // 1D scratch arrays
    AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_; // face area in x1, x2, x3 directions
    AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;         // face area in x2, x3 directions
    AthenaArray<Real> cell_volume_;                             // the volume of the cells
    AthenaArray<Real> dflx_;
    // AthenaArray<Real> dx_df_prim_;

    // fourth-order
    // 4D scratch arrays
    AthenaArray<Real> scr1_nkji_,   scr2_nkji_;
    AthenaArray<Real> df_prim_l3d_, df_prim_r3d_;
    // 1D scratch arrays
    AthenaArray<Real> laplacian_l_df_fc_, laplacian_r_df_fc_;
    void AddDiffusionFluxes();        // Add the diffusive flux on the dust flux
};
#endif // DUSTFLUIDS_HPP_
