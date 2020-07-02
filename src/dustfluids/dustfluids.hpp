#ifndef DUSTFLUIDS_HPP_
#define DUSTFLUIDS_HPP_
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file dust_fluid.hpp
//  \brief definitions for DustFluid class

// C headers

// C++ headers

// Athena++ headers
#include "../athena.hpp"
#include "../athena_arrays.hpp"
#include "../bvals/cc/bvals_cc.hpp"
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
//  \brief

class DustFluids {
  friend class EquationOfState;
  friend class Hydro;

  public:
    DustFluids(MeshBlock *pmb, ParameterInput *pin);
    MeshBlock* pmy_block;
    // Leaving as ctor parameter in case of run-time "ndustfluids" option

    // public data:
    // "conserved vars" = density, momentum of dust fluids
    AthenaArray<Real> df_cons, df_cons1, df_cons2;      // time-integrator memory register #1
    // "primitive vars" = density, velocity of dust fluids
    AthenaArray<Real> df_prim, df_prim1;        // time-integrator memory register #3
    AthenaArray<Real> df_flux[3];               // face-averaged flux vector

    AthenaArray<Real> stopping_time_array;      // Arrays of stopping time of dust
    AthenaArray<Real> nu_dustfluids_array;      // Arrays of dust diffusivity array, nu_d
    AthenaArray<Real> cs_dustfluids_array;      // Arrays of sound speed of dust, cs_d^2 = nu_d/T_eddy

    // fourth-order intermediate quantities
    AthenaArray<Real> df_cons_cc, df_prim_cc;   // cell-centered approximations
    // (only needed for 4th order EOS evaluations that have explicit dependence on species
    // concentration)

    // storage for mesh refinement, SMR/AMR
    AthenaArray<Real> coarse_df_cons_, coarse_df_prim_;
    int refinement_idx{-1};

    CellCenteredBoundaryVariable dfbvar;
    DustGasDrag                  dfdrag;  // Objects used in calculating the dust-gas drags
    DustFluidsDiffusion          dfdif;   // Objects used in calculating the diffusions of dust
    DustFluidsSourceTerms        dfsrc;   // Objects used in calculating the source terms of dust

    AthenaArray<Real> particle_density_;    // normalized dust particle internal density, used in user defined stopping time
    AthenaArray<Real> const_stopping_time_; // Constant stopping time
    AthenaArray<Real> const_nu_dust_;       // Constant concentration diffusivity of dust
    bool ConstStoppingTime_Flag_;           // true or false, the flag of using the constant stopping time of dust
    bool ConstNu_Flag_;                     // true or false, the flag of using the constant diffusivity of dust
    bool SoundSpeed_Flag_;                  // true or false, turn on the sound speed of dust fluids


    // public functions:
    void AddDustFluidsFluxDivergence(const Real wght, AthenaArray<Real> &cons_df);
    void CalculateDustFluidsFluxes(const int order, AthenaArray<Real> &prim_df);
    void CalculateDustFluidsFluxes_STS();

    // The Riemann Solver for dust fluids
    // HLLE solver without sound speed of dust
    void NoCs_RiemannSolver_DustFluids( const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &df_prim_l,
        AthenaArray<Real> &df_prim_r, AthenaArray<Real> &dust_flux);

    // HLLE solver with sound speed of dust
    void HLLE_RiemannSolver_DustFluids(const int k, const int j, const int il, const int iu,
        const int index, AthenaArray<Real> &df_prim_l,
        AthenaArray<Real> &df_prim_r, AthenaArray<Real> &dust_flux);

    // Computes the new timestep of advection of dust in a meshblock
    Real NewAdvectionDt();

    // Set up stopping time, diffusivity and sound speed of dust
    void SetDustFluidsProperties();


  private:
    // scratch space used to compute fluxes
    // 2D scratch arrays
    AthenaArray<Real> dt1_, dt2_, dt3_;                     // scratch arrays used in NewAdvectionDt
    AthenaArray<Real> df_prim_l_, df_prim_r_, df_prim_lb_;
    // 1D scratch arrays
    AthenaArray<Real> x1face_area_, x2face_area_, x3face_area_;
    AthenaArray<Real> x2face_area_p1_, x3face_area_p1_;
    AthenaArray<Real> cell_volume_;
    AthenaArray<Real> dflx_;
    //AthenaArray<Real> dx_df_prim_;

    // fourth-order
    // 4D scratch arrays
    AthenaArray<Real> scr1_nkji_, scr2_nkji_;
    AthenaArray<Real> df_prim_l3d_, df_prim_r3d_;
    // 1D scratch arrays
    AthenaArray<Real> laplacian_l_df_fc_, laplacian_r_df_fc_;
    void AddDiffusionFluxes();        // Add the diffusion flux on the dust flux

};
#endif // DUSTFLUIDS_HPP_
