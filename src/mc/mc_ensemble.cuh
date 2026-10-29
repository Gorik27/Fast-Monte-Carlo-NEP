/*
    Copyright 2017 Zheyong Fan and GPUMD development team
    This file is part of GPUMD.
    GPUMD is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    GPUMD is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with GPUMD.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once
#include "model/atom.cuh"
#include "model/box.cuh"
#include "model/group.cuh"
#include "nep_energy.cuh"
#include "nep_energy_fast.cuh"
#include "utilities/gpu_vector.cuh"
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#ifdef MC_TEST
// One logged trial swap of the parallel (fastNEP) SGC/VCSGC implementation.
// It is replayed step by step in the original SGC implementation.
struct MC_TestStep {
  int md_step = 0;
  int trial = 0;          // index of the composite trial move inside one MC call
  int domain = 0;
  int i = -1;             // swapped atom
  int type_i = -1;        // type before the swap
  int type_j = -1;        // type after the swap
  int index_old_species = -1;
  int index_new_species = -1;
  double dE_potential = 0.0; // potential energy change (NEP + ZBL), without mu/phi/kappa terms
  bool applied = false;   // swap was really applied (domain accepted AND global acceptance)
};

struct MC_TestReport {
  int num_steps = 0;
  int num_applied = 0;
  int num_energy_fail = 0; // |dE_parallel - dE_original| out of tolerance
  int num_type_fail = 0;   // state before the step differs from the logged one
  double max_abs_err = 0.0;
  double max_rel_err = 0.0;
  double max_abs_err_bin[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // max |dE diff| in fifths of the MC call
};

// Cached per-atom data of the parallel implementation (energies, descriptors) compared with
// a from-scratch recalculation at the end of an MC call
struct MC_TestCacheReport {
  bool checked = false;     // false if the ensemble has no such cache
  int num_pe_fail = 0;
  int num_q_fail = 0;
  int num_s_fail = 0;
  double max_pe_err = 0.0;  // eV
  double max_q_err = 0.0;
  double max_s_err = 0.0;
  double total_pe_diff = 0.0; // sum(cached pe) - sum(fresh pe), eV
};

#ifndef MC_TEST_DESC_TOL
#define MC_TEST_DESC_TOL 1.0e-3 // descriptors: absolute and relative tolerance
#endif
#ifndef MC_TEST_ATOL
#define MC_TEST_ATOL 2.0e-3 // eV, float arithmetic + summation over ~100 neighbors
#endif
#ifndef MC_TEST_RTOL
#define MC_TEST_RTOL 1.0e-3
#endif
#endif // MC_TEST

class MC_Ensemble
{
public:
#ifdef MC_TEST
  std::vector<MC_TestStep> mc_test_steps; // filled by the parallel implementation
  // number of atoms of every species as counted by the ensemble itself
  virtual std::vector<int> test_num_atoms_species() const { return std::vector<int>(); }
  // compare cached energies/descriptors with a from-scratch recalculation
  virtual MC_TestCacheReport test_check_cache(Atom& atom, Box& box)
  {
    return MC_TestCacheReport();
  }
#endif
  MC_Ensemble(const char** param, int num_param);
  MC_Ensemble(const char** param, int num_param, const int num_atoms, const int Nx_domains, const int Ny_domains, const int Nz_domains); // overload in order to provide number of atoms to NEP_fast initialisation
  virtual ~MC_Ensemble(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id) = 0;

  const int n_max = 200; // size of arrays with radial neighbors quantities
  const int m_max = 200; // size of arrays with angular neighbors quantities

protected:
  int num_steps_mc = 0;
  double temperature = 0.0;
  std::mt19937 rng;

  std::ofstream mc_output;
  #ifdef DEBUG
    std::ofstream mc_debug_log;
  #endif

  GPU_Vector<int> NN_radial;
  GPU_Vector<int> NN_angular;
  GPU_Vector<int> type_before;
  GPU_Vector<int> type_after;
  GPU_Vector<int> local_type_before;
  GPU_Vector<int> local_type_after;
  GPU_Vector<int> t2_radial_before;
  GPU_Vector<int> t2_radial_after;
  GPU_Vector<int> t2_angular_before;
  GPU_Vector<int> t2_angular_after;
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;
  GPU_Vector<float> x12_angular;
  GPU_Vector<float> y12_angular;
  GPU_Vector<float> z12_angular;
  GPU_Vector<float> pe_before;
  GPU_Vector<float> pe_after;

  bool check_if_small_box(const double rc, const Box& box);

  NEP_Energy nep_energy;
  NEP_Energy_fast nep_energy_fast;
};
