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
#include "mc_ensemble.cuh"

class MC_Ensemble_SGC_fastNEP : public MC_Ensemble
{
public:
  int num_atoms;
  int Nx_domains;
  int Ny_domains;
  int Nz_domains;
  int Nx_subdomains;
  int Ny_subdomains;
  int Nz_subdomains;
  int N_domains;
  int N_subdomains;

  MC_Ensemble_SGC_fastNEP(
    const int num_atoms,
    const int Nx_domains,
    const int Ny_domains,
    const int Nz_domains,
    const char** param,
    int num_param,
    double swap_fraction_mc,
    bool is_vcsgc,
    std::vector<std::string>& species,
    std::vector<int>& types,
    std::vector<int>& num_atoms_species,
    std::vector<double>& mu_or_phi,
    std::vector<double>& target_concentration,
    double kappa);
  virtual ~MC_Ensemble_SGC_fastNEP(void);

  virtual void compute(
    int md_step,
    double temperature,
    Atom& atom,
    Box& box,
    std::vector<Group>& group,
    int grouping_method,
    int group_id);

#ifdef MC_TEST
  virtual std::vector<int> test_num_atoms_species() const { return num_atoms_species; }
  virtual MC_TestCacheReport test_check_cache(Atom& atom, Box& box);
#endif

private:
  GPU_Vector<int> NN_ij;
  GPU_Vector<int> NL_ij;
  GPU_Vector<int> NN_angular_i;
  GPU_Vector<float> pe_nep_before_local;
  GPU_Vector<float> delta_pe_nep;
  GPU_Vector<float> delta_pe_zbl;
  double swap_fraction_mc;
  bool is_vcsgc = false;
  std::vector<std::string> species;
  std::vector<int> types;
  std::vector<int> num_atoms_species;
  std::vector<double> mu_or_phi;
  std::vector<double> target_concentration;
  int _index_old_species;
  double kappa;
  GPU_Vector<int> t2_radial;
  GPU_Vector<bool> is_neigh_angular;
  GPU_Vector<bool> g_is_accepted;
  GPU_Vector<float> x12_radial;
  GPU_Vector<float> y12_radial;
  GPU_Vector<float> z12_radial;

  bool allowed_species(std::string& species_found);

  int color_of(int i) const
  {
    int ix =  i % Nx_subdomains;
    int iy = (i / Nx_subdomains) % Ny_subdomains;
    int iz =  i / (Nx_subdomains * Ny_subdomains);
    return (ix & 1) * 4 + (iy & 1) * 2 + (iz & 1);
  }
};
