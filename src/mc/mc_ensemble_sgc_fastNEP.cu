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

/*----------------------------------------------------------------------------80
The semi-grand canonical (SGC) and variance-constrained SGC (VCSGC) ensembles
for MCMD.

[1] B. Sadigh, P. Erhart, A. Stukowski, A. Caro, E. Martinez, and L. Zepeda-Ruiz
Scalable parallel Monte Carlo algorithm for atomistic simulations of
precipitation in alloys, Phys. Rev. B 85, 184203 (2012).

[2] B. Sadigh and P. Erhart
Calculations of excess free energies of precipitates via direct thermodynamic
integration across phase boundaries, Phys. Rev. B 86, 134204 (2012).
------------------------------------------------------------------------------*/

#include "mc_ensemble_sgc_fastNEP.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/nep_utilities.cuh"
#include <map>
#include <cstring>
#include <cmath>
#include <array>
#include <algorithm> 

const std::map<std::string, double> MASS_TABLE{
  {"H", 1.0080000000},
  {"He", 4.0026020000},
  {"Li", 6.9400000000},
  {"Be", 9.0121831000},
  {"B", 10.8100000000},
  {"C", 12.0110000000},
  {"N", 14.0070000000},
  {"O", 15.9990000000},
  {"F", 18.9984031630},
  {"Ne", 20.1797000000},
  {"Na", 22.9897692800},
  {"Mg", 24.3050000000},
  {"Al", 26.9815385000},
  {"Si", 28.0850000000},
  {"P", 30.9737619980},
  {"S", 32.0600000000},
  {"Cl", 35.4500000000},
  {"Ar", 39.9480000000},
  {"K", 39.0983000000},
  {"Ca", 40.0780000000},
  {"Sc", 44.9559080000},
  {"Ti", 47.8670000000},
  {"V", 50.9415000000},
  {"Cr", 51.9961000000},
  {"Mn", 54.9380440000},
  {"Fe", 55.8450000000},
  {"Co", 58.9331940000},
  {"Ni", 58.6934000000},
  {"Cu", 63.5460000000},
  {"Zn", 65.3800000000},
  {"Ga", 69.7230000000},
  {"Ge", 72.6300000000},
  {"As", 74.9215950000},
  {"Se", 78.9710000000},
  {"Br", 79.9040000000},
  {"Kr", 83.7980000000},
  {"Rb", 85.4678000000},
  {"Sr", 87.6200000000},
  {"Y", 88.9058400000},
  {"Zr", 91.2240000000},
  {"Nb", 92.9063700000},
  {"Mo", 95.9500000000},
  {"Tc", 98},
  {"Ru", 101.0700000000},
  {"Rh", 102.9055000000},
  {"Pd", 106.4200000000},
  {"Ag", 107.8682000000},
  {"Cd", 112.4140000000},
  {"In", 114.8180000000},
  {"Sn", 118.7100000000},
  {"Sb", 121.7600000000},
  {"Te", 127.6000000000},
  {"I", 126.9044700000},
  {"Xe", 131.2930000000},
  {"Cs", 132.9054519600},
  {"Ba", 137.3270000000},
  {"La", 138.9054700000},
  {"Ce", 140.1160000000},
  {"Pr", 140.9076600000},
  {"Nd", 144.2420000000},
  {"Pm", 145},
  {"Sm", 150.3600000000},
  {"Eu", 151.9640000000},
  {"Gd", 157.2500000000},
  {"Tb", 158.9253500000},
  {"Dy", 162.5000000000},
  {"Ho", 164.9303300000},
  {"Er", 167.2590000000},
  {"Tm", 168.9342200000},
  {"Yb", 173.0450000000},
  {"Lu", 174.9668000000},
  {"Hf", 178.4900000000},
  {"Ta", 180.9478800000},
  {"W", 183.8400000000},
  {"Re", 186.2070000000},
  {"Os", 190.2300000000},
  {"Ir", 192.2170000000},
  {"Pt", 195.0840000000},
  {"Au", 196.9665690000},
  {"Hg", 200.5920000000},
  {"Tl", 204.3800000000},
  {"Pb", 207.2000000000},
  {"Bi", 208.9804000000},
  {"Po", 210},
  {"At", 210},
  {"Rn", 222},
  {"Fr", 223},
  {"Ra", 226},
  {"Ac", 227},
  {"Th", 232.0377000000},
  {"Pa", 231.0358800000},
  {"U", 238.0289100000},
  {"Np", 237},
  {"Pu", 244},
  {"Am", 243},
  {"Cm", 247},
  {"Bk", 247},
  {"Cf", 251},
  {"Es", 252},
  {"Fm", 257},
  {"Md", 258},
  {"No", 259},
  {"Lr", 262}};

MC_Ensemble_SGC_fastNEP::MC_Ensemble_SGC_fastNEP(
  const int num_atoms_input,
  const int Nx_domains_input,
  const int Ny_domains_input,
  const int Nz_domains_input,
  const char** param,
  int num_param,
  double swap_fraction_mc_input,
  bool is_vcsgc_input,
  std::vector<std::string>& species_input,
  std::vector<int>& types_input,
  std::vector<int>& num_atoms_species_input,
  std::vector<double>& mu_or_phi_input,
  std::vector<double>& target_conc_input,
  double kappa_input)
  : MC_Ensemble(param, num_param, num_atoms_input, Nx_domains_input, Ny_domains_input, Nz_domains_input)
{
  num_atoms = num_atoms_input;
  Nx_domains = Nx_domains_input;
  Ny_domains = Ny_domains_input;
  Nz_domains = Nz_domains_input;
  Nx_subdomains = Nx_domains*2;
  Ny_subdomains = Ny_domains*2;
  Nz_subdomains = Nz_domains*2;
  N_domains = Nx_domains * Ny_domains * Nz_domains;
  N_subdomains = N_domains*8;
  swap_fraction_mc = swap_fraction_mc_input;
  is_vcsgc = is_vcsgc_input;
  species = species_input;
  types = types_input;
  num_atoms_species = num_atoms_species_input;
  mu_or_phi = mu_or_phi_input;
  target_concentration = target_conc_input;
  kappa = kappa_input;
  int host_domains[3] = { Nx_domains, Ny_domains, Nz_domains };

  CHECK(gpuMemcpy(nep_energy_fast.nep_data.num_domains.data(), 
                  host_domains, 
                  3 * sizeof(int), 
                  gpuMemcpyHostToDevice)); 

  NN_ij.resize(N_domains);
  NL_ij.resize(n_max * N_domains);
  NN_angular_i.resize(N_domains);
  pe_nep_before_local.resize(n_max * N_domains);
  delta_pe_nep.resize(n_max * N_domains);
  delta_pe_zbl.resize(n_max * N_domains);
  t2_radial.resize(n_max * N_domains);
  x12_radial.resize(n_max * N_domains);   
  y12_radial.resize(n_max * N_domains);   
  z12_radial.resize(n_max * N_domains);   
  is_neigh_angular.resize(n_max * N_domains);
  g_is_accepted.resize(N_domains);

}

MC_Ensemble_SGC_fastNEP::~MC_Ensemble_SGC_fastNEP(void) { mc_output.close(); }

static __global__ void get_neighbors_of_i_fastNEP(
  const int n_max,
  const int N,
  const Box box,
  const int* i,
  const int N_domains,
  const float rc_radial_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  int* g_NN_i,
  int* g_NL_i,
  float* g_pe_nep_before,
  float* g_pe_nep_before_local)
{
  int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n < N) {
    double x0 = g_x[n];
    double y0 = g_y[n];
    double z0 = g_z[n];
    double pe_n = g_pe_nep_before[n];

    for (int domain = 0; domain < N_domains; ++domain){
      int target_idx = i[domain];
      if (target_idx < 0){ continue;}
      if (n != target_idx){

        double x0i = g_x[target_idx] - x0;
        double y0i = g_y[target_idx] - y0;
        double z0i = g_z[target_idx] - z0;

        apply_mic(box, x0i, y0i, z0i);
        float distance_square_i = float(x0i * x0i + y0i * y0i + z0i * z0i);

        if (distance_square_i < rc_radial_square) {
          int k = atomicAdd(&g_NN_i[domain], 1);
          if (k < n_max) {
            int index = k * N_domains + domain;
            g_pe_nep_before_local[index] = pe_n; 
            g_NL_i[index] = n;
          }
        }        
      }
    }
  }
}

static __global__ void create_inputs_for_energy_calculator_fastNEP(
  NEP_Energy_fast::ParaMB paramb,
  const int* N_local,
  const int N_domains,
  const int* i,
  const int* atom_local,
  const Box box,
  const float rc_radial_square,
  const float rc_angular_square,
  const double* __restrict__ g_x,
  const double* __restrict__ g_y,
  const double* __restrict__ g_z,
  const int* g_type,
  int* g_NN_angular,
  int* g_t2_radial,
  float* g_x12_radial,
  float* g_y12_radial,
  float* g_z12_radial,
  bool* g_is_neigh_angular,
  float* g_q_radial,
  float* g_s_angular,
  float* g_q_radial_local,
  float* g_s_angular_local)
{
  int thread = blockIdx.x * blockDim.x + threadIdx.x; 
  int k = thread/N_domains; // neighbors of the swapped atom i
  int domain = thread%N_domains;
  if (i[domain] < 0) {return;}
  if (k<N_local[domain]) {
    int n1 = atom_local[thread];
    double x2 = g_x[n1];
    double y2 = g_y[n1];
    double z2 = g_z[n1];
    double x12 = x2 - g_x[i[domain]];
    double y12 = y2 - g_y[i[domain]];
    double z12 = z2 - g_z[i[domain]];
    apply_mic(box, x12, y12, z12);
    float distance_square = float(x12 * x12 + y12 * y12 + z12 * z12);
    if (distance_square < rc_radial_square) {
      g_t2_radial[thread] = g_type[n1];
      g_x12_radial[thread] = float(x12);
      g_y12_radial[thread] = float(y12);
      g_z12_radial[thread] = float(z12);
      
      for (int n = 0; n <= paramb.n_max_radial; ++n){
        int index, index_local;
        index_local = (k*(paramb.n_max_radial+1) + n)*N_domains + domain;
        index = n1*(paramb.n_max_radial+1) + n;
        g_q_radial_local[index_local] = g_q_radial[index];
      }
      
      for (int n = 0; n <= paramb.n_max_angular; ++n){
        for (int l = 0; l<NUM_OF_ABC; ++l){
          int index, index_local;
          index_local = (k*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l) * N_domains + domain;
          index =      n1*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          g_s_angular_local[index_local] = g_s_angular[index];
        }
      }
      if (distance_square < rc_angular_square) {
        g_is_neigh_angular[thread] = true;
      }
      else {
        g_is_neigh_angular[thread] = false;
      }
    }
  }
  else if (k == N_local[domain]){// central (i) atom
    for (int n = 0; n <= paramb.n_max_radial; ++n){
        int index, index_local;
        index_local = (k*(paramb.n_max_radial+1) + n) * N_domains + domain;
        index = i[domain]*(paramb.n_max_radial+1) + n;
        g_q_radial_local[index_local] = g_q_radial[index];
      }

    for (int n = 0; n <= paramb.n_max_angular; ++n){
        for (int l = 0; l<NUM_OF_ABC; ++l){
          int index, index_local;
          index_local = (k*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l) * N_domains + domain;
          index =      i[domain]*(paramb.n_max_angular+1)*NUM_OF_ABC + n*NUM_OF_ABC + l;
          g_s_angular_local[index_local] = g_s_angular[index];
        }
      }
  }
}


// a kernel with a single thread <<<1, 1>>>
static __global__ void gpu_flip(
  const int i,
  const int type_j,
  const double mass_j,
  const double mass_scaler,
  int* g_type,
  double* g_mass,
  double* g_vx,
  double* g_vy,
  double* g_vz)
{
  g_type[i] = type_j;
  g_mass[i] = mass_j;
  g_vx[i] *= mass_scaler; // momentum conservation
  g_vy[i] *= mass_scaler;
  g_vz[i] *= mass_scaler;
}

bool MC_Ensemble_SGC_fastNEP::allowed_species(std::string& species_found)
{
  for (int k = 0; k < species.size(); ++k) {
    if (species[k] == species_found) {
      _index_old_species = k;
      return true;
    }
  }
  return false;
}

void MC_Ensemble_SGC_fastNEP::compute(
  int md_step,
  double temperature,
  Atom& atom,
  Box& box,
  std::vector<Group>& groups,
  int grouping_method,
  int group_id)
{
  if (check_if_small_box(nep_energy_fast.paramb.rc_radial, box)) {
    printf("Cannot use small box for MCMD.\n");
    exit(1);
  }



  int group_size =
    grouping_method >= 0 ? groups[grouping_method].cpu_size[group_id] : atom.number_of_atoms;

  nep_energy_fast.compute_large_box( // compute all descriptors and energy and save it to memeory (nep_data: pe_nep, q_radial, s_angular)
    box, 
    atom.type, 
    atom.position_per_atom);

  nep_energy_fast.find_subdomain_list(
    box,
    atom.position_per_atom);// in nep_energy_fast.nep_data.subdomain_class (0 - A, 1 - B, ... 7 - H)
    // TODO: реализовать случайное смещение начала отсчета чтобы убрать корреляции из-за границ доменов

  std::vector<int> subdomain_contents_cpu(nep_energy_fast.nep_data.subdomain_contents.size());
  nep_energy_fast.nep_data.subdomain_contents.copy_to_host(subdomain_contents_cpu.data());
  std::vector<int> subdomain_count_sum_cpu(nep_energy_fast.nep_data.subdomain_count_sum.size());
  nep_energy_fast.nep_data.subdomain_count_sum.copy_to_host(subdomain_count_sum_cpu.data());
  std::vector<int> subdomain_count_cpu(nep_energy_fast.nep_data.subdomain_count.size());
  nep_energy_fast.nep_data.subdomain_count.copy_to_host(subdomain_count_cpu.data());

  std::array<std::vector<int>, 8> subdomains;
  std::vector<std::uniform_int_distribution<int>> r1;
  r1.reserve(N_subdomains); // since we use emplace_back
  std::vector<bool> subdomain_is_empty(N_subdomains);

  for (int domain = 0; domain < N_subdomains; ++domain) {
    int size = subdomain_count_cpu[domain];
    if (size == 0) {
      r1.emplace_back(-1, -1);
      subdomain_is_empty[domain] = true;
    }
    else {
      r1.emplace_back(0, size - 1);
      subdomain_is_empty[domain] = false;
    }
    int color = color_of(domain);
    subdomains[color].push_back(domain);
  }
  for (int color=0; color<8; ++color){
    if (subdomains[color].size() != N_domains){
      printf("Wrong number of subdomains.\n");
      exit(1);
    }
  }

  int num_accepted = 0;
  int num_trials = static_cast<int>(std::round(swap_fraction_mc*double(group_size))); // change from total number of micro MC steps to fraction of the group size
  num_trials = ((num_trials - 1) / N_domains + 1);

  #ifdef FAST_NEP_MC_TEST
    mc_debug_log << "# VCSGC fastNEP; number of micro MC steps: " << num_trials << std::endl;
    mc_debug_log << "# domain; i; type_i; type_j; energy_difference" << std::endl;
  #endif

  std::uniform_int_distribution<int> color(0, 7);

  std::uniform_int_distribution<int> rand_int2(0, types.size() - 1);

  std::vector<int> i_cpu(N_domains, -1);
  std::vector<int> type_i_cpu(N_domains, -1);
  std::vector<int> type_j_cpu(N_domains);
  std::vector<int> index_new_species_cpu(N_domains);
  std::vector<int> index_old_species_cpu(N_domains);

  GPU_Vector<int> i(N_domains);
  GPU_Vector<int> type_i(N_domains);
  GPU_Vector<int> type_j(N_domains);
  GPU_Vector<int> index_new_species(N_domains);
  GPU_Vector<int> index_old_species(N_domains);

  for (int step = 0; step < num_trials; ++step) {
    NN_ij.fill(0);
    pe_nep_before_local.fill(0.0f);
    NN_angular_i.fill(0);
    nep_energy_fast.nep_data.q_radial_local.fill(0.0f);
    nep_energy_fast.nep_data.s_angular_local.fill(0.0f);
    is_neigh_angular.fill(false);
    t2_radial.fill(0);
    x12_radial.fill(0.0f);
    y12_radial.fill(0.0f);
    z12_radial.fill(0.0f);
    nep_energy_fast.nep_data.q_radial_trial_local.fill(0.0f);
    nep_energy_fast.nep_data.s_angular_trial_local.fill(0.0f);
    delta_pe_nep.fill(0.0f);
    delta_pe_zbl.fill(0.0f);

    int subdomain_color = color(rng);

    for (int domain = 0; domain < N_domains; ++domain){
      if (subdomain_is_empty[subdomains[subdomain_color][domain]]) 
      {
        i_cpu[domain] = -1;
        continue;
      }
      int subdomain = subdomains[subdomain_color][domain];
      std::string species_found;
      int count = 0;
      int max_tries = 1000000;
      while (!allowed_species(species_found)) {
        int local_ind = r1[subdomain](rng);
        int global_ind = subdomain_contents_cpu[subdomain_count_sum_cpu[subdomain] + local_ind];
        /*i_cpu[domain] = grouping_method >= 0
              ? groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + global_ind]
              : global_ind;
         */ //TODO сделать проход в find_subdomains_list только по группе
        i_cpu[domain] = global_ind;
        species_found = atom.cpu_atom_symbol[i_cpu[domain]];
        type_i_cpu[domain] = atom.cpu_type[i_cpu[domain]];
        if (++count > max_tries)
        {
          printf("[WARNING] Cannot suggest trial step for current domain: number of allowed species in MC is zero in the domain or very small! Skip MC step in this domain");
          i_cpu[domain] = -1;
          subdomain_is_empty[subdomains[subdomain_color][domain]] = true;
          break;
        }
      }
      index_old_species_cpu[domain] = _index_old_species;
      type_j_cpu[domain] = type_i_cpu[domain]; 
      while (type_j_cpu[domain] == type_i_cpu[domain]) {
        index_new_species_cpu[domain] = rand_int2(rng);
        type_j_cpu[domain] = types[index_new_species_cpu[domain]];
      }
    }

    i.copy_from_host(i_cpu.data());
    type_i.copy_from_host(type_i_cpu.data());
    type_j.copy_from_host(type_j_cpu.data());
    index_new_species.copy_from_host(index_new_species_cpu.data());
    index_old_species.copy_from_host(index_old_species_cpu.data());


    get_neighbors_of_i_fastNEP<<<(atom.number_of_atoms - 1) / 64 + 1, 64>>>(// TODO: переписать на быстрый вариант с cells
      n_max,
      atom.number_of_atoms,
      box,
      i.data(),
      N_domains,
      nep_energy_fast.paramb.rc_radial * nep_energy_fast.paramb.rc_radial,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms,
      atom.position_per_atom.data() + atom.number_of_atoms * 2,
      NN_ij.data(),
      NL_ij.data(),
      nep_energy_fast.nep_data.pe_nep.data(),
      pe_nep_before_local.data());
    GPU_CHECK_KERNEL



    std::vector<int> NN_ij_cpu(N_domains);
    NN_ij.copy_to_host(NN_ij_cpu.data(), N_domains);

    int n_max_actual = *std::max_element(NN_ij_cpu.begin(), NN_ij_cpu.end());
    if (n_max_actual + 1 > n_max) 
    {
      printf("[ERROR]: Number of neighbors exceed maximum!");
      exit(1);
    }

    for (int domain = 0; domain < N_domains; domain++){// TODO может быть стоит завести один массив для энергий центральных атомов или изменить его структуру - писать все в последнюю строчку (так же быстрее копирование?)
      if (subdomain_is_empty[subdomains[subdomain_color][domain]]) {continue;}
      CHECK(gpuMemcpy(pe_nep_before_local.data() + NN_ij_cpu[domain]*N_domains + domain, &nep_energy_fast.nep_data.pe_nep.data()[i_cpu[domain]], sizeof(float), gpuMemcpyDeviceToDevice)); 
    }
    

    create_inputs_for_energy_calculator_fastNEP<<<((n_max_actual + 1) * N_domains - 1) / 64 + 1, 64>>>(
      nep_energy_fast.paramb,
      NN_ij.data(),
      N_domains,
      i.data(),
      NL_ij.data(),
      box,
      nep_energy_fast.paramb.rc_radial * nep_energy_fast.paramb.rc_radial,
      nep_energy_fast.paramb.rc_angular * nep_energy_fast.paramb.rc_angular,
      atom.position_per_atom.data(),
      atom.position_per_atom.data() + atom.number_of_atoms, 
      atom.position_per_atom.data() + atom.number_of_atoms * 2, 
      atom.type.data(),
      NN_angular_i.data(),
      t2_radial.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      is_neigh_angular.data(),
      nep_energy_fast.nep_data.q_radial.data(),
      nep_energy_fast.nep_data.s_angular.data(),
      nep_energy_fast.nep_data.q_radial_local.data(),
      nep_energy_fast.nep_data.s_angular_local.data());
    GPU_CHECK_KERNEL
  
    
    const int N = (n_max_actual + 1) * N_domains;

    nep_energy_fast.find_energy( // calculate energy differences for neighbors of th central (i) atom after MC trial swap: delta_pe_nep, delta_pe_zbl
      N_domains,
      NN_ij.data(),
      N,
      i.data(),
      NN_angular_i.data(),
      type_i.data(),
      type_j.data(),
      t2_radial.data(),
      x12_radial.data(),
      y12_radial.data(),
      z12_radial.data(),
      is_neigh_angular.data(),
      delta_pe_nep.data(),
      pe_nep_before_local.data(),
      delta_pe_zbl.data());

    
    
    std::vector<float> delta_pe_nep_cpu(N);
    delta_pe_nep.copy_to_host(delta_pe_nep_cpu.data(), N);

    std::vector<float> delta_pe_zbl_cpu(N);
    delta_pe_zbl.copy_to_host(delta_pe_zbl_cpu.data(), N);

    std::vector<float> energy_difference(N_domains, 0.0f);
    for (int domain = 0; domain < N_domains; ++domain) {
      if (subdomain_is_empty[subdomains[subdomain_color][domain]]) {continue;}
      for (int n = 0; n <= NN_ij_cpu[domain]; ++n){
        energy_difference[domain] += delta_pe_nep_cpu[n*N_domains + domain];
        energy_difference[domain] += delta_pe_zbl_cpu[n*N_domains + domain];
      }
    }


    #ifdef FAST_NEP_MC_TEST
      for (int domain = 0; domain < N_domains; ++domain){
        if (subdomain_is_empty[subdomains[subdomain_color][domain]]) {
          mc_debug_log << domain << "; empty" << std::endl;
        }
        else {
          mc_debug_log << domain << "; " << i_cpu[domain] << "; " << type_i_cpu[domain] << "; " << type_j_cpu[domain] << "; " << energy_difference[domain]  << std::endl;
        }
      }
    #endif

    std::uniform_real_distribution<float> r2(0, 1);
    std::vector<bool> is_accepted(N_domains);
    bool global_acceptance = true;

    for (int domain = 0; domain < N_domains; ++domain){
      if (subdomain_is_empty[subdomains[subdomain_color][domain]])
      {
        is_accepted[domain] = false; 
        continue;
      }
      energy_difference[domain] += mu_or_phi[index_new_species_cpu[domain]] - mu_or_phi[index_old_species_cpu[domain]];
      if (is_vcsgc){ // TODO: кажется здесь ерунда щас написана
        energy_difference[domain] += 2*kappa*num_atoms*(target_concentration[index_new_species_cpu[domain]]-target_concentration[index_old_species_cpu[domain]]);
      }

      float random_number = r2(rng);
      float probability = exp(-energy_difference[domain] / (K_B * temperature));
      #ifdef FAST_NEP_MC_TEST
        probability = 0.5;
      #endif
      if (!(random_number < probability)) {
        is_accepted[domain] = false;
        energy_difference[domain] = 0.0f;
      }
      else {
        is_accepted[domain] = true;
      }
    }

    if (is_vcsgc) {
      float exponent = 0.0f;
      for (int domain = 0; domain < N_domains; ++domain){
        if (subdomain_is_empty[subdomains[subdomain_color][domain]]){continue;}
        exponent += kappa*num_atoms * (num_atoms_species[index_new_species_cpu[domain]]+1.0-target_concentration[index_new_species_cpu[domain]]); // TODO посчитать что там должно быть
        exponent += kappa*num_atoms * (num_atoms_species[index_old_species_cpu[domain]]-1.0-target_concentration[index_old_species_cpu[domain]]); // TODO посчитать что там должно быть
      }
      float random_number = r2(rng);
      float probability = exp(-exponent / (K_B * temperature));
      global_acceptance = (random_number < probability);
    }
    
    if (global_acceptance) {
      for (int domain = 0; domain < N_domains; ++domain){
        if (subdomain_is_empty[subdomains[subdomain_color][domain]]){continue;}
        if (is_accepted[domain]){
          ++num_accepted;
          ++num_atoms_species[index_new_species_cpu[domain]];
          --num_atoms_species[index_old_species_cpu[domain]];

          atom.cpu_type[i_cpu[domain]] = type_j_cpu[domain];
          atom.cpu_atom_symbol[i_cpu[domain]] = species[index_new_species_cpu[domain]];
          double mass_old = atom.cpu_mass[i_cpu[domain]];
          double mass_new = MASS_TABLE.at(species[index_new_species_cpu[domain]]);
          atom.cpu_mass[i_cpu[domain]] = mass_new;

          gpu_flip<<<1, 1>>>(
            i_cpu[domain],// TODO использовать device
            type_j_cpu[domain],// TODO использовать device
            mass_new,
            mass_old / mass_new,
            atom.type.data(),
            atom.mass.data(),
            atom.velocity_per_atom.data(),
            atom.velocity_per_atom.data() + atom.number_of_atoms,
            atom.velocity_per_atom.data() + atom.number_of_atoms * 2);
        }
      }

      std::vector<char> temp_accepted(is_accepted.begin(), is_accepted.end());
      g_is_accepted.copy_from_host(reinterpret_cast<const bool*>(temp_accepted.data()), N_domains);

      nep_energy_fast.accept_trial( // update stored values of descriptors and potential energies for NEP
        N,
        N_domains,
        NN_ij.data(), 
        g_is_accepted.data(),
        NL_ij.data(),
        delta_pe_nep.data(),
        i.data());
    }
  }

  mc_output << md_step << "  " << num_accepted / double(num_trials*N_domains) << " ";
  for (int t = 0; t < types.size(); ++t) {
    mc_output << num_atoms_species[t] / double(atom.number_of_atoms) << " ";
  }
  mc_output << std::endl;
}
