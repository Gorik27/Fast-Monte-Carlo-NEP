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
The driver class for the various MC ensembles.
------------------------------------------------------------------------------*/

#include "mc.cuh"
#include "mc_ensemble_canonical.cuh"
#include "mc_ensemble_sgc.cuh"
#include "mc_ensemble_sgc_fastNEP.cuh"
#include "model/atom.cuh"
#include "utilities/common.cuh"
#include "utilities/gpu_macro.cuh"
#include "utilities/read_file.cuh"
#include <algorithm>
#include <cmath>
#include <cstring>

void MC::initialize(void)
{
  #ifdef FAST_NEP_MC_TEST
    printf("FAST_NEP_MC_TEST is on: the MC SGC ensemble uses a constant acceptance probability (p=0.5), and debug files are generated to compare the fastNEP and original implementations.\n");
    #ifndef DEBUG
      printf("WARNING: FAST_NEP_MC_TEST should only be used with the DEBUG flag enabled.\n");
    #endif
  #endif
  // todo
}

void MC::finalize(void) { do_mcmd = false; }

#ifdef MC_TEST
namespace
{
// Complete state of the atoms that can be changed by an MC step
struct MC_TestSnapshot {
  std::vector<int> cpu_type;
  std::vector<std::string> cpu_symbol;
  std::vector<double> cpu_mass;
  std::vector<int> gpu_type;
  std::vector<double> gpu_mass;
  std::vector<double> gpu_velocity;

  void save(Atom& atom)
  {
    cpu_type = atom.cpu_type;
    cpu_symbol = atom.cpu_atom_symbol;
    cpu_mass = atom.cpu_mass;
    gpu_type.resize(atom.type.size());
    gpu_mass.resize(atom.mass.size());
    gpu_velocity.resize(atom.velocity_per_atom.size());
    atom.type.copy_to_host(gpu_type.data());
    atom.mass.copy_to_host(gpu_mass.data());
    atom.velocity_per_atom.copy_to_host(gpu_velocity.data());
  }

  void restore(Atom& atom) const
  {
    atom.cpu_type = cpu_type;
    atom.cpu_atom_symbol = cpu_symbol;
    atom.cpu_mass = cpu_mass;
    atom.type.copy_from_host(gpu_type.data());
    atom.mass.copy_from_host(gpu_mass.data());
    atom.velocity_per_atom.copy_from_host(gpu_velocity.data());
  }
};
} // namespace

// One MC call of the parallel implementation, checked against the original implementation:
//  1. save the state, run the parallel MC (all trial swaps are logged in mc_test_steps);
//  2. save the resulting state, restore the initial state;
//  3. replay the logged swaps with the original SGC code: for every swap the energy change is
//     recomputed from scratch and compared with the logged one, accepted swaps are applied;
//  4. compare the final state of the replay with the result of the parallel run.
// The result of the parallel run is kept, so that the MD continues as without the test.
void MC::run_test(
  int md_step, double temperature, Atom& atom, Box& box, std::vector<Group>& group)
{
  MC_TestSnapshot before, after, replayed;
  before.save(atom);

  mc_ensemble->mc_test_steps.clear();
  mc_ensemble->compute(md_step, temperature, atom, box, group, grouping_method, group_id);
  const std::vector<MC_TestStep> steps = mc_ensemble->mc_test_steps;
  after.save(atom);
  const MC_TestCacheReport cache = mc_ensemble->test_check_cache(atom, box);

  before.restore(atom);
  MC_TestReport report =
    mc_ensemble_ref->replay_steps(steps, atom, box, MC_TEST_ATOL, MC_TEST_RTOL);
  replayed.save(atom);

  // final states must be identical (types, symbols, masses)
  int num_state_diff = 0;
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    if (replayed.cpu_type[n] != after.cpu_type[n] || replayed.cpu_symbol[n] != after.cpu_symbol[n] ||
        replayed.cpu_mass[n] != after.cpu_mass[n] || replayed.gpu_type[n] != after.gpu_type[n]) {
      ++num_state_diff;
    }
  }
  // the host and the device copies of the types must agree as well
  int num_host_device_diff = 0;
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    if (after.cpu_type[n] != after.gpu_type[n]) {
      ++num_host_device_diff;
    }
  }

  // number of atoms of every species: own counters of both ensembles vs real count
  std::vector<int> counted(species.size(), 0);
  for (int n = 0; n < atom.number_of_atoms; ++n) {
    for (size_t s = 0; s < species.size(); ++s) {
      if (after.cpu_symbol[n] == species[s]) {
        ++counted[s];
      }
    }
  }
  const std::vector<int> counts_par = mc_ensemble->test_num_atoms_species();
  const std::vector<int> counts_ref = mc_ensemble_ref->test_num_atoms_species();
  int num_count_fail = 0;
  for (size_t s = 0; s < species.size(); ++s) {
    if (counts_par.size() == species.size() && counts_par[s] != counted[s]) {
      ++num_count_fail;
    }
    if (counts_ref.size() == species.size() && counts_ref[s] != counted[s]) {
      ++num_count_fail;
    }
  }

  // momentum of every atom (m * v) must not change when the species (mass) is changed
  const int N_atoms = atom.number_of_atoms;
  double max_dp = 0.0;
  double p_scale = 0.0;
  for (int n = 0; n < N_atoms; ++n) {
    for (int c = 0; c < 3; ++c) {
      const double p_before = before.gpu_mass[n] * before.gpu_velocity[n + c * N_atoms];
      const double p_after = after.gpu_mass[n] * after.gpu_velocity[n + c * N_atoms];
      max_dp = std::max(max_dp, std::fabs(p_after - p_before));
      p_scale = std::max(p_scale, std::fabs(p_before));
    }
  }
  const bool momentum_ok = max_dp <= 1.0e-9 * p_scale;
  const bool cache_ok =
    !cache.checked || (cache.num_pe_fail == 0 && cache.num_q_fail == 0 && cache.num_s_fail == 0);

  const bool ok = report.num_energy_fail == 0 && report.num_type_fail == 0 &&
                  num_state_diff == 0 && num_host_device_diff == 0 && cache_ok &&
                  num_count_fail == 0 && momentum_ok;
  if (!ok) {
    ++test_num_bad_calls;
  }

  printf(
    "[MC_TEST] md_step %d: %s | steps %d, applied %d | energy fails %d, type fails %d | "
    "max |dE diff| = %.3e eV (rel %.3e) | final state differences %d | host/device type "
    "differences %d | bad calls so far: %d\n",
    md_step,
    ok ? "PASSED" : "FAILED",
    report.num_steps,
    report.num_applied,
    report.num_energy_fail,
    report.num_type_fail,
    report.max_abs_err,
    report.max_rel_err,
    num_state_diff,
    num_host_device_diff,
    test_num_bad_calls);
  printf(
    "[MC_TEST]   cache: %s (pe fails %d, max %.3e eV, sum diff %.3e eV; q fails %d, max %.3e; "
    "s fails %d, max %.3e) | species counters: %s | momentum: %s (max |dp| %.3e, scale %.3e)\n",
    !cache.checked ? "not checked" : (cache_ok ? "OK" : "FAILED"),
    cache.num_pe_fail,
    cache.max_pe_err,
    cache.total_pe_diff,
    cache.num_q_fail,
    cache.max_q_err,
    cache.num_s_fail,
    cache.max_s_err,
    num_count_fail == 0 ? "OK" : "FAILED",
    momentum_ok ? "OK" : "FAILED",
    max_dp,
    p_scale);
  printf(
    "[MC_TEST]   max |dE diff| by fifth of the MC call (drift check): %.2e %.2e %.2e %.2e %.2e eV\n",
    report.max_abs_err_bin[0],
    report.max_abs_err_bin[1],
    report.max_abs_err_bin[2],
    report.max_abs_err_bin[3],
    report.max_abs_err_bin[4]);

  // continue MD from the result of the parallel run
  after.restore(atom);
}
#endif

void MC::compute(int step, int num_steps, Atom& atom, Box& box, std::vector<Group>& group)
{
  if (do_mcmd) {
    if ((step + 2) % num_steps_md == 0) {
      double temperature =
        temperature_initial + step * (temperature_final - temperature_initial) / num_steps;
#ifdef MC_TEST
      if (mc_ensemble_ref) {
        run_test(step + 2, temperature, atom, box, group);
      } else {
        mc_ensemble->compute(step + 2, temperature, atom, box, group, grouping_method, group_id);
      }
#else
      mc_ensemble->compute(step + 2, temperature, atom, box, group, grouping_method, group_id);
#endif
    }
  }
}

void MC::parse_group(
  const char** param, int num_param, std::vector<Group>& groups, int num_param_before_group)
{
  if (strcmp(param[num_param_before_group], "group") != 0) {
    PRINT_INPUT_ERROR("invalid option for mc.\n");
  }
  if (!is_valid_int(param[num_param_before_group + 1], &grouping_method)) {
    PRINT_INPUT_ERROR("grouping method of MCMD should be an integer.\n");
  }
  if (grouping_method < 0) {
    PRINT_INPUT_ERROR("grouping method of MCMD should >= 0.\n");
  }
  if (grouping_method >= groups.size()) {
    PRINT_INPUT_ERROR("Grouping method should < number of grouping methods.");
  }
  if (!is_valid_int(param[num_param_before_group + 2], &group_id)) {
    PRINT_INPUT_ERROR("group ID of MCMD should be an integer.\n");
  }
  if (group_id < 0) {
    PRINT_INPUT_ERROR("group ID of MCMD should >= 0.\n");
  }
  if (group_id >= groups[grouping_method].number) {
    PRINT_INPUT_ERROR("Group ID should < number of groups.");
  }
}

void MC::check_species_canonical(std::vector<Group>& groups, Atom& atom)
{
  bool has_multi_types = false;
  int type0 = 0;

  if (grouping_method < 0) {
    for (int n = 0; n < atom.number_of_atoms; ++n) {
      if (n == 0) {
        type0 = atom.cpu_type[n];
      } else {
        if (atom.cpu_type[n] != type0) {
          has_multi_types = true;
          break;
        }
      }
    }
  } else {
    for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
      int n =
        groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
      if (k == 0) {
        type0 = atom.cpu_type[n];
      } else {
        if (atom.cpu_type[n] != type0) {
          has_multi_types = true;
          break;
        }
      }
    }
  }

  if (!has_multi_types) {
    PRINT_INPUT_ERROR("Must have more than one atom type for canonical MCMD.");
  }
}

static std::string get_potential_file_name()
{
  std::ifstream input_run("run.in");
  if (!input_run.is_open()) {
    PRINT_INPUT_ERROR("Cannot open run.in.");
  }
  std::string potential_file_name;
  std::string line;
  while (std::getline(input_run, line)) {
    std::vector<std::string> tokens = get_tokens(line);
    if (tokens.size() != 0) {
      if (tokens[0] == "potential") {
        potential_file_name = tokens[1];
        break;
      }
    }
  }

  input_run.close();
  return potential_file_name;
}

static std::vector<std::string> get_atom_symbols_in_nep()
{
  auto potential_file_name = get_potential_file_name();
  std::ifstream input_potential(potential_file_name);
  if (!input_potential.is_open()) {
    PRINT_INPUT_ERROR("Cannot open potential file.");
  }
  std::string line;
  std::getline(input_potential, line);
  std::vector<std::string> tokens = get_tokens(line);
  if (tokens[0].substr(0, 3) != "nep") {
    PRINT_INPUT_ERROR("MCMD only supports NEP models.");
  }

  int num_types = get_int_from_token(tokens[1], __FILE__, __LINE__);
  std::vector<std::string> atom_symbols_in_nep(num_types);
  for (int n = 0; n < num_types; ++n) {
    atom_symbols_in_nep[n] = tokens[2 + n];
  }
  input_potential.close();
  return atom_symbols_in_nep;
}

void MC::check_species_sgc(std::vector<Group>& groups, Atom& atom)
{
  auto atom_symbols_in_nep = get_atom_symbols_in_nep();
  for (int s = 0; s < species.size(); ++s) {
    bool allowed_species = false;
    for (int n = 0; n < atom_symbols_in_nep.size(); ++n) {
      if (species[s] == atom_symbols_in_nep[n]) {
        allowed_species = true;
        types[s] = n;
        break;
      }
    }
    if (!allowed_species) {
      PRINT_INPUT_ERROR("There are listed species not allowed in the NEP model.");
    }
  }

  if (grouping_method < 0) {
    for (int s = 0; s < species.size(); ++s) {
      for (int n = 0; n < atom.number_of_atoms; ++n) {
        if (atom.cpu_atom_symbol[n] == species[s]) {
          ++num_atoms_species[s];
        }
      }
    }
  } else {
    for (int s = 0; s < species.size(); ++s) {
      for (int k = 0; k < groups[grouping_method].cpu_size[group_id]; ++k) {
        int n =
          groups[grouping_method].cpu_contents[groups[grouping_method].cpu_size_sum[group_id] + k];
        if (atom.cpu_atom_symbol[n] == species[s]) {
          ++num_atoms_species[s];
        }
      }
    }
  }

  printf("    the initial system or specified group has:\n");
  bool has_at_least_one_species = false;
  for (int s = 0; s < species.size(); ++s) {
    printf("        %d %s atoms.\n", num_atoms_species[s], species[s].c_str());
    if (num_atoms_species[s] > 0) {
      has_at_least_one_species = true;
    }
  }

  if (!has_at_least_one_species) {
    PRINT_INPUT_ERROR(
      "Must have at least one listed species in the initial model system or specified group.");
  }
}

void MC::parse_mc(const char** param, int num_param, std::vector<Group>& groups, Atom& atom)
{
  if (num_param < 6) {
    PRINT_INPUT_ERROR("mc should have at least 5 parameters.\n");
  }

  int mc_ensemble_type = 0;
  if (strcmp(param[1], "canonical") == 0) {
    printf("Perform canonical MCMD:\n");
    mc_ensemble_type = 0;
  } else if (strcmp(param[1], "sgc") == 0) {
    printf("Perform SGC MCMD:\n");
    mc_ensemble_type = 1;
  } else if (strcmp(param[1], "vcsgc") == 0) {
    printf("Perform VCSGC MCMD:\n");
    mc_ensemble_type = 2;
  } else if (strcmp(param[1], "sgc_fast") == 0) {
    printf("Perform SGC MCMD using the fast implementation:\n");
    mc_ensemble_type = 3;
  } else if (strcmp(param[1], "vcsgc_fast") == 0) {
    printf("Perform VCSGC MCMD using the fast implementation:\n");
    mc_ensemble_type = 4;
  } else if (strcmp(param[1], "sgc_par") == 0) {
    printf("Perform SGC MCMD using the parallel implementation:\n");
    mc_ensemble_type = 5;
  } else if (strcmp(param[1], "vcsgc_par") == 0) {
    printf("Perform VCSGC MCMD using the parallel implementation:\n");
    mc_ensemble_type = 6;
  } else {
    PRINT_INPUT_ERROR("invalid MC ensemble for MCMD.\n");
  }

  if (!is_valid_int(param[2], &num_steps_md)) {
      PRINT_INPUT_ERROR("number of MD steps for MCMD should be an integer.\n");
    }
  if (num_steps_md <= 0) {
      PRINT_INPUT_ERROR("number of MD steps for MCMD should be positive.\n");
    }

  if (mc_ensemble_type >=0 && mc_ensemble_type <=2) {
    if (!is_valid_int(param[3], &num_steps_mc)) {
      PRINT_INPUT_ERROR("number of MC steps for MCMD should be an integer.\n");
    }
    if (num_steps_mc <= 0) {
      PRINT_INPUT_ERROR("number of MC steps for MCMD should be positive.\n");
    }
    printf("    after every %d MD steps, do %d MC trials.\n", num_steps_md, num_steps_mc);
  }
  else if (mc_ensemble_type >=3 && mc_ensemble_type <=6){
    if (!is_valid_real(param[3], &swap_fraction_mc)) {
      PRINT_INPUT_ERROR("swap fraction for MC steps for MCMD should be a number.\n");
    }
    if (swap_fraction_mc <= 0) {
      PRINT_INPUT_ERROR("swap fraction for MC steps for MCMD should be positive.\n");
    }
    printf("    after every %d MD steps, swap %g atoms via MC.\n", num_steps_md, swap_fraction_mc);
  }

  if (!is_valid_real(param[4], &temperature_initial)) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be a number.\n");
  }
  if (temperature_initial <= 0) {
    PRINT_INPUT_ERROR("initial temperature for MCMD should be positive.\n");
  }

  if (!is_valid_real(param[5], &temperature_final)) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be a number.\n");
  }
  if (temperature_final <= 0) {
    PRINT_INPUT_ERROR("final temperature for MCMD should be positive.\n");
  }

  printf(
    "    with an initial temperature of %g K and a final temperature of %g K.\n",
    temperature_initial,
    temperature_final);

  if (mc_ensemble_type == 1 || mc_ensemble_type == 2 || mc_ensemble_type == 3 || mc_ensemble_type == 4) {
    if (num_param < 7) {
      PRINT_INPUT_ERROR("reading error for num_types in SGC/VCSGC MCMD.\n");
    }
    if (!is_valid_int(param[6], &num_types_mc)) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be an integer.\n");
    }
    if (num_types_mc < 2 || num_types_mc > 4) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be 2 to 4.\n");
    }
    printf("    number of species involved in SGC/VCSGC = %d.\n", num_types_mc);

    if (num_param < (7 + num_types_mc * 2)) {
      PRINT_INPUT_ERROR("not enough (species, mu) or (species, phi) inputs.\n");
    }

    species.resize(num_types_mc);
    mu_or_phi.resize(num_types_mc);
    for (int n = 0; n < num_types_mc; ++n) {
      species[n] = param[7 + n * 2];
      if (!is_valid_real(param[7 + n * 2 + 1], &mu_or_phi[n])) {
        PRINT_INPUT_ERROR("mu or phi should be a number.\n");
      }
      printf("        species = %s, mu/phi= %g\n", species[n].c_str(), mu_or_phi[n]);
    }
  }

  if (mc_ensemble_type == 5) {
    if (num_param < 10) {
      PRINT_INPUT_ERROR("reading error for num_types in SGC/VCSGC MCMD.\n");
    }
    if (!is_valid_int(param[6], &num_types_mc)) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be an integer.\n");
    }
    if (num_types_mc < 2 || num_types_mc > 4) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be 2 to 4.\n");
    }
    printf("    number of species involved in SGC/VCSGC = %d.\n", num_types_mc);

    if (num_param < (10 + num_types_mc * 2)) {
      PRINT_INPUT_ERROR("not enough (species, mu) or (species, phi) inputs.\n");
    }

    species.resize(num_types_mc);
    mu_or_phi.resize(num_types_mc);
    target_conc.resize(num_types_mc, 0.0);
    for (int n = 0; n < num_types_mc; ++n) {
      species[n] = param[7 + n * 2];
      if (!is_valid_real(param[7 + n * 2 + 1], &mu_or_phi[n])) {
        PRINT_INPUT_ERROR("mu or phi should be a number.\n");
      }
      printf("        species = %s, mu/phi= %g\n", species[n].c_str(), mu_or_phi[n]);
    }
  }

  if (mc_ensemble_type == 6) {
    if (num_param < 10) {
      PRINT_INPUT_ERROR("reading error for num_types in SGC/VCSGC MCMD.\n");
    }
    if (!is_valid_int(param[6], &num_types_mc)) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be an integer.\n");
    }
    if (num_types_mc < 2 || num_types_mc > 4) {
      PRINT_INPUT_ERROR("number of types in SGC/VCSGC MC trials should be 2 to 4.\n");
    }
    printf("    number of species involved in SGC/VCSGC = %d.\n", num_types_mc);

    if (num_param < (10 + num_types_mc * 3)) {
      PRINT_INPUT_ERROR("not enough (species, mu) or (species, phi) inputs.\n");
    }

    species.resize(num_types_mc);
    mu_or_phi.resize(num_types_mc);
    target_conc.resize(num_types_mc);
    for (int n = 0; n < num_types_mc; ++n) {
      species[n] = param[7 + n * 3];
      if (!is_valid_real(param[7 + n * 3 + 1], &mu_or_phi[n])) {
        PRINT_INPUT_ERROR("mu or phi should be a number.\n");
      }
      if (!is_valid_real(param[7 + n * 3 + 2], &target_conc[n])) {
        PRINT_INPUT_ERROR("target concentration should be a number.\n");
      }
      if (target_conc[n] < 0.0 || target_conc[n] > 1.0){
        PRINT_INPUT_ERROR("target concentration should be from 0 to 1.\n");
      }
      printf("        species = %s, mu/phi= %g, c0= %g\n", species[n].c_str(), mu_or_phi[n], target_conc[n]);
    }
  }

  if (mc_ensemble_type == 2 || mc_ensemble_type == 4) {
    if (num_param < 7 + num_types_mc * 2 + 1) {
      PRINT_INPUT_ERROR("Should have kappa for VCSGC.\n");
    }
    if (!is_valid_real(param[7 + num_types_mc * 2], &kappa)) {
      PRINT_INPUT_ERROR("kappa should be a number.\n");
    }
    if (kappa < 0) {
      PRINT_INPUT_ERROR("kappa should be positive.\n");
    }
    printf("    kappa = %g\n", kappa);
  }

  int Nx_domains = 1;
  int Ny_domains = 1;
  int Nz_domains = 1;
  if (mc_ensemble_type == 6) {
    if (num_param < 10 + num_types_mc * 3 + 1) {
      PRINT_INPUT_ERROR("Should have kappa for VCSGC.\n");
    }
    if (!is_valid_real(param[7 + num_types_mc * 3], &kappa)) {
      PRINT_INPUT_ERROR("kappa should be a number.\n");
    }
    if (kappa < 0) {
      PRINT_INPUT_ERROR("kappa should be positive.\n");
    }
    printf("    kappa = %g\n", kappa);

    if (!is_valid_int(param[8 + num_types_mc * 3], &Nx_domains)) {
      PRINT_INPUT_ERROR("number of domains should be an integer.\n");
    }
    if (!is_valid_int(param[9 + num_types_mc * 3], &Ny_domains)) {
      PRINT_INPUT_ERROR("number of domains should be an integer.\n");
    }
    if (!is_valid_int(param[10 + num_types_mc * 3], &Nz_domains)) {
      PRINT_INPUT_ERROR("number of domains should be an integer.\n");
    }
    if (Nx_domains <= 0 || Ny_domains <= 0 || Nz_domains <= 0) {
      PRINT_INPUT_ERROR("number of domains should be positive.\n");
    }
    printf("    number of domains = (%d, %d, %d)\n", Nx_domains, Ny_domains, Nz_domains);
  }

  if (mc_ensemble_type == 5) {
    if (!is_valid_int(param[7 + num_types_mc * 2], &Nx_domains)) {
      PRINT_INPUT_ERROR("number of domains should be an integer.\n");
    }
    if (!is_valid_int(param[8 + num_types_mc * 2], &Ny_domains)) {
      PRINT_INPUT_ERROR("number of domains should be an integer.\n");
    }
    if (!is_valid_int(param[9 + num_types_mc * 2], &Nz_domains)) {
      PRINT_INPUT_ERROR("number of domains should be an integer.\n");
    }
    if (Nx_domains <= 0 || Ny_domains <= 0 || Nz_domains <= 0) {
      PRINT_INPUT_ERROR("number of domains should be positive.\n");
    }
    printf("    number of domains = (%d, %d, %d)\n", Nx_domains, Ny_domains, Nz_domains);
  }

  int num_param_before_group = 6;
  if (mc_ensemble_type == 1 || mc_ensemble_type == 3) {
    num_param_before_group = 7 + num_types_mc * 2;
  } else if (mc_ensemble_type == 2 || mc_ensemble_type == 4) {
    num_param_before_group = 8 + num_types_mc * 2;
  } else if (mc_ensemble_type == 5) { // TODO что это?
    num_param_before_group = 10 + num_types_mc * 2;
  } else if (mc_ensemble_type == 6) {
    num_param_before_group = 11 + num_types_mc * 3;
  }

  if (num_param > num_param_before_group) {
    if (num_param != num_param_before_group + 3) {
      PRINT_INPUT_ERROR("reading error grouping method.\n");
    }
    parse_group(param, num_param, groups, num_param_before_group);
    printf("    only for atoms in group %d of grouping method %d.\n", group_id, grouping_method);
  }

  if (mc_ensemble_type == 5 || mc_ensemble_type == 6){
    if (grouping_method >= 0){
      PRINT_INPUT_ERROR("grouping unsuported in MCMD parallel now! Groups will be ignored");
    }
  }

  types.resize(num_types_mc);
  num_atoms_species.resize(num_types_mc, 0);
  std::fill(num_atoms_species.begin(), num_atoms_species.end(), 0);
  if (mc_ensemble_type == 0) {
    check_species_canonical(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_Canonical(param, num_param, num_steps_mc));
  } else if (mc_ensemble_type == 1) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC(
      param, num_param, num_steps_mc, false, species, types, num_atoms_species, mu_or_phi, kappa));
  } else if (mc_ensemble_type == 2) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC(
      param, num_param, num_steps_mc, true, species, types, num_atoms_species, mu_or_phi, kappa));
  } else if (mc_ensemble_type == 3) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC_fastNEP(
      atom.number_of_atoms, 1, 1, 1, param, num_param, swap_fraction_mc, false, species, types, num_atoms_species, mu_or_phi, target_conc, kappa));
  } else if (mc_ensemble_type == 4) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC_fastNEP(
      atom.number_of_atoms, 1, 1, 1, param, num_param, swap_fraction_mc, true, species, types, num_atoms_species, mu_or_phi, target_conc, kappa));
  } else if (mc_ensemble_type == 5) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC_fastNEP(
      atom.number_of_atoms, Nx_domains, Ny_domains, Nz_domains, param, num_param, swap_fraction_mc, false, species, types, num_atoms_species, mu_or_phi, target_conc, kappa));
  } else if (mc_ensemble_type == 6) {
    check_species_sgc(groups, atom);
    mc_ensemble.reset(new MC_Ensemble_SGC_fastNEP(
      atom.number_of_atoms, Nx_domains, Ny_domains, Nz_domains, param, num_param, swap_fraction_mc, true, species, types, num_atoms_species, mu_or_phi, target_conc, kappa));
  }

#ifdef MC_TEST
  if (mc_ensemble_type >= 3) { // parallel/fast implementations are checked against the original SGC
    std::vector<int> num_atoms_species_ref = num_atoms_species;
    mc_ensemble_ref.reset(new MC_Ensemble_SGC(
      param, num_param, 1, false, species, types, num_atoms_species_ref, mu_or_phi, kappa));
    printf("MC_TEST is on: every MC call of the fast/parallel implementation is replayed with the original SGC implementation.\n");
  }
#endif

  do_mcmd = true;
}
