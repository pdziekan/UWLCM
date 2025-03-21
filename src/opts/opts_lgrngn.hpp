/** 
 * @file
 * @copyright University of Warsaw
 * @section LICENSE
 * GPLv3+ (see the COPYING file or http://www.gnu.org/licenses/)
 */

#pragma once

#include <boost/assign/ptr_map_inserter.hpp>  // for 'ptr_map_insert()'

#include "opts_common.hpp"

#include <libcloudph++/lgrngn/opts.hpp>
#include <libcloudph++/lgrngn/backend.hpp>
#include <libcloudph++/lgrngn/advection_scheme.hpp>
#include <libcloudph++/lgrngn/kernel.hpp>
#include <libcloudph++/lgrngn/terminal_velocity.hpp>
#include <libcloudph++/lgrngn/RH_formula.hpp>
#include <libcloudph++/lgrngn/ccn_source.hpp>

// string parsing
#include <boost/spirit/include/qi.hpp>    
#include <boost/fusion/adapted/std_pair.hpp> 

/*
  with boost 1.81 to 1.83 we get multiple definition of
  `boost::phoenix::placeholders::uargX` errors. A solution
  suggested in https://github.com/boostorg/phoenix/issues/111
  is to define BOOST_PHOENIX_STL_TUPLE_H_ so that
  boost/stl/tuple.h is not included in boost/phoenix.hpp
*/
#define BOOST_PHOENIX_STL_TUPLE_H_
#include <boost/spirit/include/phoenix_core.hpp>
#include <boost/spirit/include/phoenix_stl.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>

#include "../detail/outmom.hpp"
#include "../detail/subs_t.hpp"
#include <UWLCM/output_bins.hpp>

// simulation and output parameters for micro=lgrngn
template <class solver_t, class user_params_t, class case_ptr_t>
void setopts_micro(
  typename solver_t::rt_params_t &rt_params, 
  const user_params_t &user_params,
  const case_ptr_t &case_ptr,
  typename std::enable_if<std::is_same<
    decltype(solver_t::rt_params_t::cloudph_opts),
    libcloudphxx::lgrngn::opts_t<typename solver_t::real_t>
  >::value>::type* = 0
)
{
  using thrust_real_t = setup::real_t; // TODO: make it a choice?

  po::options_description opts("Lagrangian microphysics options"); 
  opts.add_options()
    ("backend", po::value<std::string>()->required() , "one of: CUDA, multi_CUDA, OpenMP, serial")
    ("async", po::value<bool>()->default_value(true), "use CPU for advection while GPU does micro (ignored if backend != CUDA)")
    ("sd_conc", po::value<unsigned long long>()->required() , "super-droplet number per grid cell (unsigned long long)")
    ("sd_const_multi", po::value<double>()->default_value(rt_params.cloudph_opts_init.sd_const_multi) , "multiplicity in constant multiplicity mode (double)")
    // processes
    ("adve", po::value<bool>()->default_value(rt_params.cloudph_opts.adve) , "particle advection     (1=on, 0=off)")
    ("sedi", po::value<bool>()->default_value(rt_params.cloudph_opts.sedi) , "particle sedimentation (1=on, 0=off)")
    ("cond", po::value<bool>()->default_value(rt_params.cloudph_opts.cond) , "condensational growth  (1=on, 0=off)")
    ("rcyc", po::value<bool>()->default_value(false) , "SDs recycling  (1=on, 0=off)")
    ("coal", po::value<bool>()->default_value(rt_params.cloudph_opts.coal) , "collisional growth     (1=on, 0=off)")
    ("chem_dsl", po::value<bool>()->default_value(rt_params.cloudph_opts.chem_dsl) , "dissolving trace gases (1=on, 0=off)")
    ("chem_dsc", po::value<bool>()->default_value(rt_params.cloudph_opts.chem_dsc) , "dissociation           (1=on, 0=off)")
    ("chem_rct", po::value<bool>()->default_value(rt_params.cloudph_opts.chem_rct) , "aqueous chemistry      (1=on, 0=off)")
    ("dev_count", po::value<int>()->default_value(0), "no. of CUDA devices")
    ("dev_id", po::value<int>()->default_value(-1), "CUDA backend - id of device to be used")
    // free parameters
    ("exact_sstp_cond", po::value<bool>()->default_value(rt_params.cloudph_opts_init.exact_sstp_cond), "exact(per-particle) logic for substeps for condensation")
    ("diag_incloud_time", po::value<bool>()->default_value(rt_params.cloudph_opts_init.diag_incloud_time), "diagnose incloud time of droplets")
    ("sd_conc_large_tail", po::value<bool>()->default_value(rt_params.cloudph_opts_init.sd_conc_large_tail), "add SDs to better represent the large tail")
    ("sstp_cond", po::value<int>()->default_value(rt_params.cloudph_opts_init.sstp_cond), "no. of substeps for condensation")
    ("sstp_coal", po::value<int>()->default_value(rt_params.cloudph_opts_init.sstp_coal), "no. of substeps for coalescence")
    ("sstp_chem", po::value<int>()->default_value(rt_params.cloudph_opts_init.sstp_chem), "no. of substeps for chemistry")
    // 
    ("out_dry", po::value<std::string>()->default_value(""),  "dry radius ranges and moment numbers (r1:r2|n1,n2...;...)")
    ("out_wet", po::value<std::string>()->default_value(""),  "wet radius ranges and moment numbers (r1:r2|n1,n2...;...)")
    ("gccn", po::value<setup::real_t>()->default_value(0) , "concentration of giant aerosols = gccn * VOCALS observations")
//    ("unit_test", po::value<bool>()->default_value(false) , "very low number concentration for unit tests")
    ("adve_scheme", po::value<std::string>()->default_value("euler") , "one of: euler, implicit, pred_corr")
    ("turb_cond", po::value<bool>()->default_value(rt_params.cloudph_opts.turb_cond), "turbulence effects in SD condensation (1=on, 0=off)")
    ("turb_adve", po::value<bool>()->default_value(rt_params.cloudph_opts.turb_adve), "turbulence effects in SD motion (1=on, 0=off)")
    ("turb_coal", po::value<bool>()->default_value(rt_params.cloudph_opts.turb_coal) , "turbulence effects in SD coalescence (1=on, 0=off)")
    ("ReL", po::value<setup::real_t>()->default_value(100) , "taylor-microscale reynolds number (onishi kernel)")
    ("out_dry_spec", po::value<bool>()->default_value(false), "enable output for plotting dry spectrum")
    ("out_wet_spec", po::value<bool>()->default_value(false), "enable output for plotting wet spectrum")
    ("rd_min", po::value<setup::real_t>()->default_value(rt_params.cloudph_opts_init.rd_min), "minimum dry radius of initialized droplets [m] (negative means automatic detection)")
    ("rd_max", po::value<setup::real_t>()->default_value(rt_params.cloudph_opts_init.rd_max), "maximum dry radius of initialized droplets [m] (negative means automatic detection); sd_conc_large_tail==true may result in initialization of even larger droplets")
    ("relax_ccn", po::value<bool>()->default_value(false) , "add CCN if per-level mean of CCN concentration is lower than (case-specific) desired concentration")
    ("coal_kernel", po::value<std::string>()->default_value("hall_davis"), "one of: hall, hall_davis")
    ("term_vel", po::value<std::string>()->default_value("beard77fast"), "one of: beard76, beard77fast")
    ("outfreq_spec", po::value<int>()->default_value(0), "frequency (in timesteps) of spectrum output; 0 for outfreq_spec=outfreq")
    // TODO: MAC, HAC, vent_coef
  ;
  po::variables_map vm;
  handle_opts(opts, vm);
      
  std::string backend_str = vm["backend"].as<std::string>();
  if (backend_str == "CUDA") rt_params.backend = libcloudphxx::lgrngn::CUDA;
  else if (backend_str == "multi_CUDA") rt_params.backend = libcloudphxx::lgrngn::multi_CUDA;
  else if (backend_str == "OpenMP") rt_params.backend = libcloudphxx::lgrngn::OpenMP;
  else if (backend_str == "serial") rt_params.backend = libcloudphxx::lgrngn::serial;

  rt_params.async = vm["async"].as<bool>();
  rt_params.gccn = vm["gccn"].as<setup::real_t>();
  rt_params.outfreq_spec = vm["outfreq_spec"].as<int>();
  if(rt_params.outfreq_spec == 0) rt_params.outfreq_spec = user_params.outfreq;
  assert((rt_params.outfreq_spec % user_params.outfreq == 0) && "outfreq_spec needs to be a multiple of outfreq");
//  bool unit_test = vm["unit_test"].as<bool>();
  setup::real_t ReL = vm["ReL"].as<setup::real_t>();

  rt_params.cloudph_opts_init.sd_conc = vm["sd_conc"].as<unsigned long long>();
  rt_params.cloudph_opts_init.sd_const_multi = vm["sd_const_multi"].as<double>();

  rt_params.cloudph_opts_init.rd_min = vm["rd_min"].as<setup::real_t>();
  rt_params.cloudph_opts_init.rd_max = vm["rd_max"].as<setup::real_t>();

  std::string adve_scheme_str = vm["adve_scheme"].as<std::string>();
  if (adve_scheme_str == "euler") rt_params.cloudph_opts_init.adve_scheme = libcloudphxx::lgrngn::as_t::euler;
  else if (adve_scheme_str == "implicit") rt_params.cloudph_opts_init.adve_scheme = libcloudphxx::lgrngn::as_t::implicit;
  else if (adve_scheme_str == "pred_corr") rt_params.cloudph_opts_init.adve_scheme = libcloudphxx::lgrngn::as_t::pred_corr;
  else throw std::runtime_error("UWLCM: unrecognized adve_scheme optsion");

  setup::arr_1D_t neg_w_LS = rt_params.w_LS->copy(); 
  neg_w_LS *= -1.; // libcloudphxx defines w_LS>0 for downward direction
  std::vector<setup::real_t> vneg_w_LS(neg_w_LS.begin(), neg_w_LS.end());
  rt_params.cloudph_opts_init.w_LS = vneg_w_LS;
  rt_params.cloudph_opts_init.SGS_mix_len = std::vector<setup::real_t>(rt_params.mix_len->begin(), rt_params.mix_len->end());
  rt_params.cloudph_opts_init.aerosol_independent_of_rhod = rt_params.aerosol_independent_of_rhod;
  rt_params.cloudph_opts_init.aerosol_conc_factor = rt_params.aerosol_conc_factor;

  {
    if(user_params.n1_stp*si::cubic_metres >= 0 && user_params.n2_stp*si::cubic_metres >= 0 && user_params.kappa1 == user_params.kappa2) {
        throw std::runtime_error("UWLCM: cannot emplace two modes with same kappa");
    }
    if(user_params.n1_stp*si::cubic_metres >= 0) {
      rt_params.cloudph_opts_init.dry_distros.emplace(
        user_params.kappa1,
        std::make_shared<setup::log_dry_radii<thrust_real_t>> (
          user_params.mean_rd1,
          thrust_real_t(1.0e-6) * si::metres,
          user_params.sdev_rd1,
          thrust_real_t(1.2),
          user_params.n1_stp,
          thrust_real_t(0) / si::cubic_metres
        )
      );
    } 
    if(user_params.n2_stp*si::cubic_metres >= 0) {
      rt_params.cloudph_opts_init.dry_distros.emplace(
        user_params.kappa2,
        std::make_shared<setup::log_dry_radii<thrust_real_t>> (
          thrust_real_t(1.0e-6) * si::metres,
          user_params.mean_rd2,
          thrust_real_t(1.2),
          user_params.sdev_rd2,
          thrust_real_t(0) / si::cubic_metres,
          user_params.n2_stp
        )
      );
    } 
    if(user_params.n1_stp*si::cubic_metres < 0 && user_params.n2_stp*si::cubic_metres < 0) {
      rt_params.cloudph_opts_init.dry_distros.emplace(
        case_ptr->kappa,
        std::make_shared<setup::log_dry_radii<thrust_real_t>> (
          case_ptr->mean_rd1,
          case_ptr->mean_rd2,
          case_ptr->sdev_rd1,
          case_ptr->sdev_rd2,
          user_params.case_n_stp_multiplier * case_ptr->n1_stp,
          user_params.case_n_stp_multiplier * case_ptr->n2_stp
        )
      );
    }
 
    // GCCNs using a fitted lognormal function to Jensen and Nugent, JAS 2016
    /*
    rt_params.cloudph_opts_init.dry_distros.emplace(
      1.28, // key
      std::make_shared<setup::log_dry_radii<thrust_real_t>> (
        quantity<si::length, setup::real_t>(setup::real_t(0.283e-6) * si::meters), // parameters
        quantity<si::length, setup::real_t>(setup::real_t(1e-6) * si::meters), // whatever, n2=0...
        quantity<si::dimensionless, setup::real_t>(setup::real_t(2.235)),
        quantity<si::dimensionless, setup::real_t>(setup::real_t(1.2)), // n2=0...
        quantity<power_typeof_helper<si::length, static_rational<-3>>::type, setup::real_t>(setup::real_t(2.216e6) / si::cubic_meters),
        quantity<power_typeof_helper<si::length, static_rational<-3>>::type, setup::real_t>(setup::real_t(0e6) / si::cubic_meters)
      )
    );
    */

//std::cout << "kappa 0.61 dry distros for 1e-14: " << (*(rt_params.cloudph_opts_init.dry_distros[0.61]))(1e-14) << std::endl;
//std::cout << "kappa 1.28 dry distros for 1e-14: " << (*(rt_params.cloudph_opts_init.dry_distros[1.28]))(1e-14) << std::endl;

    // CCN relaxation stuff
    rt_params.user_params.relax_ccn = vm["relax_ccn"].as<bool>();
    if(rt_params.user_params.relax_ccn)
    {
      rt_params.cloudph_opts_init.rlx_switch = 1;
      rt_params.cloudph_opts_init.rlx_bins = 100;
      rt_params.cloudph_opts_init.rlx_sd_per_bin = 400;
      rt_params.cloudph_opts_init.supstp_rlx = 120 / rt_params.dt; // relaxation every two minutes
      rt_params.cloudph_opts_init.rlx_timescale = 600; // 10 min

      // define kappa ranges of user-defined aerosol distros
      std::pair<thrust_real_t, thrust_real_t> user_kpa_rng1, user_kpa_rng2;

      if(user_params.n1_stp*si::cubic_metres >= 0 || user_params.n2_stp*si::cubic_metres >= 0) {
        if(rt_params.gccn > setup::real_t(0)) 
          throw std::runtime_error("UWLCM: CCN relaxation + GCCN + user-defined aerosol spectra does not work, because kappa ranges for relaxation are not known");
        if(user_params.n1_stp*si::cubic_metres < 0)
          user_kpa_rng2 = std::make_pair<thrust_real_t, thrust_real_t>(0,10); // only one user-defined distribution, whole kappa range
        if(user_params.n2_stp*si::cubic_metres < 0)
          user_kpa_rng1 = std::make_pair<thrust_real_t, thrust_real_t>(0,10); // only one user-defined distribution, whole kappa range
        if(user_params.n1_stp*si::cubic_metres >= 0 && user_params.n2_stp*si::cubic_metres >= 0) {
          if(user_params.kappa1 < user_params.kappa2)
          {
            user_kpa_rng1 = std::make_pair<thrust_real_t, thrust_real_t>(0, (user_params.kappa1 + user_params.kappa2) / 2.);
            user_kpa_rng2 = std::make_pair<thrust_real_t, thrust_real_t>((user_params.kappa1 + user_params.kappa2) / 2., 10);
          }
          else
          {
            user_kpa_rng2 = std::make_pair<thrust_real_t, thrust_real_t>(0, (user_params.kappa1 + user_params.kappa2) / 2.);
            user_kpa_rng1 = std::make_pair<thrust_real_t, thrust_real_t>((user_params.kappa1 + user_params.kappa2) / 2., 10);
          }
        }
      }

      if(user_params.n1_stp*si::cubic_metres >= 0) {
        rt_params.cloudph_opts_init.rlx_dry_distros.emplace(
          user_params.kappa1,
          std::make_tuple(
            std::make_shared<setup::log_dry_radii<thrust_real_t>> (
              user_params.mean_rd1,
              thrust_real_t(1.0e-6) * si::metres,
              user_params.sdev_rd1,
              thrust_real_t(1.2),
              user_params.n1_stp,
              thrust_real_t(0) / si::cubic_metres
            ),
            user_kpa_rng1,
            std::make_pair<thrust_real_t>(0, case_ptr->Z / si::meters)
          )
        );
      } 
      if(user_params.n2_stp*si::cubic_metres >= 0) {
        rt_params.cloudph_opts_init.rlx_dry_distros.emplace(
          user_params.kappa2,
          std::make_tuple(
            std::make_shared<setup::log_dry_radii<thrust_real_t>> (
              thrust_real_t(1.0e-6) * si::metres,
              user_params.mean_rd2,
              thrust_real_t(1.2),
              user_params.sdev_rd2,
              thrust_real_t(0) / si::cubic_metres,
              user_params.n2_stp
            ),
            user_kpa_rng2,
            std::make_pair<thrust_real_t>(0, case_ptr->Z / si::meters)
          )
        );
      } 

      if(user_params.n1_stp*si::cubic_metres < 0 && user_params.n2_stp*si::cubic_metres < 0) {
        rt_params.cloudph_opts_init.rlx_dry_distros.emplace(
          case_ptr->kappa,
          std::make_tuple(
            std::make_shared<setup::log_dry_radii<thrust_real_t>> (
              case_ptr->mean_rd1,
              case_ptr->mean_rd2,
              case_ptr->sdev_rd1,
              case_ptr->sdev_rd2,
              user_params.case_n_stp_multiplier * case_ptr->n1_stp,
              user_params.case_n_stp_multiplier * case_ptr->n2_stp
              //thrust_real_t(4*90e6) / si::cubic_metres,
              //thrust_real_t(4*15e6) / si::cubic_metres 
            ),
            std::make_pair<thrust_real_t>(0., (0.61 + 1.28) / 2.),
            //std::make_pair<thrust_real_t>(1000, case_ptr->Z / si::meters)
            std::make_pair<thrust_real_t>(0, case_ptr->Z / si::meters)
          )
        );
      }
    }

 
    // GCCNs following Jensen and Nugent, JAS 2016
    if(rt_params.gccn > setup::real_t(0))
    {
      // TODO: src_x0, src_x1, src_y0 and src_y1 should exclude half of outside cells, like x0, x1, y0, y1?
      rt_params.cloudph_opts_init.src_type = libcloudphxx::lgrngn::src_t::simple;
      rt_params.cloudph_opts_init.src_x0 = 0;
      rt_params.cloudph_opts_init.src_x1 = case_ptr->X / si::meters;
      rt_params.cloudph_opts_init.src_y0 = 0;
      rt_params.cloudph_opts_init.src_y1 = case_ptr->Y / si::meters;
      rt_params.cloudph_opts_init.src_z0 = 0;
//      rt_params.cloudph_opts_init.src_z1 = case_ptr->Z / si::meters;
      rt_params.cloudph_opts_init.src_z1 = case_ptr->gccn_max_height / si::meters;// 700;
  //    rt_params.cloudph_opts_init.src_z1 = 200;

      rt_params.cloudph_opts_init.src_sd_conc = 38;

/*
      rt_params.cloudph_opts_init.src_dry_sizes.emplace(
        1.28, // kappa
        std::map<setup::real_t, std::pair<setup::real_t, int> > {
          {0.8e-6, {rt_params.gccn / rt_params.dt * 111800, 1}},
          {1.0e-6, {rt_params.gccn / rt_params.dt * 68490,  1}},
          {1.2e-6, {rt_params.gccn / rt_params.dt * 38400,  1}},
          {1.4e-6, {rt_params.gccn / rt_params.dt * 21820,  1}},
          {1.6e-6, {rt_params.gccn / rt_params.dt * 13300,  1}},
          {1.8e-6, {rt_params.gccn / rt_params.dt * 8496,  1}},
          {2.0e-6, {rt_params.gccn / rt_params.dt * 5486,  1}},
          {2.2e-6, {rt_params.gccn / rt_params.dt * 3805,  1}},
          {2.4e-6, {rt_params.gccn / rt_params.dt * 2593,  1}},
          {2.6e-6, {rt_params.gccn / rt_params.dt * 1919,  1}},
          {2.8e-6, {rt_params.gccn / rt_params.dt * 1278,  1}},
          {3.0e-6, {rt_params.gccn / rt_params.dt * 988.4,  1}},
          {3.2e-6, {rt_params.gccn / rt_params.dt * 777.9,  1}},
          {3.4e-6, {rt_params.gccn / rt_params.dt * 519.5,  1}},
          {3.6e-6, {rt_params.gccn / rt_params.dt * 400.5,  1}},
          {3.8e-6, {rt_params.gccn / rt_params.dt * 376.9,  1}},
          {4.0e-6, {rt_params.gccn / rt_params.dt * 265.3,  1}},
          {4.2e-6, {rt_params.gccn / rt_params.dt * 212.4,  1}},
          {4.4e-6, {rt_params.gccn / rt_params.dt * 137.8,  1}},
          {4.6e-6, {rt_params.gccn / rt_params.dt * 121.4,  1}},
          {4.8e-6, {rt_params.gccn / rt_params.dt * 100.9,  1}},
          {5.0e-6, {rt_params.gccn / rt_params.dt * 122.2,  1}},
          {5.2e-6, {rt_params.gccn / rt_params.dt * 50.64,  1}},
          {5.4e-6, {rt_params.gccn / rt_params.dt * 38.30,  1}},
          {5.6e-6, {rt_params.gccn / rt_params.dt * 55.47,  1}},
          {5.8e-6, {rt_params.gccn / rt_params.dt * 21.45,  1}},
          {6.0e-6, {rt_params.gccn / rt_params.dt * 12.95,  1}},
          {6.2e-6, {rt_params.gccn / rt_params.dt * 43.23,  1}},
          {6.4e-6, {rt_params.gccn / rt_params.dt * 26.26,  1}},
          {6.6e-6, {rt_params.gccn / rt_params.dt * 30.50,  1}},
          {6.8e-6, {rt_params.gccn / rt_params.dt * 4.385,  1}},
          {7.0e-6, {rt_params.gccn / rt_params.dt * 4.372,  1}},
          {7.2e-6, {rt_params.gccn / rt_params.dt * 4.465,  1}},
          {7.4e-6, {rt_params.gccn / rt_params.dt * 4.395,  1}},
          {7.6e-6, {rt_params.gccn / rt_params.dt * 4.427,  1}},
          {7.8e-6, {rt_params.gccn / rt_params.dt * 4.411,  1}},
          {8.6e-6, {rt_params.gccn / rt_params.dt * 4.522,  1}},
          {9.0e-6, {rt_params.gccn / rt_params.dt * 4.542,  1}}
        }
      );
      */

      rt_params.cloudph_opts_init.src_dry_distros.emplace(
        1.28, // kappa
        std::make_shared<setup::log_dry_radii_gccn<thrust_real_t>> (
          log(0.8e-6),      // minimum radius  
          log(10e-6),   // maximum radius
          rt_params.gccn / rt_params.dt // concenctration multiplier
        )
      );

      // GCCN relaxation stuff
      if(rt_params.user_params.relax_ccn)
      {
        rt_params.cloudph_opts_init.rlx_dry_distros.emplace(
          1.28, // kappa
          std::make_tuple(
            std::make_shared<setup::log_dry_radii_gccn<thrust_real_t>> (
              log(0.8e-6),      // minimum radius  
              log(10e-6),   // maximum radius
              rt_params.gccn // concenctration multiplier
            ),
            std::make_pair<thrust_real_t>((0.61 + 1.28) / 2., 10000),
            std::make_pair<thrust_real_t>(0, rt_params.cloudph_opts_init.src_z1)
            //std::make_pair<thrust_real_t>(0, 700)
          )
        );
      }
    }
   }
/*  else if(unit_test)
    boost::assign::ptr_map_insert<
      setup::log_dry_radii_unit_test<thrust_real_t> // value type
    >(
      rt_params.cloudph_opts_init.dry_distros // map
    )(
      setup::kappa // key
    );*/
/*  
  if(gccn) // add the gccns spectra
    boost::assign::ptr_map_insert<
      setup::log_dry_radii_gccn<thrust_real_t> // value type
    >(
      rt_params.cloudph_opts_init.dry_distros // map
    )(
      setup::kappa_gccn // key
    );
*/

  // process toggling
  rt_params.cloudph_opts.adve = vm["adve"].as<bool>();
  rt_params.cloudph_opts.sedi = vm["sedi"].as<bool>();
  rt_params.cloudph_opts.cond = vm["cond"].as<bool>();
  rt_params.cloudph_opts.coal = vm["coal"].as<bool>();

  rt_params.cloudph_opts.rcyc = vm["rcyc"].as<bool>();
  rt_params.cloudph_opts.chem_dsl = vm["chem_dsl"].as<bool>();
  rt_params.cloudph_opts.chem_dsc = vm["chem_dsc"].as<bool>();
  rt_params.cloudph_opts.chem_rct = vm["chem_rct"].as<bool>();

  rt_params.cloudph_opts_init.dev_count = vm["dev_count"].as<int>();
  rt_params.cloudph_opts_init.dev_id = vm["dev_id"].as<int>();
  // free parameters
  rt_params.cloudph_opts_init.sstp_cond = vm["sstp_cond"].as<int>();
  rt_params.cloudph_opts_init.sstp_coal = vm["sstp_coal"].as<int>();
  rt_params.cloudph_opts_init.sstp_chem = vm["sstp_chem"].as<int>();
  rt_params.cloudph_opts_init.exact_sstp_cond = vm["exact_sstp_cond"].as<bool>();
  rt_params.cloudph_opts_init.diag_incloud_time = vm["diag_incloud_time"].as<bool>();
  rt_params.cloudph_opts_init.sd_conc_large_tail = vm["sd_conc_large_tail"].as<bool>();

  rt_params.cloudph_opts_init.rng_seed = user_params.rng_seed;
  rt_params.cloudph_opts_init.rng_seed_init = user_params.rng_seed_init;
  rt_params.cloudph_opts_init.rng_seed_init_switch = true;

  // coalescence kernel choice
  std::string kernel_str = vm["coal_kernel"].as<std::string>();

  if(!vm["turb_coal"].as<bool>())
    rt_params.cloudph_opts_init.kernel = kernel_str == "hall" ? libcloudphxx::lgrngn::kernel_t::hall : libcloudphxx::lgrngn::kernel_t::hall_davis_no_waals;
  else
  {
    rt_params.cloudph_opts_init.kernel = kernel_str == "hall" ? libcloudphxx::lgrngn::kernel_t::onishi_hall : libcloudphxx::lgrngn::kernel_t::onishi_hall_davis_no_waals;
    rt_params.cloudph_opts_init.kernel_parameters.push_back(ReL);
    rt_params.cloudph_opts_init.turb_coal_switch = 1;
    rt_params.cloudph_opts.turb_coal = 1;
  }
  // terminal velocity choice
  std::string vt_str = vm["term_vel"].as<std::string>();
  rt_params.cloudph_opts_init.terminal_velocity = vt_str == "beard76" ? libcloudphxx::lgrngn::vt_t::beard76 : libcloudphxx::lgrngn::vt_t::beard77fast;

  rt_params.cloudph_opts_init.RH_formula = libcloudphxx::lgrngn::RH_formula_t::rv_cc; // use rv to be consistent with Lipps Hemler

  // turbulence effects for SDs
  rt_params.cloudph_opts_init.turb_cond_switch = vm["turb_cond"].as<bool>();
  rt_params.cloudph_opts.turb_cond = vm["turb_cond"].as<bool>();
  
  rt_params.cloudph_opts_init.turb_adve_switch = vm["turb_adve"].as<bool>();
  rt_params.cloudph_opts.turb_adve = vm["turb_adve"].as<bool>();
  
  // subsidence of SDs
  rt_params.cloudph_opts_init.subs_switch = rt_params.subsidence == subs_t::local || rt_params.subsidence == subs_t::mean ? true : false;
  rt_params.cloudph_opts.subs = rt_params.subsidence == subs_t::local || rt_params.subsidence == subs_t::mean ? true : false;

  // parsing --out_dry and --out_wet options values
  // the format is: "rmin:rmax|0,1,2;rmin:rmax|3;..."
  for (auto &opt : std::set<std::string>({"out_dry", "out_wet"}))
  {
    namespace qi = boost::spirit::qi;
    namespace phoenix = boost::phoenix;

    std::string val = vm[opt].as<std::string>();
    auto first = val.begin();
    auto last  = val.end();

    std::vector<std::pair<std::string, std::string>> min_maxnum;
    outmom_t<thrust_real_t> &moms = 
      opt == "out_dry"
        ? rt_params.out_dry
        : rt_params.out_wet;

    const bool result = qi::phrase_parse(first, last, 
      *(
	*(qi::char_-":")  >>  qi::lit(":") >>  
	*(qi::char_-";")  >> -qi::lit(";") 
      ),
      boost::spirit::ascii::space, min_maxnum
    );    
    if (!result || first != last) BOOST_THROW_EXCEPTION(po::validation_error(
        po::validation_error::invalid_option_value, opt, val 
    ));  

    for (auto &ss : min_maxnum)
    {
      int sep = ss.second.find('|'); 

      moms.push_back(outmom_t<thrust_real_t>::value_type({
        outmom_t<thrust_real_t>::value_type::first_type(
          boost::lexical_cast<setup::real_t>(ss.first) * si::metres,
          boost::lexical_cast<setup::real_t>(ss.second.substr(0, sep)) * si::metres
        ), 
        outmom_t<setup::real_t>::value_type::second_type()
      }));

      // TODO catch (boost::bad_lexical_cast &)

      std::string nums = ss.second.substr(sep+1);;
      auto nums_first = nums.begin();
      auto nums_last  = nums.end();

      const bool result = qi::phrase_parse(
        nums_first, 
        nums_last, 
	(
	  qi::int_[phoenix::push_back(phoenix::ref(moms.back().second), qi::_1)]
	      >> *(',' >> qi::int_[phoenix::push_back(phoenix::ref(moms.back().second), qi::_1)])
	),
	boost::spirit::ascii::space
      );    
      if (!result || nums_first != nums_last) BOOST_THROW_EXCEPTION(po::validation_error(
	  po::validation_error::invalid_option_value, opt, val // TODO: report only the relevant part?
      ));  
    }
  } 

  for (auto &opt : std::set<std::string>({"out_dry_spec", "out_wet_spec"}))
  {
    if(vm[opt].as<bool>())
    {
      auto left_edges = opt == "out_dry_spec" ? bins_dry() : bins_wet();
      auto &out = opt == "out_dry_spec" ? rt_params.out_dry : rt_params.out_wet;
      for (int i = 0; i < left_edges.size()-1; ++i)
      {
        out.push_back(outmom_t<thrust_real_t>::value_type(
          outmom_t<thrust_real_t>::value_type::first_type(
            left_edges.at(i),
            left_edges.at(i+1)
          ), 
          outmom_t<setup::real_t>::value_type::second_type{0} // 0-th moment only, e.g. {0,1,3} would store 0-th, 1-st and 3-rd moments
        ));
      }
    }
  }
//  if(vm["out_dry_spec"].as<bool>())
//  {
//    auto left_edges = bins_dry();
//    for (int i = 0; i < left_edges.size()-1; ++i)
//    {
//      rt_params.out_dry.push_back(outmom_t<thrust_real_t>::value_type(
//        outmom_t<thrust_real_t>::value_type::first_type(
//          left_edges.at(i),
//          left_edges.at(i+1)
//        ), 
//        outmom_t<setup::real_t>::value_type::second_type{0}
//      ));
//    }
//  }
//
  if(rt_params.subsidence == subs_t::mean)
    std::cerr << "UWLCM warning: case requires that subsidence is done for horizontally-averaged fields, but that is not implemented for liquid water in Lagrangian microphysics. Subsidence of liquid water (super-droplets) will be done per column." << std::endl;
}
