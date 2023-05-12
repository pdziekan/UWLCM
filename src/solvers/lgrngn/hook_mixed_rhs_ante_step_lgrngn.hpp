#pragma once
#include "../slvr_lgrngn.hpp"
#include "../../detail/func_time.hpp"
#if defined(STD_FUTURE_WORKS)
#  include <future>
#endif

template <class ct_params_t>
void slvr_lgrngn<ct_params_t>::hook_mixed_rhs_ante_step()
{
/*
  courants[0] = NAN;
  courants[1] = NAN;
  courants[2] = NAN;

  courants[0] = 0;
  courants[1] = 0;
  courants[2] = 0;

  this->mem->barrier();

  courants[0](this->ijk_ref) = 0;
  courants[1](this->ijk_ref) = 0;
  courants[2](this->ijk_ref) = 0;

  this->mem->barrier();
  */

  this->interpolate_refinee(ix::u);
  this->interpolate_refinee(ix::v);
  this->interpolate_refinee(ix::w);

  this->interpolate_refined_courants(courants, this->uvw_ref);

//  this->mem->refinee(this->ix_r2r.at(ix::u))(this->ijkm_ref) = 0;
//  this->mem->refinee(this->ix_r2r.at(ix::v))(this->ijkm_ref) = 0;
//  this->mem->refinee(this->ix_r2r.at(ix::w))(this->ijkm_ref) = 0;
//
//  courants[0] = 0;
//  courants[1] = 0;
//  courants[2] = 0;

  //this->xchng_ref(ix::u);
  //this->xchng_ref(ix::v);
  //this->xchng_ref(ix::w);

//  courants[0](this->ijkm_ref) = this->mem->psi_ref.at(this->ix_r2r.at(ix::u))(this->ijkm_ref);
//  courants[0](this->ijkm_ref) = this->mem->refinee(this->ix_r2r.at(ix::u))(this->ijkm_ref) * this->dt / this->di;
//  courants[1](this->ijkm_ref) = this->mem->refinee(this->ix_r2r.at(ix::v))(this->ijkm_ref) * this->dt / this->dj;
//  courants[2](this->ijkm_ref) = this->mem->refinee(this->ix_r2r.at(ix::w))(this->ijkm_ref) * this->dt / this->dk;



  negtozero(this->mem->advectee(ix::rv)(this->ijk), "rv at start of mixed_rhs_ante_step");

  // ---- reconstruction of th_l ----
  this->generate_stretching_parameters(std::random_device{}());//, libmpdataxx::formulae::fractal::stretch_params::d_distro_t::LES_th_supersaturated);

  // store a copy of th
  auto& th_copy(this->tmp1);
  th_copy(this->ijk) = this->mem->advectee(ix::th)(this->ijk);
  
  // calculate th_l in place of th using r_l on resolved scales
  const auto l_tri = libcloudphxx::common::const_cp::l_tri<setup::real_t>() * si::kilograms / si::joules;
  const auto c_pd = libcloudphxx::common::moist_air::c_pd<setup::real_t>() * si::kilograms * si::kelvins / si::joules;
  this->mem->advectee(ix::th)(this->ijk) -= this->r_l(this->ijk) * l_tri / c_pd;  // TODO:  l_v(T) instead of l_tri; 
                                                                                  // TODO2: divide by exner(p)=T/theta 
                                                                                  // NOTE:  r_l was calculated after condensation in previous step 
                                                                                  //        and was advected with upwind
  
  // do the reconstruction of th_l
  this->reconstruct_refinee(ix::th);

  // calculate rl_ref, r_l on refined scales
  // NOTE:  very similar to part of diag_rx
  // NOTE2: this could be done in diag_rl, but diag_rl is called in ante_delayed_step, hence we would need to advect rl_ref just like we advect r_l...
  // NOTE3: prtcls could be currently doing some computations? Would this be a problem?
  typename parent_t::arr_t &rl_ref(this->tmp_ref);
  if(this->rank == 0)
  {
    prtcls->diag_all();
    prtcls->diag_wet_mom(3);
    rl_ref(this->domain_ref) = typename parent_t::arr_t(prtcls->outbuf(), this->shape(this->domain_ref), blitz::neverDeleteData);
  }
  this->mem->barrier();
  
  // calculate reconstructed th from reconstructed th_l and from r_l on reconstructed scales
  this->mem->refinee(this->ix_r2r.at(ix::th))(this->ijk_ref) += 4./3. * 1000. * 3.14159 * rl_ref(this->ijk_ref) * l_tri / c_pd; // TODO: same as above in calculation of th_l 
  
  // restore th from the backup
  this->mem->barrier();
  this->mem->advectee(ix::th)(this->ijk) = th_copy(this->ijk);

  // ---- reconstruction of r_tot ---- TODO: make common function with th_l
  // TODO: generate different parameters than for th?
  //this->generate_stretching_parameters(std::random_device{}());//, libmpdataxx::formulae::fractal::stretch_params::d_distro_t::LES_rv_supersaturated);

  // store a copy of rv
  auto& rv_copy(this->tmp1);
  rv_copy(this->ijk) = this->mem->advectee(ix::rv)(this->ijk);
  
  // calculate rtot in place of th using rv on resolved scales
  this->mem->advectee(ix::rv)(this->ijk) += this->r_l(this->ijk);
  
  // do the reconstruction of rt
  this->reconstruct_refinee(ix::rv);
  
  // calculate reconstructed rv from reconstructed rt and from rl on reconstructed scales
  // NOTE: rl_ref already caculated when reconstructing th
  this->mem->refinee(this->ix_r2r.at(ix::rv))(this->ijk_ref) -= 4./3. * 1000. * 3.14159 * rl_ref(this->ijk_ref);
  
  // restore th from the backup
  this->mem->barrier();
  this->mem->advectee(ix::rv)(this->ijk) = rv_copy(this->ijk);

  // ---- post-reconstruction sanity checks ----
  
  //negtozero(this->mem->refinee(this->ix_r2r.at(ix::rv))(this->ijk_ref), "refined rv at start of mixed_rhs_ante_step");
  negtozero2(this->mem->refinee(this->ix_r2r.at(ix::rv))(this->ijk_ref), "refined rv at start of mixed_rhs_ante_step",
	     this->negref_dbg_arrs, this->negref_dbg_arr_names);

  rv_pre_cond(this->ijk_ref) = this->mem->refinee(this->ix_r2r.at(ix::rv))(this->ijk_ref); 
  th_pre_cond(this->ijk_ref) = this->mem->refinee(this->ix_r2r.at(ix::th))(this->ijk_ref); 

  this->mem->barrier();

  // pass Eulerian fields to microphysics 
  if (this->rank == 0) 
  {

    nancheck(courants[0], "courants[0] after interpolation from refined velocities");
    nancheck(courants[1], "courants[1] after interpolation from refined velocities");
    nancheck(courants[2], "courants[2] after interpolation from refined velocities");

    // temporarily Cx & Cz are multiplied by this->rhod ...
    /*
    auto 
      Cx = this->mem->GC[0](this->Cx_domain).copy(),
      Cy = this->mem->GC[1](this->Cy_domain).copy(), // TODO: no need to copy in 2D
      Cz = this->mem->GC[ix::w](this->Cz_domain).copy(); 
    nancheck(Cx, "Cx after copying from mpdata");
    nancheck(Cy, "Cy after copying from mpdata");
    nancheck(Cz, "Cz after copying from mpdata");

    // ... and now dividing them by this->rhod (TODO: z=0 is located at k=1/2)
    {
      Cx.reindex(this->zero) /= (params.profs.rhod)(this->vert_idx);
      Cy.reindex(this->zero) /= (params.profs.rhod)(this->vert_idx);
      Cz.reindex(this->zero) /= (params.profs.rhod)(this->vert_idx); // TODO: should be interpolated, since theres a shift between positions of rhod and Cz
    }
    */

    // assuring previous async step finished ...
#if defined(STD_FUTURE_WORKS)
    if (
      params.async && 
      this->timestep != 0 && // ... but not in first timestep ...
      ((this->timestep ) % this->outfreq != 0) // ... and not after diag call, note: timestep is updated after ante_step
    ) {
      assert(ftr.valid());
#if defined(UWLCM_TIMING)
      tbeg = setup::clock::now();
#endif
#if defined(UWLCM_TIMING)
      parent_t::tasync_gpu += ftr.get();
#else
      ftr.get();
#endif
#if defined(UWLCM_TIMING)
      tend = setup::clock::now();
      parent_t::tasync_wait += std::chrono::duration_cast<setup::timer>( tend - tbeg );
#endif
    } else assert(!ftr.valid()); 
#endif

    // change src and rlx flags after the first step. needs to be done after async finished, because async uses opts reference
    if(this->timestep == 1)
    {
      // turn off aerosol src, because it was only used to initialize gccn below some height
      params.cloudph_opts.src = false;
      // if relaxation is to be done, turn it on after gccn were created by src
      if(params.user_params.relax_ccn)
        params.cloudph_opts.rlx = true;
    }

    // start synchronous stuff timer
#if defined(UWLCM_TIMING)
    tbeg = setup::clock::now();
#endif

    using libcloudphxx::lgrngn::particles_t;
    using libcloudphxx::lgrngn::CUDA;
    using libcloudphxx::lgrngn::multi_CUDA;

/*
    prtcls->sync_in(
      make_arrinfo(this->mem->advectee(ix::th)),
      make_arrinfo(this->mem->advectee(ix::rv)),
      libcloudphxx::lgrngn::arrinfo_t<real_t>(),
      make_arrinfo(Cx),
      this->n_dims == 2 ? libcloudphxx::lgrngn::arrinfo_t<real_t>() : make_arrinfo(Cy),
      make_arrinfo(Cz),
      (ct_params_t::sgs_scheme == libmpdataxx::solvers::iles) || (!params.cloudph_opts.turb_cond && !params.cloudph_opts.turb_adve && !params.cloudph_opts.turb_coal) ?
                                  libcloudphxx::lgrngn::arrinfo_t<real_t>() :
                                  make_arrinfo(this->diss_rate(this->domain).reindex(this->zero))
    );
    */

/*
    std::cerr << "Cx: " << Cx << std::endl;
    std::cerr << "Cy: " << Cy << std::endl;
    std::cerr << "Cz: " << Cz << std::endl;

    std::cerr << "curants[0]: " << courants[0] << std::endl;
    std::cerr << "curants[1]: " << courants[1] << std::endl;
    std::cerr << "curants[2]: " << courants[2] << std::endl;

    // TODO: make courants have the desired sizes (?)
    auto 
      Cx_ref = courants[0].copy(),
      Cy_ref = courants[1].copy(),
      Cz_ref = courants[2].copy();


    std::cerr << "Cx_ref: " << Cx_ref << std::endl;
    std::cerr << "Cy_ref: " << Cy_ref << std::endl;
    std::cerr << "Cz_ref: " << Cz_ref << std::endl;
    */

    prtcls->sync_in(
      make_arrinfo(this->mem->refinee(this->ix_r2r.at(ix::th))),
      make_arrinfo(this->mem->refinee(this->ix_r2r.at(ix::rv))),
      libcloudphxx::lgrngn::arrinfo_t<real_t>(),
      make_arrinfo(courants[0]),
      this->n_dims == 2 ? libcloudphxx::lgrngn::arrinfo_t<real_t>() : make_arrinfo(courants[1]),
      make_arrinfo(courants[2])
      /*,
      (ct_params_t::sgs_scheme == libmpdataxx::solvers::iles) || (!params.cloudph_opts.turb_cond && !params.cloudph_opts.turb_adve && !params.cloudph_opts.turb_coal) ?
                                  libcloudphxx::lgrngn::arrinfo_t<real_t>() :
                                  make_arrinfo(this->diss_rate(this->domain).reindex(this->zero))
                                  */
    );

    // start sync/async run of step_cond
    // step_cond takes th and rv only for sync_out purposes - the values of th and rv before condensation come from sync_in, i.e. before apply_rhs

#if defined(STD_FUTURE_WORKS)
    if (params.async)
    {
      assert(!ftr.valid());
      if(params.backend == CUDA)
        ftr = async_launcher(
          &particles_t<real_t, CUDA>::step_cond, 
          dynamic_cast<particles_t<real_t, CUDA>*>(prtcls.get()),
          params.cloudph_opts,
          make_arrinfo(th_post_cond(this->domain_ref).reindex(this->zero)),
          make_arrinfo(rv_post_cond(this->domain_ref).reindex(this->zero)),
          std::map<enum libcloudphxx::common::chem::chem_species_t, libcloudphxx::lgrngn::arrinfo_t<real_t> >()
        );
      else if(params.backend == multi_CUDA)
        ftr = async_launcher(
          &particles_t<real_t, multi_CUDA>::step_cond, 
          dynamic_cast<particles_t<real_t, multi_CUDA>*>(prtcls.get()),
          params.cloudph_opts,
          make_arrinfo(th_post_cond(this->domain_ref).reindex(this->zero)),
          make_arrinfo(rv_post_cond(this->domain_ref).reindex(this->zero)),
          std::map<enum libcloudphxx::common::chem::chem_species_t, libcloudphxx::lgrngn::arrinfo_t<real_t> >()
        );
      assert(ftr.valid());
    } else 
#endif
    {
      prtcls->step_cond(
        params.cloudph_opts,
        make_arrinfo(th_post_cond(this->domain_ref).reindex(this->zero)),
        make_arrinfo(rv_post_cond(this->domain_ref).reindex(this->zero))
      );
    }

#if defined(UWLCM_TIMING)
    tend = setup::clock::now();
    parent_t::tsync += std::chrono::duration_cast<setup::timer>( tend - tbeg );
#endif
  }
  this->mem->barrier();

  parent_t::hook_mixed_rhs_ante_step();
}
