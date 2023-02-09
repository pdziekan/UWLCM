#pragma once
#include <libmpdata++/solvers/mpdata_rhs_vip_prs_sgs.hpp>
#include <libmpdata++/output/hdf5_xdmf.hpp>
#include "../detail/checknan.cpp"
#include <H5Cpp.h>
#include <libmpdata++/output/hdf5.hpp>

template <class ct_params_t, class enableif = void>
class slvr_piggy
{};

using namespace libmpdataxx; // TODO: get rid of it?

constexpr int minhalo = 1; 

// driver
template <class ct_params_t>
class slvr_piggy<
  ct_params_t,
  typename std::enable_if<ct_params_t::piggy == 0 >::type
> : public 
  output::hdf5_xdmf<
    solvers::mpdata_rhs_vip_prs_sgs<ct_params_t, minhalo>
  >
{
  private:
  bool save_vel; // should velocity field be stored for piggybacking
  setup::real_t prs_tol; // store a copy for output purposes

  protected:
  using parent_t = output::hdf5_xdmf<
    solvers::mpdata_rhs_vip_prs_sgs<ct_params_t, minhalo>
  >;  

  std::unique_ptr<H5::H5File> hdfpu_vel;
  std::map<int, H5::DataSet> vels;

  const std::string vel_out_name = "velocity_out.h5";
  std::ofstream f_vel_out; // file for velocity field

  // TODO: create record_halo_hlpr in libmpdata hdf5.hpp and move all the record stuff there
  
//  const H5::FloatType
//    flttype_output = H5::PredType::NATIVE_FLOAT;
//
//  hid_t fapl_id;

  void hook_ante_loop(int nt) 
  {
    parent_t::hook_ante_loop(nt); 
    if(this->rank==0)
    {
      // open file for out vel
      if(save_vel)
      {     
      try
        {
          f_vel_out.open(this->outdir+"/velocity_out.dat"); 
        }
        catch(...)
        {
          throw std::runtime_error("error opening velocity output file '{outdir}/velocity_out.dat'");
        }
      }
      this->record_aux_const("save_vel", "piggy", save_vel);  
      this->record_aux_const("rt_params prs_tol", "piggy", prs_tol);  
    }
    this->mem->barrier();
  }

  void hook_post_step()
  {
    parent_t::hook_post_step(); // includes changes of velocity field due to vip_rhs_impl_fnlz()
    this->mem->barrier();
    // save velocity field
    if(this->rank==0 && save_vel)
    {
      hdfpu_vel.reset(new H5::H5File(this->outdir + "/" + this->hdf_name(this->base_name("timestep")), H5F_ACC_TRUNC
#if defined(USE_MPI)
          , H5P_DEFAULT, this->fapl_id
#endif
        ));

        for (int d = 0; d < parent_t::n_dims; ++d)
          this->record_aux_halo_hlpr(
            this->outvars[this->vip_ixs[d]].name,
            this->state(this->vip_ixs[d]),
            *hdfpu_vel
          );
    }
  }

  struct rt_params_t : parent_t::rt_params_t 
  {
    bool save_vel;

    // ctor
    rt_params_t()
    {
      po::options_description opts("Driver options"); 
      opts.add_options()
        ("save_vel", po::value<bool>()->default_value(false), "should velocity field be stored (for future piggybacking)")
      ;
      opts.add_options()
        ("prs_tol", po::value<setup::real_t>()->default_value(1e-6) , "pressure solver tolerance"); // not really related to piggybacking, but convenient to put here as it is the first solver to inherit from libmpdata++
      po::variables_map vm;
      handle_opts(opts, vm);
          
      save_vel = vm["save_vel"].as<bool>();
      this->prs_tol = vm["prs_tol"].as<setup::real_t>();
    }
  };

  // ctor
  slvr_piggy(
    typename parent_t::ctor_args_t args,
    rt_params_t p
  ) :
    parent_t(args, p),
    save_vel(p.save_vel),
    prs_tol(p.prs_tol)
    {}
};


// piggybacker
template <class ct_params_t>
class slvr_piggy<
  ct_params_t,
  typename std::enable_if<ct_params_t::piggy == 1 >::type
> : public 
  output::hdf5_xdmf<
    solvers::mpdata_rhs_vip<ct_params_t, minhalo>
  >
{

  protected:
  using parent_t = output::hdf5_xdmf<
    solvers::mpdata_rhs_vip<ct_params_t, minhalo>
  >;  

  private:
  typename parent_t::arr_t in_bfr; // input buffer for velocity
  std::string vel_in;
  
  protected:

  std::ifstream f_vel_in; // input velocity file


  void hook_ante_loop(int nt) 
  {
    parent_t::hook_ante_loop(nt); 

    if(this->rank==0)
    {
      po::options_description opts("Piggybacker options"); 
      opts.add_options()
        ("vel_in", po::value<std::string>()->required(), "file with input velocities (for piggybacking)")
      ;
      po::variables_map vm;
      handle_opts(opts, vm);
          
      vel_in = vm["vel_in"].as<std::string>();
      std::cout << "piggybacking from: " << vel_in << std::endl;

      in_bfr.resize(this->state(this->vip_ixs[0]).shape());
      // open file for in vel
      // TODO: somehow check dimensionality of the input arrays
      try{
        f_vel_in.open(vel_in); 
      }
      catch(...)
      {
        throw std::runtime_error("error opening velocities input file defined by --vel_in");
      }
      this->record_aux_const("piggybacking", "piggy", "true");
      this->record_aux_const("vel_in", "piggy", vel_in); 
    }
    this->mem->barrier();
  }

  void hook_post_step() 
  {
    parent_t::hook_post_step(); // do whatever
    this->mem->barrier(); //necessary?
    // read velo, overwrite any vel rhs
    if(this->rank==0)
    {
      using ix = typename ct_params_t::ix;
      H5::H5File h5f(vel_in+ "/" + this->hdf_name(this->base_name("timestep")), H5F_ACC_RDONLY);
      std::cout<<" timestep "<<this->timestep<< " ";
      
      for (int d = 0; d < parent_t::n_dims; ++d)
      {
        H5::DataSet dataset = h5f.openDataSet(this->outvars[this->vip_ixs[d]].name);
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();
        hsize_t dims[rank];
        dataspace.getSimpleExtentDims(dims, NULL);

        dataset.read(this->state(this->vip_ixs[d]).data(), H5::PredType::NATIVE_FLOAT);
        // f_vel_in >> in_bfr;
        // this->state(this->vip_ixs[d]) = in_bfr;
      }
    }
    this->mem->barrier();
  }

  // ctor
  slvr_piggy(
    typename parent_t::ctor_args_t args,
    typename parent_t::rt_params_t const &p
  ) :
    parent_t(args, p) {}
};

