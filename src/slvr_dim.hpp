#pragma once
#include "slvr_common.hpp"

// custom 3D idxperm that accepts idx_t; todo: make it part of libmpdata?
namespace libmpdataxx
{
  namespace idxperm
  {
    template<int d>
    inline idx_t<3> pi(const rng_t &rng, const idx_t<2> &idx) { return pi<d>(rng, idx[0], idx[1]); }

    template<int d>
    inline idx_t<3> pi(const int &i, const idx_t<2> &idx) { return pi<d>(rng_t(i,i), idx); }
  };
};

template <class ct_params_t, class enableif = void>
class slvr_dim
{};

using libmpdataxx::arakawa_c::h;

// 2D version 
template <class ct_params_t>
class slvr_dim<
  ct_params_t,
  typename std::enable_if<ct_params_t::n_dims == 2 >::type
> : public slvr_common<ct_params_t>
{
  using parent_t = slvr_common<ct_params_t>;
  using ix = typename ct_params_t::ix;

  protected:
  // inject dimension-independent ranges
  idx_t<2> domain = idx_t<2>({this->mem->grid_size[0], this->mem->grid_size[1]});
  rng_t hrzntl_domain = this->mem->grid_size[0];
  rng_t hrzntl_subdomain = this->i;
  idx_t<2> Cx_domain = idx_t<2>({this->mem->grid_size[0]^h, this->mem->grid_size[1]});
  idx_t<2> Cy_domain;
  idx_t<2> Cz_domain = idx_t<2>({this->mem->grid_size[0], this->mem->grid_size[1]^h});

  enum {vert_dim = 1};
  blitz::TinyVector<int, 2> zero = blitz::TinyVector<int, 2>({0,0});
  blitz::secondIndex vert_idx;
  std::map<int, int> vip_map = std::map<int, int>{{0, ix::vip_i}};

  void vert_grad_fwd(typename parent_t::arr_t &in, typename parent_t::arr_t &out, setup::real_t dz)
  {
    for (auto &bc : this->bcs[1]) bc->fill_halos_sclr(in, this->i, false);
    out(this->i, this->j) = ( in(this->i, this->j+1) - in(this->i, this->j)) / dz;
    // top and bottom cells are two times lower
    out(this->i, 0) *= 2; 
    out(this->i, this->j.last()) *= 2; 
  }

  void vert_grad_cnt(typename parent_t::arr_t &in, typename parent_t::arr_t &out, setup::real_t dz)
  {
    for (auto &bc : this->bcs[1]) bc->fill_halos_sclr(in, this->i, false);
    out(this->i, this->j) = ( in(this->i, this->j+1) - in(this->i, this->j-1)) / 2./ dz;
    // top and bottom cells are two times lower
    out(this->i, 0) *= 2; 
    out(this->i, this->j.last()) *= 2; 
  }

  void smooth(typename parent_t::arr_t &in, typename parent_t::arr_t &out)
  {
    for (auto &bc : this->bcs[1]) bc->fill_halos_sclr(in, this->i, false);
    out(this->i, this->j) = 0.25 * (in(this->i, this->j + 1) + 2 * in(this->i, this->j) + in(this->i, this->j - 1));
  }

  auto calc_U() 
    return_macro(,
    abs(this->state(ix::vip_i))
  )

  // ctor
  slvr_dim(
    typename parent_t::ctor_args_t args,
    typename parent_t::rt_params_t const &p
  ) :
    parent_t(args, p)
  {}
};

// 3D version
template <class ct_params_t>
class slvr_dim<
  ct_params_t,
  typename std::enable_if<ct_params_t::n_dims == 3 >::type
> : public slvr_common<ct_params_t>
{
  using parent_t = slvr_common<ct_params_t>;
  using ix = typename ct_params_t::ix;

  protected:
  // inject dimension-independent ranges
  idx_t<3> domain = idx_t<3>({this->mem->grid_size[0], this->mem->grid_size[1], this->mem->grid_size[2]});
  idx_t<2> hrzntl_domain = idx_t<2>({this->mem->grid_size[0], this->mem->grid_size[1]});
  idx_t<2> hrzntl_subdomain = idx_t<2>({this->i, this->j});
  idx_t<3> Cx_domain = idx_t<3>({this->mem->grid_size[0]^h, this->mem->grid_size[1], this->mem->grid_size[2]});
  idx_t<3> Cy_domain = idx_t<3>({this->mem->grid_size[0], this->mem->grid_size[1]^h, this->mem->grid_size[2]});
  idx_t<3> Cz_domain = idx_t<3>({this->mem->grid_size[0], this->mem->grid_size[1], this->mem->grid_size[2]^h});

  enum {vert_dim = 2};
  blitz::TinyVector<int, 3> zero = blitz::TinyVector<int, 3>({0,0,0});
  blitz::thirdIndex vert_idx;
  std::map<int, int> vip_map = std::map<int, int>{{0, ix::vip_i}, {1, ix::vip_j}};

  void vert_grad_fwd(typename parent_t::arr_t &in, typename parent_t::arr_t &out, setup::real_t dz)
  {
    for (auto &bc : this->bcs[2]) bc->fill_halos_sclr(in, this->i, this->j, false);
    out(this->i, this->j, this->k) = ( in(this->i, this->j, this->k+1) - in(this->i, this->j, this->k)) / dz;
    // top and bottom cells are two times lower
    out(this->i, this->j, 0) *= 2; 
    out(this->i, this->j, this->k.last()) *= 2; 
  }

  void vert_grad_cnt(typename parent_t::arr_t &in, typename parent_t::arr_t &out, setup::real_t dz)
  {
    for (auto &bc : this->bcs[2]) bc->fill_halos_sclr(in, this->i, this->j, false);
    out(this->i, this->j, this->k) = ( in(this->i, this->j, this->k+1) - in(this->i, this->j, this->k-1)) / 2./ dz;
    // top and bottom cells are two times lower
    out(this->i, this->j, 0) *= 2; 
    out(this->i, this->j, this->k.last()) *= 2; 
  }

  void smooth(typename parent_t::arr_t &in, typename parent_t::arr_t &out)
  {
    for (auto &bc : this->bcs[2]) bc->fill_halos_sclr(in, this->i, this->j, false);
    out(this->i, this->j, this->k) = 0.25 * (in(this->i, this->j, this->k + 1) + 2 * in(this->i, this->j, this->k) + in(this->i, this->j, this->k - 1));
  }

  auto calc_U() 
    return_macro(,
    sqrt(pow2(this->state(ix::vip_i)) + pow2(this->state(ix::vip_j)))
  )

  // ctor
  slvr_dim(
    typename parent_t::ctor_args_t args, 
    typename parent_t::rt_params_t const &p
  ) : 
    parent_t(args, p)
  {}
};

