#pragma once

//TODO: make these functions return arrays

template <class ct_params_t>
void slvr_common<ct_params_t>::subsidence(const typename parent_t::arr_t &val) // large-scale vertical wind
{
  const auto &ijk = this->ijk;
  if(params.subsidence)
  {
    tmp1(ijk) = val(ijk);
    this->vert_grad_fwd(tmp1, F, params.dz); 
    F(ijk).reindex(this->zero) *= - (*params.w_LS)(this->vert_idx);
    //this->smooth(tmp1, F);
  }
  else
    F(ijk)=0.;
}
