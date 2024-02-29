#pragma once
#define NO_EMBEDDED_CL
#include "cl_FACADE.h"
#include "cl_global_custom.h"
#include "cl_embedded.h"
#include <string>
class cl_acc {
 private:

  template <class data_in_T, class data_out_T,class data_type, class... Args>

  void gpgpu_facade(const std::string& CL_C_code,
                    const std::string& CL_C_entry_name, data_in_T*& in_data_P,
                    const data_type& in_data_length, data_out_T*& out_data_P,
                    const data_type& out_data_length,
                    const data_type& core_size, const Args&... args);

  cl_embed CLS=cl_embed();

 public:
  std::vector<std::pair<std::string, std::string>> get_platform_device_list();
  void set_platform_device(const std::string& platform,
                           const std::string& device);
  Platform PF;
  Device DV;
  Context CT;
  cl_acc(const std::string& platform, const std::string& device);
  cl_acc() {}
  ~cl_acc() {}
  // void STFT_TESTER();
};

#ifndef NO_EMBEDDED_CL

template <class data_in_T, class data_out_T, class data_type, class... Args>
void cl_acc::gpgpu_facade(const std::string& CL_C_code,
                          const std::string& CL_C_entry_name,
                          data_in_T*& in_data_P,
                          const data_type& in_data_length,
                          data_out_T*& out_data_P,
                          const data_type& out_data_length,
                          const data_type& core_size, const Args&... args) {
  Kernel KN = cl_facade::create_kernel(CL_C_code, CL_C_entry_name, CT, DV);
  CommandQueue CQ = clboost::make_cq(CT, DV);

  Buffer data_in =
      clboost::make_r_buf<data_in_T>(CT, in_data_length, in_data_P);
  Buffer data_out = clboost::make_w_buf<data_out_T>(CT, out_data_length);

  int arg_set_index = 0;
  KN.setArg(arg_set_index++, data_in);
  KN.setArg(arg_set_index++, data_out);
  (KN.setArg(arg_set_index++, args), ...);  // Fold

  clboost::enq_q(CQ, KN, core_size);
  if (out_data_P == nullptr) {
    out_data_P = new data_out_T[out_data_length];
  }
  clboost::q_read<data_out_T>(CQ, data_out, true, out_data_length, out_data_P);
}

#endif  // !NO_EMBEDDED_CL

#ifdef NO_EMBEDDED_CL

template <class data_in_T, class data_out_T, class data_type, class... Args>
void cl_acc::gpgpu_facade(const std::string& CL_C_path,
                          const std::string& CL_C_entry_name,
                          data_in_T*& in_data_P,
                          const data_type& in_data_length,
                          data_out_T*& out_data_P,
                          const data_type& out_data_length,
                          const data_type& core_size, const Args&... args) {
  Kernel KN =
      cl_facade::create_kernel(CL_C_path, CL_C_entry_name, CT, DV, true);
  CommandQueue CQ = clboost::make_cq(CT, DV);

  Buffer data_in =
      clboost::make_r_buf<data_in_T>(CT, in_data_length, in_data_P);
  Buffer data_out = clboost::make_w_buf<data_out_T>(CT, out_data_length);

  int arg_set_index = 0;
  KN.setArg(arg_set_index++, data_in);
  KN.setArg(arg_set_index++, data_out);
  (KN.setArg(arg_set_index++, args), ...);  // Fold

  clboost::enq_q(CQ, KN, core_size);
  if (out_data_P == nullptr) {
    out_data_P = new data_out_T[out_data_length];
  }
  clboost::q_read<data_out_T>(CQ, data_out, true, out_data_length, out_data_P);
}
#endif  // NO_EMBEDDED_CL