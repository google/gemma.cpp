#pragma once
#include "cl_FACADE.h"
#include "cl_global_custom.h"
#include "cl_inside.h"
#include <thread>
#include <vector>
#define to_big_radix_2(var) pow(2,std::ceil(log2(var)));
typedef unsigned long long ma_uint64;
class OPENCL_ACC{
private:
	template<class data_in_T, class data_out_T, class ... Args>
	void gpgpu_facade(
		const std::string& CL_C_code, 
		data_in_T*& in_data_P, 
		const ma_uint64& in_data_length,
		data_out_T*& out_data_P,
		const ma_uint64& out_data_length,
		const ma_uint64& core_size,
		const Args& ... args);

	template<class data_in_T, class data_out_T, class ... Args>
	void gpgpu_facade_read_clfile(
		const std::string& CL_C_code,
		data_in_T*& in_data_P,
		const ma_uint64& in_data_length,
		data_out_T*& out_data_P,
		const ma_uint64& out_data_length,
		const ma_uint64& core_size,
		const Args& ... args);

	CL_INSIDE *CLS;
	[[nodiscard]]
	cl_float2* bit_reverse(float* data_array, const int& data_length_radix_2);
	void butterfly_stage_radix_2(cl_float2* data, const int& data_length_radix_2, float* data_out);

	[[nodiscard]]
	cl_float2* overlap_and_extend_for_STFT(float* data_origin, const ma_uint64& origin_length, const ma_uint64& overlaped_length, const int& radix_2_size, const int& overlap_frame, const int& both_side_z_padding_size);
	
	void window_STFT(cl_float2* overlap_array, const ma_uint64& frame_length, const int& window_radix_size);

	void bit_reverse_STFT(cl_float2* overlap_array,const ma_uint64& frame_length, const int& window_radix_size);

	void butterfly_STFT(cl_float2* overlap_array, const ma_uint64& frame_length, const int& window_radix_2_size);
	[[nodiscard]]//deletes overlap_array here
	float* power_them(cl_float2* overlap_array, const ma_uint64& frame_length, const int& window_radix_size);
	[[nodiscard]]//deletes powered_STFT here
	cl_float3* three_bander(float* powered_STFT, const int& window_radix_size, int& low, int& mid, int& high, const int& quot);
	cl_float3* three_divide_and_conquer(
		float*& low, const int& low_range,
		float*& mid, const int& mid_range,
		float*& high, const int& high_range,
		const int& number_of_owner);
	
	void to_dbfs(cl_float3* data, const int& window_radix_size, const int& low, const int& mid, const int& high, const int& quot );
	bool ready = false;
public:
	Platform PF;
	Device DV;
	Context CT;
	OPENCL_ACC();
	OPENCL_ACC(const std::string& platform, const std::string& device);
	~OPENCL_ACC();
	void thread_worker(Kernel& KN, CommandQueue& CQ, float*& data, const int& number_of_owner, const int& range);
	std::vector<std::pair<std::string, std::string>> get_platform_device_list();
	void set_platform_device(const std::string& platform, const std::string& device);
	//void cl_fft(float* data_array, const int& data_length_radix_2);
	float* cl_STFT(float* full_frame, const ma_uint64& full_length, const int& window_radix_2, const double& overlap_ratio, const int& both_side_z_padding_size, int& number_of_index);
	//void STFT_TESTER();
};



template <class data_in_T, class data_out_T, class ... Args>
void 
OPENCL_ACC::gpgpu_facade(
	const std::string& CL_C_code,
	data_in_T*& in_data_P,
	const ma_uint64& in_data_length,
	data_out_T*& out_data_P,
	const ma_uint64& out_data_length,
	const ma_uint64& core_size,
	const Args& ... args)
{
	Kernel KN = cl_facade::create_kernel(CL_C_code, "entry_point", CT, DV);
	CommandQueue CQ = clboost::make_cq(CT, DV);

	Buffer data_in = clboost::make_r_buf<data_in_T>(CT, in_data_length, in_data_P);
	Buffer data_out = clboost::make_w_buf<data_out_T>(CT, out_data_length);

	int arg_set_index = 0;
	KN.setArg(arg_set_index++, data_in);
	KN.setArg(arg_set_index++, data_out);
	(KN.setArg(arg_set_index++, args), ...);//Fold

	clboost::enq_q(CQ, KN, core_size);
	if (out_data_P == nullptr) {
		out_data_P = new data_out_T[out_data_length];
	}
	clboost::q_read<data_out_T>(CQ, data_out, true, out_data_length, out_data_P);
	
}

template <class data_in_T, class data_out_T, class ... Args>
void
OPENCL_ACC::gpgpu_facade_read_clfile(
	const std::string& CL_C_path,
	data_in_T*& in_data_P,
	const ma_uint64& in_data_length,
	data_out_T*& out_data_P,
	const ma_uint64& out_data_length,
	const ma_uint64& core_size,
	const Args& ... args)
{
	Kernel KN = cl_facade::create_kernel(CL_C_path, "entry_point", CT, DV, true);
	CommandQueue CQ = clboost::make_cq(CT, DV);

	Buffer data_in = clboost::make_r_buf<data_in_T>(CT, in_data_length, in_data_P);
	Buffer data_out = clboost::make_w_buf<data_out_T>(CT, out_data_length);

	int arg_set_index = 0;
	KN.setArg(arg_set_index++, data_in);
	KN.setArg(arg_set_index++, data_out);
	(KN.setArg(arg_set_index++, args), ...);//Fold

	clboost::enq_q(CQ, KN, core_size);
	if (out_data_P == nullptr) {
		out_data_P = new data_out_T[out_data_length];
	}
	clboost::q_read<data_out_T>(CQ, data_out, true, out_data_length, out_data_P);

}