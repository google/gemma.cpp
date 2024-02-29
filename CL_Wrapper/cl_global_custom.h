#pragma once
#include <string>
#include <CL/opencl.hpp>
#include "custom_assert.h"
using namespace cl;
namespace clboost {

	Platform get_platform();
	Device get_gpu_device(const Platform& pf);
	Context get_context(const Device& dev);
	CommandQueue make_cq(const Context& ct,const Device& dev);
	Program make_prog(const std::string& path,const Context& ct,const Device& dev);
	Event make_event();

	template <typename T>
	Buffer make_r_buf(const Context& ct, const int& size, std::vector<T>& vec);
	template <typename T>
	Buffer make_r_buf(const Context& ct, const int& size, T* data);
	template <typename W>
	Buffer make_w_buf(const Context& ct, const int& size);

	Kernel make_kernel(const Program& prog,const std::string& class_name);

	template <typename... Args>
	void set_args(Kernel& kn, const Args ... args);

	void enq_q(CommandQueue& q, const Kernel& kernel, Event& this_event, const vector<Event>& wait_ev, const int global_size, const int local_size = NULL);
	void enq_q(CommandQueue& q, const Kernel& kernel, const int global_size, const int local_size = NULL);

	template<typename P>
	void q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, vector<P>& data);
	
	template<typename P>
	void q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, P* data);

	//template<typename P>
	//void q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, vector<P>& data, Event& this_event, const vector<Event>& wait_ev);

}


template <typename T>
Buffer
clboost::make_r_buf(const Context& ct, const int& size, std::vector<T>& vec)
{
	return Buffer(ct, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, vec.data());
}

template <typename T>
Buffer
clboost::make_r_buf(const Context& ct, const int& size, T* data)
{
	return Buffer(ct, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(T) * size, data);
}
template<typename W>
Buffer
clboost::make_w_buf(const Context& ct, const int& size)
{
	return Buffer(ct, CL_MEM_WRITE_ONLY, sizeof(W) * size);
}


template <typename... Args>
void
clboost::set_args(Kernel& kn, const Args ... args)
{
	int index = 0;
	(ASSERT_EQ(kn.setArg(index++, args),0), ...);
	ASSERT_UEQ(index, 0);
}

template <typename P>
void
clboost::q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, vector<P>& data)
{
	q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data.data());
}
//
//template <typename P>
//void
//clboost::q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, vector<P>& data, Event& this_event, const vector<Event>& wait_ev)
//{
//	q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data.data(), wait_ev, this_event);
//}

template<typename P>
void
clboost::q_read(CommandQueue& q, Buffer& wbuf, const bool check_dirct, const int& size, P* data)
{	
	ASSERT_EQ(q.enqueueReadBuffer(wbuf, (check_dirct ? CL_TRUE : CL_FALSE), 0, sizeof(P) * size, data),0);
}
