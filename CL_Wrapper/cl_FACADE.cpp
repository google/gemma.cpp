
#include "cl_FACADE.h"


using namespace clboost;
Kernel
cl_facade::create_kernel(const std::string& path, const std::string& class_name, const Context& ct, const Device& dev)
{
	return make_kernel(make_prog(path, ct, dev), class_name);
}

Kernel
cl_facade::create_kernel(const std::string& path, const std::string& class_name, const Context& ct, const Device& dev, bool to_text)
{
	return make_kernel(make_prog(path, ct, dev,to_text), class_name);
}