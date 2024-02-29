#pragma once
#include "cl_global_custom.h"
namespace cl_facade
{
	Kernel create_kernel(const std::string& path, const std::string& entry_name, const Context& ct, const Device& dev);
}
