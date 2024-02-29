#pragma once

// OpenCL SDK includes
#include "OpenCLUtilsCpp_Export.h"

#include <CL/Utils/Error.hpp>

// OpenCL includes
#include <CL/opencl.hpp>


namespace cl {
namespace util {

    std::string UTILSCPP_EXPORT read_text_file(const char* const filename,
                                               cl_int* const error);

    std::vector<unsigned char> UTILSCPP_EXPORT
    read_binary_file(const char* const filename, cl_int* const error);

    Program::Binaries UTILSCPP_EXPORT
    read_binary_files(const std::vector<cl::Device>& devices,
                      const char* const program_file_name, cl_int* const error);

    cl_int UTILSCPP_EXPORT
    write_binaries(const cl::Program::Binaries& binaries,
                   const std::vector<cl::Device>& devices,
                   const char* const program_file_name);
}
}
