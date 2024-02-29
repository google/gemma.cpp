
#pragma once
//generated with python code
#include <string>
class cl_embed {
    public:
    #ifndef NO_EMBEDDED_CL
std::string sample_entry = 
	"//or\n"
	"\n"
	"\n"
	"\n"
	"void feel_free_to_make_functions()//but you can't use recursive function\n"
	"{\n"
	"    return;\n"
	"}\n"
	"\n"
	"//Kernel entry point, equals to main() function\n"
	"__kernel void sample_entry(__global int* from_main_code)\n"
	"{\n"
	"    int myid = get_global_id(0);\n"
	"    feel_free_to_make_functions();\n"
	"}\n"
	;
#endif
#ifdef NO_EMBEDDED_CL
std::string sample_entry = 
	"CL_C_kernel_files/OpenclCTemplate.cl\n"
	;
#endif

};