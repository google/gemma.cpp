
#pragma once
#include <string>
class cl_embed {
    public:
    std::string sample_entry_code = 
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
	"__kernel void sample_entry_code(__global int* from_main_code)\n"
	"{\n"
	"    int myid = get_global_id(0);\n"
	"    feel_free_to_make_functions();\n"
	"}\n"
	;

};