//No_Embed         <-----This reservation word will allow you to ignore the line when embedded.
//or
//-ne



void feel_free_to_make_functions()//but you can't use recursive function
{
    return;
}

//Kernel entry point, equals to main() function
__kernel void sample_entry_code(__global int* from_main_code)
{
    int myid = get_global_id(0);
    printf("test %d sample",from_main_code[myid]);//also printf will be ignored in embedded
    feel_free_to_make_functions();
    printf("this line will be ignored after embedding");//-ne
}
