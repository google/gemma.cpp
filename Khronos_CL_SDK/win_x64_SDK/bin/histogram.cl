uint binary_search(float value, __global float* levels_array, uint bins)
{
    int left_id = 0;
    uint right_id = bins;

    while(left_id < right_id)
    {
        uint mid_id = (left_id + right_id) / 2;
        float level = levels_array[mid_id];
        bool smaller = level <= value;
        left_id = smaller ? mid_id + 1 : left_id;
        right_id = smaller ? right_id : mid_id;
    }

    return max(0, left_id - 1);
}

__kernel void histogram_shared(
    uint input_size,
    uint bins,
    uint items_per_thread,
    __global float* input_array,
    __global float* levels_array,
    __local  uint* block_histogram,
    __global uint* histogram
) {
    size_t gid = get_global_id(0);
    int lid = get_local_id(0);
    uint lsize = get_local_size(0);
    uint channel_per_thread = ( bins + lsize - 1 ) / lsize;

    for(
        uint channel = channel_per_thread * lid;
        channel < min(channel_per_thread * (lid + 1) , bins);
        channel++
    ){
        block_histogram[channel] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);


    // Histogram calculation on shared memory
    for( uint index = 0; index < items_per_thread; index++ )
    {
        size_t element_index = gid * items_per_thread + index;

        if( element_index < input_size )
        {
            float value = input_array[element_index];
            if( levels_array[0] < value && value < levels_array[bins] )
            {
                uint channel = binary_search(value, levels_array, bins);

                atomic_add(&block_histogram[channel], 1);
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Sum up the results to global memory
    for(
        uint channel = channel_per_thread * lid;
        channel < min(channel_per_thread * (lid + 1) , bins);
        channel++
    ){
        if( block_histogram[channel] > 0)
        {
            atomic_add(&histogram[channel], block_histogram[channel]);
        }
    }
}

__kernel void histogram_global(
    uint input_size,
    uint bins,
    __global float* input_array,
    __global float* levels_array,
    __global uint* histogram
) {
    int gid = get_global_id(0);

    float value = input_array[gid];

    if( levels_array[0] < value && value < levels_array[bins] )
        return;

    // binary search
    uint channel = binary_search(input_array[gid], levels_array, bins);

    atomic_add(&histogram[channel], 1u);
}
