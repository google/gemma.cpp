__kernel void saxpy(float a,
                    __global float* x,
                    __global float* y)
{
    int gid = get_global_id(0);

    y[gid] = fma(a, x[gid], y[gid]);
}
