#define read_cell(delta) read_imageui(front, periodic, (gidf + delta) * inv_image_size + d_2).x

__kernel void conway(
    __read_only image2d_t front,
    __write_only image2d_t back,
    float2 inv_image_size
)
{
    const sampler_t periodic = CLK_NORMALIZED_COORDS_TRUE | CLK_FILTER_NEAREST | CLK_ADDRESS_REPEAT;

    int2 gid = (int2)(get_global_id(0), get_global_id(1));
    float2 gidf = (float2)(gid.x, gid.y);
    float2 d_2 = inv_image_size * 0.5f;

    uchar self = read_cell((float2)(0, 0));
    uchar count =
        read_cell((float2)(-1,+1)) + read_cell((float2)(0,+1)) + read_cell((float2)(+1,+1)) +
        read_cell((float2)(-1, 0)) +                           + read_cell((float2)(+1, 0)) +
        read_cell((float2)(-1,-1)) + read_cell((float2)(0,-1)) + read_cell((float2)(+1,-1));

    write_imageui(back, gid, self ?
        (count < 2 || count > 3 ? 0 : 1) :
        (count == 3 ? 1 : 0)
    );
}
