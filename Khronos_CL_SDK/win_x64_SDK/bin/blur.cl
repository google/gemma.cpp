kernel void blur_box(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    // coordinates of the pixel to work on
    const int2 coord = { get_global_id(0), get_global_id(1) };

    uint4 sum = 0;
    uint num = 0;
    int2 shift;
    for (shift.x = -size; shift.x <= size; ++shift.x)
        for (shift.y = -size; shift.y <= size; ++shift.y) {
            int2 cur = coord + shift;
            if ((0 <= cur.x) && (cur.x < width) && (0 <= cur.y) && (cur.y < height)) {
                ++num;
                sum += read_imageui(input_image, cur);
            }
        }
    write_imageui(output_image, coord, (sum + num / 2) / num);
}


kernel void blur_box_horizontal(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size
)
{
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    uint4 sum = 0;
    uint num = 0;
    int2 shift = 0;
    for (shift.x = -size; shift.x <= size; ++shift.x) {
        int2 cur = coord + shift;
        if ((0 <= cur.x) && (cur.x < width)) {
            ++num;
            sum += read_imageui(input_image, cur);
        }
    }
    write_imageui(output_image, coord, (sum + num / 2) / num);
}

kernel void blur_box_vertical(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size
)
{
    const int width = get_global_size(0);
    const int height = get_global_size(1);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    uint4 sum = 0;
    uint num = 0;
    int2 shift = 0;
    for (shift.y = -size; shift.y <= size; ++shift.y) {
        int2 cur = coord + shift;
        if ((0 <= cur.y) && (cur.y < height)) {
            ++num;
            sum += read_imageui(input_image, cur);
        }
    }
    write_imageui(output_image, coord, (sum + num / 2) / num);
}


kernel void blur_kernel_horizontal(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    constant float * kern
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    float4 sum = 0;
    float weight = 0;
    int2 shift = 0;
    for (shift.x = -size; shift.x <= size; ++shift.x) {
        int2 cur = coord + shift;
        if ((0 <= cur.x) && (cur.x < width)) {
            const float w = kern[size + shift.x];
            weight += w;
            sum += convert_float4(read_imageui(input_image, cur)) * w;
        }
    }
    uint4 res = convert_uint4(round(sum / weight));
    write_imageui(output_image, coord, res);
}

kernel void blur_kernel_vertical(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    constant float * kern
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    float4 sum = 0;
    float weight = 0;
    int2 shift = 0;
    for (shift.y = -size; shift.y <= size; ++shift.y) {
        int2 cur = coord + shift;
        if ((0 <= cur.y) && (cur.y < height)) {
            const float w = kern[size + shift.y];
            weight += w;
            sum += convert_float4(read_imageui(input_image, cur)) * w;
        }
    }
    write_imageui(output_image, coord, convert_uint4(round(sum / weight)));
}


kernel void blur_box_horizontal_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    local uchar4 * line
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    const int grs = get_local_size(0);
    const int lid = get_local_id(0);
    // coordinates of the leftmost and rightmost pixels needed for the workgroup
    const int start = get_group_id(0) * grs;
    const int2 start_coord = { max(start - size, 0), coord.y };
    const int2 end_coord = { min(start + grs + size, width - 1), coord.y};

    // copy all pixels needed for workgroup into local memory
    int2 cur = start_coord + (int2)(lid, 0);
    uint pos = lid;
    while (cur.x <= end_coord.x) {
        line[pos] = convert_uchar4(read_imageui(input_image, cur));
        cur.x += grs;
        pos += grs;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // blur
    if (coord.x < width) {
        uint4 sum = 0;
        uint num = 0;
        pos = lid + (start - size - start_coord.x);
        for (cur.x = coord.x - size; cur.x <= coord.x + size; ++cur.x, ++pos)
            if ((0 <= cur.x) && (cur.x < width)) {
                ++num;
                sum += convert_uint4(line[pos]);
            }
        write_imageui(output_image, coord, (sum + num / 2) / num);
    }
}

kernel void blur_box_vertical_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    local uchar4 * line
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    const int grs = get_local_size(1);
    const int lid = get_local_id(1);
    // coordinates of the topmost and lowest pixels needed for the workgroup
    const int start = get_group_id(1) * grs;
    const int2 start_coord = { coord.x, max(start - size, 0) };
    const int2 end_coord = { coord.x, min(start + grs + size, height - 1) };

    // copy all pixels needed for workgroup into local memory
    int2 cur = start_coord + (int2)(0, lid);
    uint pos = lid;
    while (cur.y <= end_coord.y) {
        line[pos] = convert_uchar4(read_imageui(input_image, cur));
        cur.y += grs;
        pos += grs;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // blur
    if (coord.y < height) {
        uint4 sum = 0;
        uint num = 0;
        pos = lid + (start - size - start_coord.y);
        for (cur.y = coord.y - size; cur.y <= coord.y + size; ++cur.y, ++pos)
            if ((0 <= cur.y) && (cur.y < height)) {
                ++num;
                sum += convert_uint4(line[pos]);
            }
        write_imageui(output_image, coord, (sum + num / 2) / num);
    }
}


kernel void blur_kernel_horizontal_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    constant float * kern,
    local uchar4 * line
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    const int grs = get_local_size(0);
    const int lid = get_local_id(0);
    // coordinates of the leftmost and rightmost pixels needed for the workgroup
    const int start = get_group_id(0) * grs;
    const int2 start_coord = { max(start - size, 0), coord.y };
    const int2 end_coord = { min(start + grs + size, width - 1), coord.y};

    // copy all pixels needed for workgroup into local memory
    int2 cur = start_coord + (int2)(lid, 0);
    uint pos = lid;
    while (cur.x <= end_coord.x) {
        line[pos] = convert_uchar4(read_imageui(input_image, cur));
        cur.x += grs;
        pos += grs;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // blur
    if (coord.x < width) {
        float4 sum = 0;
        float weight = 0;
        pos = lid + (start - size - start_coord.x);
        for (int shift = -size; shift <= size; ++shift, ++pos) {
            cur.x = coord.x + shift;
            if ((0 <= cur.x) && (cur.x < width)) {
                float w = kern[size + shift];
                weight += w;
                sum += convert_float4(line[pos]) * w;
            }
        }
        write_imageui(output_image, coord, convert_uint4(round(sum / weight)));
    }
}

kernel void blur_kernel_vertical_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    constant float * kern,
    local uchar4 * line
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    const int grs = get_local_size(1);
    const int lid = get_local_id(1);
    // coordinates of the topmost and lowest pixels needed for the workgroup
    const int start = get_group_id(1) * grs;
    const int2 start_coord = { coord.x, max(start - size, 0) };
    const int2 end_coord = { coord.x, min(start + grs + size, height - 1) };

    // copy all pixels needed for workgroup into local memory
    int2 cur = start_coord + (int2)(0, lid);
    uint pos = lid;
    while (cur.y <= end_coord.y) {
        line[pos] = convert_uchar4(read_imageui(input_image, cur));
        cur.y += grs;
        pos += grs;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // blur
    if (coord.y < height) {
        float4 sum = 0;
        float weight = 0;
        pos = lid + (start - size - start_coord.y);
        for (int shift = -size; shift <= size; ++shift, ++pos) {
            cur.y = coord.y + shift;
            if ((0 <= cur.y) && (cur.y < height)) {
                float w = kern[size + shift];
                weight += w;
                sum += convert_float4(line[pos]) * w;
            }
        }
        write_imageui(output_image, coord, convert_uint4(round(sum / weight)));
    }
}


#if defined(USE_SUBGROUP_EXCHANGE_RELATIVE) || defined(USE_SUBGROUP_EXCHANGE)

kernel void blur_box_horizontal_subgroup_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    if (coord.x < width) {
        uint4 pixel = 0;
        uint4 sum = 0;
        uint num = 0;

        // initial read
        int2 cur = coord + (int2)(size, 0);
        if (cur.x < width) {
            pixel = read_imageui(input_image, cur);
            sum = pixel;
            num = 1;
        }
        // shifts and reads
        const uint sglid = get_sub_group_local_id();
        const uint shift = (sglid != 0);
        for (int i = size - 1; i >= -size; --i) {
            --cur.x;
#if defined(USE_SUBGROUP_EXCHANGE_RELATIVE)
            pixel = (uint4)(sub_group_shuffle_up(pixel.s0, shift),
                            sub_group_shuffle_up(pixel.s1, shift),
                            sub_group_shuffle_up(pixel.s2, shift),
                            sub_group_shuffle_up(pixel.s3, shift));
#elif defined(USE_SUBGROUP_EXCHANGE)
            pixel = (uint4)(sub_group_shuffle(pixel.s0, sglid - shift),
                            sub_group_shuffle(pixel.s1, sglid - shift),
                            sub_group_shuffle(pixel.s2, sglid - shift),
                            sub_group_shuffle(pixel.s3, sglid - shift));
#endif
            if ((cur.x >= 0) && (cur.x < width)) {
                if (!shift) // 0th workitem reads new pixel
                    pixel = read_imageui(input_image, cur);
                sum += pixel;
                ++num;
            }
        }

        write_imageui(output_image, coord, (sum + num / 2) / num);
    }
}

kernel void blur_box_vertical_subgroup_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    if (coord.y < height) {
        uint4 pixel = 0;
        uint4 sum = 0;
        uint num = 0;

        // initial read
        int2 cur = coord + (int2)(0, size);
        if (cur.y < height) {
            pixel = read_imageui(input_image, cur);
            sum = pixel;
            num = 1;
        }
        // shifts and reads
        const uint sglid = get_sub_group_local_id();
        const uint shift = (sglid != 0);
        for (int i = size - 1; i >= -size; --i) {
            --cur.y;
#if defined(USE_SUBGROUP_EXCHANGE_RELATIVE)
            pixel = (uint4)(sub_group_shuffle_up(pixel.s0, shift),
                            sub_group_shuffle_up(pixel.s1, shift),
                            sub_group_shuffle_up(pixel.s2, shift),
                            sub_group_shuffle_up(pixel.s3, shift));
#elif defined(USE_SUBGROUP_EXCHANGE)
            pixel = (uint4)(sub_group_shuffle(pixel.s0, sglid - shift),
                            sub_group_shuffle(pixel.s1, sglid - shift),
                            sub_group_shuffle(pixel.s2, sglid - shift),
                            sub_group_shuffle(pixel.s3, sglid - shift));
#endif
            if ((cur.y >= 0) && (cur.y < height)) {
                if (!shift) // 0th workitem reads new pixel
                    pixel = read_imageui(input_image, cur);
                sum += pixel;
                ++num;
            }
        }

        write_imageui(output_image, coord, (sum + num / 2) / num);
    }
}


kernel void blur_kernel_horizontal_subgroup_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    constant float * kern
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    if (coord.x < width) {
        uint4 pixel = 0;
        float4 sum = 0;
        float weight = 0;

        // initial read
        int2 cur = coord + (int2)(size, 0);
        if (cur.x < width) {
            pixel = read_imageui(input_image, cur);
            weight = kern[2 * size];
            sum = convert_float4(pixel) * weight;
        }
        // shifts and reads
        const uint sglid = get_sub_group_local_id();
        const uint shift = (sglid != 0);
        for (int i = size - 1; i >= -size; --i) {
            --cur.x;
#if defined(USE_SUBGROUP_EXCHANGE_RELATIVE)
            pixel = (uint4)(sub_group_shuffle_up(pixel.s0, shift),
                            sub_group_shuffle_up(pixel.s1, shift),
                            sub_group_shuffle_up(pixel.s2, shift),
                            sub_group_shuffle_up(pixel.s3, shift));
#elif defined(USE_SUBGROUP_EXCHANGE)
            pixel = (uint4)(sub_group_shuffle(pixel.s0, sglid - shift),
                            sub_group_shuffle(pixel.s1, sglid - shift),
                            sub_group_shuffle(pixel.s2, sglid - shift),
                            sub_group_shuffle(pixel.s3, sglid - shift));
#endif
            if ((cur.x >= 0) && (cur.x < width)) {
                if (!shift) // 0th workitem reads new pixel
                    pixel = read_imageui(input_image, cur);
                float w = kern[size + i];
                weight += w;
                sum += convert_float4(pixel) * w;
            }
        }

        write_imageui(output_image, coord, convert_uint4(round(sum / weight)));
    }
}

kernel void blur_kernel_vertical_subgroup_exchange(
    read_only image2d_t input_image,
    write_only image2d_t output_image,
    int size,
    constant float * kern
)
{
    const int width = get_image_width(input_image);
    const int height = get_image_height(input_image);
    const int2 coord = { get_global_id(0), get_global_id(1) };

    if (coord.y < height) {
        uint4 pixel = 0;
        float4 sum = 0;
        float weight = 0;

        // initial read
        int2 cur = coord + (int2)(0, size);
        if (cur.y < height) {
            pixel = read_imageui(input_image, cur);
            weight = kern[2 * size];
            sum = convert_float4(pixel) * weight;
        }
        // shifts and reads
        const uint sglid = get_sub_group_local_id();
        const uint shift = (sglid != 0);
        for (int i = size - 1; i >= -size; --i) {
            --cur.y;
#if defined(USE_SUBGROUP_EXCHANGE_RELATIVE)
            pixel = (uint4)(sub_group_shuffle_up(pixel.s0, shift),
                            sub_group_shuffle_up(pixel.s1, shift),
                            sub_group_shuffle_up(pixel.s2, shift),
                            sub_group_shuffle_up(pixel.s3, shift));
#elif defined(USE_SUBGROUP_EXCHANGE)
            pixel = (uint4)(sub_group_shuffle(pixel.s0, sglid - shift),
                            sub_group_shuffle(pixel.s1, sglid - shift),
                            sub_group_shuffle(pixel.s2, sglid - shift),
                            sub_group_shuffle(pixel.s3, sglid - shift));
#endif
            if ((cur.y >= 0) && (cur.y < height)) {
                if (!shift) // 0th workitem reads new pixel
                    pixel = read_imageui(input_image, cur);
                float w = kern[size + i];
                sum += convert_float4(pixel) * w;
                weight += w;
            }
        }

        write_imageui(output_image, coord, convert_uint4(round(sum / weight)));
    }
}

#endif // USE_SUBGROUP_EXCHANGE_RELATIVE || USE_SUBGROUP_EXCHANGE
