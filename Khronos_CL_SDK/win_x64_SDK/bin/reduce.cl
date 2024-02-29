/*
 * Copyright (c) 2020 The Khronos Group Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

int op(int lhs, int rhs);
int work_group_reduce_op(int val);
int sub_group_reduce_op(int val);

int read_local(local int* shared, size_t count, int zero, size_t i)
{
    return i < count ? shared[i] : zero;
}

size_t zmin(size_t a, size_t b)
{
    return a < b ? a : b;
}

kernel void reduce(
    global int* front,
    global int* back,
    local int* shared,
    unsigned long length,
    int zero_elem
)
{
    const size_t lid = get_local_id(0),
                 lsi = get_local_size(0),
                 wid = get_group_id(0),
                 wsi = get_num_groups(0);

    const size_t wg_stride = lsi * 2,
                 valid_count = zmin(wg_stride, (size_t)(length) - wid * wg_stride);

    // Copy real data to local
    event_t read;
    async_work_group_copy(
        shared,
        front + wid * wg_stride,
        valid_count,
        read);
    wait_group_events(1, &read);
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef USE_WORK_GROUP_REDUCE
    int temp = work_group_reduce_op(
        op(
            read_local(shared, valid_count, zero_elem, lid),
            read_local(shared, valid_count, zero_elem, lid + lsi)
        )
    );
    if (lid == 0) back[wid] = temp;
#else // USE_WORK_GROUP_REDUCE
#ifdef USE_SUB_GROUP_REDUCE
    const uint sid = get_sub_group_id();
    const uint ssi = get_sub_group_size();
    const uint slid= get_sub_group_local_id();
    for(int i = valid_count ; i != 0 ; i /= ssi*2)
    {
        int temp = zero_elem;
        if (sid*ssi < valid_count)
        {
            temp = sub_group_reduce_op(
                op(
                    read_local(shared, i, zero_elem, sid * 2 * ssi + slid),
                    read_local(shared, i, zero_elem, sid * 2 * ssi + slid + ssi)
                )
            );
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (sid*ssi < valid_count)
            shared[sid] = temp;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) back[wid] = shared[0];
#else // USE_SUB_GROUP_REDUCE
    for (int i = lsi; i != 0; i /= 2)
    {
        if (lid < i)
            shared[lid] =
                op(
                    read_local(shared, valid_count, zero_elem, lid),
                    read_local(shared, valid_count, zero_elem, lid + i)
                );
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) back[wid] = shared[0];
#endif // USE_SUB_GROUP_REDUCE
#endif // USE_WORK_GROUP_REDUCE
}
