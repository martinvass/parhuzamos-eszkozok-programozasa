// Egyszeru redukcio csoporton belul (local mem) -> csoportonkent 1 reszosszeg.
__kernel void reduce_sum(__global const float* in,
                         __global float* out,
                         const int n,
                         __local float* sdata)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);
    int lsize = get_local_size(0);

    float x = (gid < n) ? in[gid] : 0.0f;
    sdata[lid] = x;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sdata[lid] += sdata[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) out[group] = sdata[0];
}

__kernel void reduce_sum_sqdiff(__global const float* in,
                                __global float* out,
                                const float mean,
                                const int n,
                                __local float* sdata)
{
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group = get_group_id(0);
    int lsize = get_local_size(0);

    float v = 0.0f;
    if (gid < n) {
        float d = in[gid] - mean;
        v = d * d;
    }
    sdata[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int stride = lsize / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            sdata[lid] += sdata[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) out[group] = sdata[0];
}