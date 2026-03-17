__kernel void histogram(__global const int* input,
                        __global int* hist,
                        const int n)
{
    int gid = get_global_id(0);

    if (gid >= n) return;

    int value = input[gid];

    // value 0..100 között van
    atomic_inc(&hist[value]);
}