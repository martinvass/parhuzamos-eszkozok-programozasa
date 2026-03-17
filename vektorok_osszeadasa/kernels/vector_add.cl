__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* out,
                         const unsigned int n)
{
    const uint i = (uint)get_global_id(0);
    if (i < n) {
        out[i] = a[i] + b[i];
    }
}