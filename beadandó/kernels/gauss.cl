__kernel void gauss_eliminate(
    __global float* A,
    __global float* b,
    const int N,
    const int k)
{
    int i = get_global_id(0) + k + 1;

    if (i >= N)
        return;

    float pivot = A[k * N + k];
    float factor = A[i * N + k] / pivot;

    for (int j = k; j < N; j++) {
        A[i * N + j] -= factor * A[k * N + j];
    }
    b[i] -= factor * b[k];
}
