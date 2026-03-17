#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "cl_utils.h"

static size_t round_up(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}

static float cpu_stddev(const float* a, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i];
    double mean = sum / (double)n;

    double s2 = 0.0;
    for (int i = 0; i < n; i++) {
        double d = (double)a[i] - mean;
        s2 += d * d;
    }
    double var = s2 / (double)n; // populacios szoras (N-nel osztunk)
    return (float)sqrt(var);
}

// reszosszegek osszegzese hoston
static float sum_partials(const float* p, int m) {
    double s = 0.0;
    for (int i = 0; i < m; i++) s += p[i];
    return (float)s;
}

int main(int argc, char** argv) {
    int n = 1 << 20; // default
    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            fprintf(stderr, "Hasznalat: %s <n>\n", argv[0]);
            return 1;
        }
    }

    // veletlen adatok
    float* a = (float*)malloc(sizeof(float) * (size_t)n);
    if (!a) {
        fprintf(stderr, "Memoria hiba.\n");
        return 1;
    }
    srand((unsigned)time(NULL));
    for (int i = 0; i < n; i++) {
        // [0,1) veletlen float
        a[i] = (float)rand() / (float)RAND_MAX;
    }

    // OpenCL init
    cl_env_t env = cl_create_env();

    // Program + kernelek
    cl_program program = cl_build_program_from_file(env.context, env.device,
                                                    "kernels/reduction.cl", "");
    cl_int err = CL_SUCCESS;
    cl_kernel k_sum = clCreateKernel(program, "reduce_sum", &err);
    cl_check(err, "clCreateKernel(reduce_sum)");
    cl_kernel k_sq = clCreateKernel(program, "reduce_sum_sqdiff", &err);
    cl_check(err, "clCreateKernel(reduce_sum_sqdiff)");

    const size_t lsize = 256;
    const size_t gsize = round_up((size_t)n, lsize);
    const int num_groups = (int)(gsize / lsize);

    // Bufferek
    cl_mem d_in = clCreateBuffer(env.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 sizeof(float) * (size_t)n, a, &err);
    cl_check(err, "clCreateBuffer(d_in)");

    cl_mem d_part = clCreateBuffer(env.context, CL_MEM_READ_WRITE,
                                   sizeof(float) * (size_t)num_groups, NULL, &err);
    cl_check(err, "clCreateBuffer(d_part)");

    float* partials = (float*)malloc(sizeof(float) * (size_t)num_groups);
    if (!partials) {
        fprintf(stderr, "Memoria hiba.\n");
        return 1;
    }

    // 1) SUM -> MEAN
    cl_check(clSetKernelArg(k_sum, 0, sizeof(cl_mem), &d_in), "setArg k_sum 0");
    cl_check(clSetKernelArg(k_sum, 1, sizeof(cl_mem), &d_part), "setArg k_sum 1");
    cl_check(clSetKernelArg(k_sum, 2, sizeof(int), &n), "setArg k_sum 2");
    cl_check(clSetKernelArg(k_sum, 3, sizeof(float) * lsize, NULL), "setArg k_sum 3(local)");

    cl_check(clEnqueueNDRangeKernel(env.queue, k_sum, 1, NULL, &gsize, &lsize, 0, NULL, NULL),
             "enqueue k_sum");
    cl_check(clFinish(env.queue), "finish after k_sum");

    cl_check(clEnqueueReadBuffer(env.queue, d_part, CL_TRUE, 0,
                                 sizeof(float) * (size_t)num_groups, partials,
                                 0, NULL, NULL),
             "read d_part (sum partials)");

    float sum = sum_partials(partials, num_groups);
    float mean = sum / (float)n;

    // 2) SUM((x-mean)^2) -> VAR -> STDDEV
    cl_check(clSetKernelArg(k_sq, 0, sizeof(cl_mem), &d_in), "setArg k_sq 0");
    cl_check(clSetKernelArg(k_sq, 1, sizeof(cl_mem), &d_part), "setArg k_sq 1");
    cl_check(clSetKernelArg(k_sq, 2, sizeof(float), &mean), "setArg k_sq 2(mean)");
    cl_check(clSetKernelArg(k_sq, 3, sizeof(int), &n), "setArg k_sq 3");
    cl_check(clSetKernelArg(k_sq, 4, sizeof(float) * lsize, NULL), "setArg k_sq 4(local)");

    cl_check(clEnqueueNDRangeKernel(env.queue, k_sq, 1, NULL, &gsize, &lsize, 0, NULL, NULL),
             "enqueue k_sq");
    cl_check(clFinish(env.queue), "finish after k_sq");

    cl_check(clEnqueueReadBuffer(env.queue, d_part, CL_TRUE, 0,
                                 sizeof(float) * (size_t)num_groups, partials,
                                 0, NULL, NULL),
             "read d_part (sqdiff partials)");

    float s2 = sum_partials(partials, num_groups);
    float var = s2 / (float)n;          // populacios variancia
    float stddev_gpu = sqrtf(var);

    // CPU referencia (ellenorzes)
    float stddev_cpu = cpu_stddev(a, n);
    float abs_err = fabsf(stddev_gpu - stddev_cpu);

    printf("n=%d\n", n);
    printf("mean (GPU)   = %.8f\n", mean);
    printf("stddev (GPU) = %.8f\n", stddev_gpu);
    printf("stddev (CPU) = %.8f\n", stddev_cpu);
    printf("abs_err      = %.8g\n", abs_err);

    // cleanup
    free(partials);
    clReleaseMemObject(d_part);
    clReleaseMemObject(d_in);
    clReleaseKernel(k_sq);
    clReleaseKernel(k_sum);
    clReleaseProgram(program);
    cl_release_env(&env);
    free(a);

    return 0;
}