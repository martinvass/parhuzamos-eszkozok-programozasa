#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include "cl_utils.h"

static size_t round_up(size_t x, size_t m) {
    return (x + m - 1) / m * m;
}

int main(int argc, char** argv) {
    int n = 1 << 20;

    if (argc >= 2) {
        n = atoi(argv[1]);
        if (n <= 0) {
            printf("Használat: %s <n>\n", argv[0]);
            return 1;
        }
    }

    printf("n = %d\n", n);

    int* data = malloc(sizeof(int) * n);
    int cpu_hist[101];
    memset(cpu_hist, 0, sizeof(cpu_hist));

    srand((unsigned)time(NULL));

    for (int i = 0; i < n; i++) {
        int value = rand() % 101; // 0..100
        data[i] = value;
        cpu_hist[value]++;
    }

    cl_env_t env = cl_create_env();
    cl_int err;

    cl_program program = cl_build_program_from_file(
        env.context,
        env.device,
        "kernels/histogram.cl",
        ""
    );

    cl_kernel kernel = clCreateKernel(program, "histogram", &err);
    cl_check(err, "clCreateKernel");

    cl_mem d_input = clCreateBuffer(
        env.context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int) * n,
        data,
        &err
    );
    cl_check(err, "clCreateBuffer input");

    cl_mem d_hist = clCreateBuffer(
        env.context,
        CL_MEM_READ_WRITE,
        sizeof(int) * 101,
        NULL,
        &err
    );
    cl_check(err, "clCreateBuffer hist");

    int zero_hist[101];
    memset(zero_hist, 0, sizeof(zero_hist));

    cl_check(clEnqueueWriteBuffer(env.queue,
                                  d_hist,
                                  CL_TRUE,
                                  0,
                                  sizeof(int) * 101,
                                  zero_hist,
                                  0, NULL, NULL),
             "write zero hist");

    cl_check(clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_input), "setArg 0");
    cl_check(clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_hist), "setArg 1");
    cl_check(clSetKernelArg(kernel, 2, sizeof(int), &n), "setArg 2");

    size_t local_size = 256;
    size_t global_size = round_up(n, local_size);

    cl_check(clEnqueueNDRangeKernel(env.queue,
                                    kernel,
                                    1,
                                    NULL,
                                    &global_size,
                                    &local_size,
                                    0,
                                    NULL,
                                    NULL),
             "enqueue kernel");

    cl_check(clFinish(env.queue), "finish");


    int gpu_hist[101];

    cl_check(clEnqueueReadBuffer(env.queue,
                                 d_hist,
                                 CL_TRUE,
                                 0,
                                 sizeof(int) * 101,
                                 gpu_hist,
                                 0, NULL, NULL),
             "read hist");

    int error = 0;
    for (int i = 0; i <= 100; i++) {
        if (cpu_hist[i] != gpu_hist[i]) {
            error = 1;
            printf("Eltérés bin %d: CPU=%d GPU=%d\n",
                   i, cpu_hist[i], gpu_hist[i]);
        }
    }

    if (!error)
        printf("Hisztogram OK\n");


    clReleaseMemObject(d_input);
    clReleaseMemObject(d_hist);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    cl_release_env(&env);

    free(data);

    return 0;
}