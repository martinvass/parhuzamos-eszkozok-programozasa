#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void check_cl(cl_int err, const char* msg)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "[ERROR] %s (code: %d)\n", msg, err);
        exit(1);
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Hasznalat: %s matrix.txt [-b]\n", argv[0]);
        return 1;
    }
    char* file = argv[1];
    int bench = 0;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-b") == 0) bench = 1;
    }

    FILE* f = fopen(file, "r");
    if (!f) {
        perror("fopen");
        return 1;
    }

    int N;
    fscanf(f, "%d", &N);

    float* A = (float*)malloc(N * N * sizeof(float));
    float* b = (float*)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(f, "%f", &A[i * N + j]);
    for (int i = 0; i < N; i++)
        fscanf(f, "%f", &b[i]);
    fclose(f);

    cl_int err;
    cl_platform_id platform_id;
    cl_uint n_platforms;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    check_cl(err, "clGetPlatformIDs");

    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &n_devices);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, &n_devices);
        check_cl(err, "clGetDeviceIDs (sem GPU, sem CPU)");
        printf("[INFO] GPU nem talalhato, CPU eszkoz hasznalva.\n");
    }

    char dev_name[256];
    clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(dev_name), dev_name, NULL);
    printf("[INFO] Eszkoz: %s\n", dev_name);

    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    check_cl(err, "clCreateContext");

    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    cl_command_queue queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);
    #pragma GCC diagnostic pop
    check_cl(err, "clCreateCommandQueue");

    int loader_err;
    const char* kernel_source = load_kernel_source("kernels/gauss.cl", &loader_err);
    if (loader_err != 0) {
        fprintf(stderr, "[ERROR] Kernel forraskod betoltese sikertelen!\n");
        return 1;
    }

    cl_program program = clCreateProgramWithSource(
        context, 1, &kernel_source, NULL, &err);
    check_cl(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device_id, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t log_size;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char* build_log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, log_size + 1, build_log, NULL);
        fprintf(stderr, "[BUILD ERROR]\n%s\n", build_log);
        free(build_log);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "gauss_eliminate", &err);
    check_cl(err, "clCreateKernel");

    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_WRITE,
        N * N * sizeof(float), NULL, &err);
    check_cl(err, "clCreateBuffer A");

    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_WRITE,
        N * sizeof(float), NULL, &err);
    check_cl(err, "clCreateBuffer b");

    err = clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0,
        N * N * sizeof(float), A, 0, NULL, NULL);
    check_cl(err, "clEnqueueWriteBuffer A");

    err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
        N * sizeof(float), b, 0, NULL, NULL);
    check_cl(err, "clEnqueueWriteBuffer b");

    struct timespec t1, t2;
    if (bench) clock_gettime(CLOCK_MONOTONIC, &t1);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    clSetKernelArg(kernel, 2, sizeof(int), &N);

    size_t local_work_size = 256;

    for (int k = 0; k < N - 1; k++) {
        int rows_to_process = N - k - 1;
        if (rows_to_process <= 0) break;

        clSetKernelArg(kernel, 3, sizeof(int), &k);

        size_t global_work_size = ((rows_to_process + local_work_size - 1)
                                   / local_work_size) * local_work_size;

        err = clEnqueueNDRangeKernel(
            queue, kernel, 1, NULL,
            &global_work_size, &local_work_size,
            0, NULL, NULL);
        check_cl(err, "clEnqueueNDRangeKernel");
    }

    clFinish(queue);

    if (bench) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        double elapsed = (t2.tv_sec - t1.tv_sec) +
                         (t2.tv_nsec - t1.tv_nsec) / 1e9;
        printf("OCL_TIME %f\n", elapsed);
    }

    err = clEnqueueReadBuffer(queue, d_A, CL_TRUE, 0,
        N * N * sizeof(float), A, 0, NULL, NULL);
    check_cl(err, "clEnqueueReadBuffer A");

    err = clEnqueueReadBuffer(queue, d_b, CL_TRUE, 0,
        N * sizeof(float), b, 0, NULL, NULL);
    check_cl(err, "clEnqueueReadBuffer b");

    float* x = (float*)malloc(N * sizeof(float));
    for (int i = N - 1; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < N; j++)
            sum -= A[i * N + j] * x[j];
        x[i] = sum / A[i * N + i];
    }

    if (!bench) {
        for (int i = 0; i < N; i++) {
            printf("%d %.6f\n", i, x[i]);
        }
    }

    clReleaseMemObject(d_A);
    clReleaseMemObject(d_b);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    free(A);
    free(b);
    free(x);

    return 0;
}
