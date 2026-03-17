#include "vector_add.h"
#include "cl_utils.h"
#include <stdio.h>
#include <stdlib.h>

static const char* KERNEL_SRC =
"__kernel void vector_add(__global const float* a,              \n"
"                         __global const float* b,              \n"
"                         __global float* out,                  \n"
"                         const unsigned int n)                 \n"
"{                                                             \n"
"    const uint i = (uint)get_global_id(0);                     \n"
"    if (i < n) {                                               \n"
"        out[i] = a[i] + b[i];                                  \n"
"    }                                                         \n"
"}                                                             \n";

static int vector_add_opencl(const float* a, const float* b, float* out, size_t n) {
  ClContext ctx;
  int rc = cl_init(&ctx);
  if (rc != 0) return rc;

  cl_int err = CL_SUCCESS;

  // Program a beégetett kernel forrásból
  const char* sources[] = { KERNEL_SRC };
  size_t lengths[] = { 0 }; // 0 -> null-terminált string

  cl_program program = clCreateProgramWithSource(ctx.context, 1, sources, lengths, &err);
  if (!program || err != CL_SUCCESS) {
    fprintf(stderr, "clCreateProgramWithSource failed: %s (%d)\n", cl_errstr(err), err);
    cl_cleanup(&ctx);
    return -11;
  }

  err = clBuildProgram(program, 1, &ctx.device, NULL, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clBuildProgram failed: %s (%d)\n", cl_errstr(err), err);

    // build log
    size_t log_size = 0;
    clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
    if (log_size > 1) {
      char* log = (char*)malloc(log_size);
      if (log) {
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "OpenCL build log:\n%s\n", log);
        free(log);
      }
    }

    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -12;
  }

  cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
  if (!kernel || err != CL_SUCCESS) {
    fprintf(stderr, "clCreateKernel failed: %s (%d)\n", cl_errstr(err), err);
    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -13;
  }

  const size_t bytes = n * sizeof(float);

  cl_mem bufA = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY,  bytes, NULL, &err);
  cl_mem bufB = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY,  bytes, NULL, &err);
  cl_mem bufO = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, bytes, NULL, &err);

  if (err != CL_SUCCESS || !bufA || !bufB || !bufO) {
    fprintf(stderr, "clCreateBuffer failed: %s (%d)\n", cl_errstr(err), err);
    if (bufA) clReleaseMemObject(bufA);
    if (bufB) clReleaseMemObject(bufB);
    if (bufO) clReleaseMemObject(bufO);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -14;
  }

  err = clEnqueueWriteBuffer(ctx.queue, bufA, CL_TRUE, 0, bytes, a, 0, NULL, NULL);
  if (err == CL_SUCCESS) err = clEnqueueWriteBuffer(ctx.queue, bufB, CL_TRUE, 0, bytes, b, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clEnqueueWriteBuffer failed: %s (%d)\n", cl_errstr(err), err);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufO);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -15;
  }

  unsigned int N = (unsigned int)n;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufO);
  err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clSetKernelArg failed: %s (%d)\n", cl_errstr(err), err);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufO);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -16;
  }

  size_t global = n;
  err = clEnqueueNDRangeKernel(ctx.queue, kernel, 1, NULL, &global, NULL, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clEnqueueNDRangeKernel failed: %s (%d)\n", cl_errstr(err), err);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufO);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -17;
  }

  clFinish(ctx.queue);

  err = clEnqueueReadBuffer(ctx.queue, bufO, CL_TRUE, 0, bytes, out, 0, NULL, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clEnqueueReadBuffer failed: %s (%d)\n", cl_errstr(err), err);
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufO);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    cl_cleanup(&ctx);
    return -18;
  }

  clReleaseMemObject(bufA);
  clReleaseMemObject(bufB);
  clReleaseMemObject(bufO);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  cl_cleanup(&ctx);

  return 0;
}

int vector_add(const float* a, const float* b, float* out, size_t n) {
  if (!a || !b || !out || n == 0) return -1;
  return vector_add_opencl(a, b, out, n);
}