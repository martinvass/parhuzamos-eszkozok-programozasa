#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif
#include <stdio.h>
#include "quicksort.h"

void run_quicksort(cl_context context,
                   cl_command_queue queue,
                   cl_program program,
                   int* data,
                   int size,
                   int left,
                   int right)
{
    cl_int err;

    cl_kernel kernel = clCreateKernel(program, "quicksort_segment", &err);

    cl_mem buffer = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(int) * size,
                                   data,
                                   &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer);
    clSetKernelArg(kernel, 1, sizeof(int), &left);
    clSetKernelArg(kernel, 2, sizeof(int), &right);

    size_t global = 1;

    clEnqueueNDRangeKernel(queue,
                           kernel,
                           1,
                           NULL,
                           &global,
                           NULL,
                           0,
                           NULL,
                           NULL);

    clFinish(queue);

    clEnqueueReadBuffer(queue,
                        buffer,
                        CL_TRUE,
                        0,
                        sizeof(int) * size,
                        data,
                        0,
                        NULL,
                        NULL);

    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
}