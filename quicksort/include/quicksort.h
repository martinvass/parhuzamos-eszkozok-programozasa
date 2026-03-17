#ifndef QUICKSORT_H
#define QUICKSORT_H

#include <stddef.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

void run_quicksort(cl_context context,
                   cl_command_queue queue,
                   cl_program program,
                   int* data,
                   int size,
                   int left,
                   int right);

#endif