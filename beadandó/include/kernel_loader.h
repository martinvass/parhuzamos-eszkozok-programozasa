#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#ifdef __APPLE__
    #include <OpenCL/opencl.h>
#else
    #define CL_TARGET_OPENCL_VERSION 220
    #include <CL/cl.h>
#endif

char* load_kernel_source(const char* const path, int* error_code);

#endif
