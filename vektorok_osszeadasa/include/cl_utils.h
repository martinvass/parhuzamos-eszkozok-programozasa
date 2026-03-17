#ifndef CL_UTILS_H
#define CL_UTILS_H

#include <stddef.h>

#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif

typedef struct 
{
  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
} ClContext;

int cl_init(ClContext* ctx);

void cl_cleanup(ClContext* ctx);

char* cl_load_text_file(const char* path, size_t* out_len);

const char* cl_errstr(cl_int err);

char* cl_find_kernel_path(const char* relative_path);

#endif