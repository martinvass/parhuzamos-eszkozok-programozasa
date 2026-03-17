#pragma once

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <stddef.h>

typedef struct {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
} cl_env_t;

void cl_check(cl_int err, const char* what);

cl_env_t cl_create_env(void);

void cl_release_env(cl_env_t* env);

char* read_text_file(const char* path, size_t* out_size);

cl_program cl_build_program_from_file(cl_context ctx,
                                      cl_device_id dev,
                                      const char* path,
                                      const char* build_opts);