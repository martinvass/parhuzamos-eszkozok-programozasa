#include "cl_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_build_log(cl_program program, cl_device_id device)
{
    size_t log_size = 0;

    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          0,
                          NULL,
                          &log_size);

    char* log = (char*)malloc(log_size + 1);
    if (!log) return;

    clGetProgramBuildInfo(program,
                          device,
                          CL_PROGRAM_BUILD_LOG,
                          log_size,
                          log,
                          NULL);

    log[log_size] = '\0';

    fprintf(stderr,
            "\n=== OpenCL BUILD LOG ===\n%s\n========================\n",
            log);

    free(log);
}

void cl_check(cl_int err, const char* what)
{
    if (err != CL_SUCCESS) {
        fprintf(stderr, "OpenCL error (%d): %s\n", err, what);
        exit(1);
    }
}

char* read_text_file(const char* path, size_t* out_size)
{
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "Cannot open file: %s\n", path);
        exit(1);
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* buffer = (char*)malloc(size + 1);
    if (!buffer) {
        fprintf(stderr, "Out of memory reading file: %s\n", path);
        exit(1);
    }

    size_t read_bytes = fread(buffer, 1, size, f);
    fclose(f);

    buffer[read_bytes] = '\0';

    if (out_size)
        *out_size = read_bytes;

    return buffer;
}

cl_program cl_build_program_from_file(cl_context ctx,
                                      cl_device_id dev,
                                      const char* path,
                                      const char* build_opts)
{
    size_t src_size = 0;
    char* src = read_text_file(path, &src_size);

    const char* sources[] = { src };
    cl_int err;

    cl_program program =
        clCreateProgramWithSource(ctx,
                                  1,
                                  sources,
                                  &src_size,
                                  &err);

    free(src);

    cl_check(err, "clCreateProgramWithSource");

    err = clBuildProgram(program,
                         1,
                         &dev,
                         build_opts,
                         NULL,
                         NULL);

    if (err != CL_SUCCESS) {
        print_build_log(program, dev);
        cl_check(err, "clBuildProgram");
    }

    return program;
}

cl_env_t cl_create_env(void)
{
    cl_env_t env;
    memset(&env, 0, sizeof(env));

    cl_int err;

    cl_uint num_platforms = 0;
    cl_check(clGetPlatformIDs(0, NULL, &num_platforms),
             "clGetPlatformIDs count");

    if (num_platforms == 0) {
        fprintf(stderr, "No OpenCL platform found.\n");
        exit(1);
    }

    cl_platform_id* platforms =
        (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);

    cl_check(clGetPlatformIDs(num_platforms,
                              platforms,
                              NULL),
             "clGetPlatformIDs list");

    env.platform = platforms[0];
    free(platforms);

    cl_uint num_devices = 0;

    err = clGetDeviceIDs(env.platform,
                         CL_DEVICE_TYPE_GPU,
                         1,
                         &env.device,
                         &num_devices);

    if (err != CL_SUCCESS || num_devices == 0) {
        cl_check(clGetDeviceIDs(env.platform,
                                CL_DEVICE_TYPE_CPU,
                                1,
                                &env.device,
                                &num_devices),
                 "clGetDeviceIDs CPU fallback");
    }

    env.context = clCreateContext(NULL,
                                  1,
                                  &env.device,
                                  NULL,
                                  NULL,
                                  &err);

    cl_check(err, "clCreateContext");

    env.queue = clCreateCommandQueue(env.context,
                                     env.device,
                                     0,
                                     &err);

    cl_check(err, "clCreateCommandQueue");

    return env;
}

void cl_release_env(cl_env_t* env)
{
    if (!env) return;

    if (env->queue)
        clReleaseCommandQueue(env->queue);

    if (env->context)
        clReleaseContext(env->context);

    memset(env, 0, sizeof(*env));
}