#include "cl_utils.h"
#include <stdio.h>
#include <stdlib.h>

#include <string.h>
#include <limits.h>
#include <unistd.h>

#ifdef __APPLE__
  #include <mach-o/dyld.h>
#endif

static void print_build_log(cl_program program, cl_device_id device) {
  size_t log_size = 0;
  clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
  if (log_size > 1) {
    char* log = (char*)malloc(log_size);
    if (!log) return;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
    fprintf(stderr, "OpenCL build log:\n%s\n", log);
    free(log);
  }
}

int cl_init(ClContext* ctx) {
  if (!ctx) return -1;

  ctx->platform = NULL;
  ctx->device   = NULL;
  ctx->context  = NULL;
  ctx->queue    = NULL;

  cl_int err = CL_SUCCESS;

  // Platform
  err = clGetPlatformIDs(1, &ctx->platform, NULL);
  if (err != CL_SUCCESS) {
    fprintf(stderr, "clGetPlatformIDs failed: %s (%d)\n", cl_errstr(err), err);
    return -2;
  }

  // Prefer GPU, fallback CPU
  err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_GPU, 1, &ctx->device, NULL);
  if (err != CL_SUCCESS) {
    err = clGetDeviceIDs(ctx->platform, CL_DEVICE_TYPE_CPU, 1, &ctx->device, NULL);
    if (err != CL_SUCCESS) {
      fprintf(stderr, "clGetDeviceIDs failed: %s (%d)\n", cl_errstr(err), err);
      return -3;
    }
  }

  // Context
  ctx->context = clCreateContext(NULL, 1, &ctx->device, NULL, NULL, &err);
  if (!ctx->context || err != CL_SUCCESS) {
    fprintf(stderr, "clCreateContext failed: %s (%d)\n", cl_errstr(err), err);
    return -4;
  }

  // Queue
#if defined(CL_VERSION_2_0)
  const cl_queue_properties props[] = { 0 };
  ctx->queue = clCreateCommandQueueWithProperties(ctx->context, ctx->device, props, &err);
#else
  ctx->queue = clCreateCommandQueue(ctx->context, ctx->device, 0, &err);
#endif
  if (!ctx->queue || err != CL_SUCCESS) {
    fprintf(stderr, "clCreateCommandQueue failed: %s (%d)\n", cl_errstr(err), err);
    cl_cleanup(ctx);
    return -5;
  }

  return 0;
}

void cl_cleanup(ClContext* ctx) {
  if (!ctx) return;
  if (ctx->queue)   clReleaseCommandQueue(ctx->queue);
  if (ctx->context) clReleaseContext(ctx->context);
  ctx->queue = NULL;
  ctx->context = NULL;
  ctx->device = NULL;
  ctx->platform = NULL;
}

char* cl_load_text_file(const char* path, size_t* out_len) {
  FILE* f = fopen(path, "rb");
  if (!f) return NULL;

  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long sz = ftell(f);
  if (sz < 0) { fclose(f); return NULL; }
  rewind(f);

  char* buf = (char*)malloc((size_t)sz + 1);
  if (!buf) { fclose(f); return NULL; }

  size_t got = fread(buf, 1, (size_t)sz, f);
  fclose(f);

  if (got != (size_t)sz) { free(buf); return NULL; }

  buf[sz] = '\0';
  if (out_len) *out_len = (size_t)sz;
  return buf;
}

const char* cl_errstr(cl_int err) {
  switch (err) {
    case CL_SUCCESS: return "CL_SUCCESS";
    case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
    case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
    case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
    case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
    case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
    case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
    case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
    case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
    case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
    case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
    case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
    case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
    case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
    case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
    case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
    case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
    case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
    case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
    case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
    case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
    case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
    case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
    case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
    case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
    case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
    case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
    case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
    case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
    case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
    case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
    case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
    case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
    case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
    case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
    case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
    case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
    case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
    default: return "CL_UNKNOWN_ERROR";
  }
}

static int file_exists(const char* path) {
  return (path && access(path, R_OK) == 0);
}

static char* dup_str(const char* s) {
  if (!s) return NULL;
  size_t n = strlen(s);
  char* r = (char*)malloc(n + 1);
  if (!r) return NULL;
  memcpy(r, s, n + 1);
  return r;
}

static char* join2(const char* a, const char* b) {
  if (!a || !b) return NULL;
  size_t na = strlen(a), nb = strlen(b);
  int need_slash = (na > 0 && a[na - 1] != '/');
  char* r = (char*)malloc(na + nb + (need_slash ? 2 : 1));
  if (!r) return NULL;
  memcpy(r, a, na);
  size_t pos = na;
  if (need_slash) r[pos++] = '/';
  memcpy(r + pos, b, nb);
  r[pos + nb] = '\0';
  return r;
}

static void dirname_inplace(char* path) {
  if (!path) return;
  size_t n = strlen(path);
  if (n == 0) return;
  // strip trailing slashes
  while (n > 1 && path[n - 1] == '/') path[--n] = '\0';
  char* last = strrchr(path, '/');
  if (!last) { path[0] = '.'; path[1] = '\0'; return; }
  if (last == path) { path[1] = '\0'; return; } // keep root "/"
  *last = '\0';
}

char* cl_find_kernel_path(const char* relative_path) {
  if (!relative_path) return NULL;

  // 1) CWD-ből
  if (file_exists(relative_path)) return dup_str(relative_path);

  // 2) ./ + relative_path
  {
    char* p = join2(".", relative_path);
    if (p && file_exists(p)) return p;
    free(p);
  }

  // 3) ../ + relative_path
  {
    char* p = join2("..", relative_path);
    if (p && file_exists(p)) return p;
    free(p);
  }

#ifdef __APPLE__
  // 4) futtatható mappájából + relative_path
  {
    uint32_t sz = 0;
    _NSGetExecutablePath(NULL, &sz);
    char* exepath = (char*)malloc(sz);
    if (exepath && _NSGetExecutablePath(exepath, &sz) == 0) {
      // canonicalizálás (valós út)
      char realbuf[PATH_MAX];
      const char* base = exepath;
      if (realpath(exepath, realbuf)) base = realbuf;

      char* tmp = dup_str(base);
      dirname_inplace(tmp); // exe dir
      char* p = join2(tmp, relative_path);
      free(tmp);

      if (p && file_exists(p)) { free(exepath); return p; }
      free(p);

      // 5) exe dir/../ + relative_path
      tmp = dup_str(base);
      dirname_inplace(tmp);
      char* up = join2(tmp, "..");
      free(tmp);
      char* p2 = join2(up, relative_path);
      free(up);
      if (p2 && file_exists(p2)) { free(exepath); return p2; }
      free(p2);
    }
    free(exepath);
  }
#endif

  return NULL;
}