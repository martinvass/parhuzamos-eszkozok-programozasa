#ifdef __APPLE__
  #include <OpenCL/opencl.h>
#else
  #include <CL/cl.h>
#endif
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 16

char* readKernelSource(const char* filename)
{
    FILE* fp = fopen(filename, "r");
    if (!fp)
    {
        printf("Cannot open kernel file\n");
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    rewind(fp);

    char* source = (char*)malloc(size + 1);
    fread(source, 1, size, fp);
    source[size] = '\0';

    fclose(fp);
    return source;
}

int main()
{
    int data[ARRAY_SIZE] = {9, 4, 6, 2, 7, 1, 5, 3, 8, 11, 15, 10, 14, 13, 12, 0};

    printf("Before sorting:\n");
    for(int i=0;i<ARRAY_SIZE;i++)
        printf("%d ", data[i]);
    printf("\n");

    cl_int err;

    // platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, NULL);

    // context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);

    // queue
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);

    // kernel source
    char* source = readKernelSource("kernels/quicksort.cl");

    cl_program program = clCreateProgramWithSource(context, 1,
                                                    (const char**)&source,
                                                    NULL, &err);

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    if(err != CL_SUCCESS)
    {
        char buffer[2048];
        clGetProgramBuildInfo(program, device,
                              CL_PROGRAM_BUILD_LOG,
                              sizeof(buffer),
                              buffer,
                              NULL);

        printf("Build error:\n%s\n", buffer);
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "quicksort_segment", &err);

    // buffer
    cl_mem buffer = clCreateBuffer(context,
                                   CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                   sizeof(int) * ARRAY_SIZE,
                                   data,
                                   &err);

    int left = 0;
    int right = ARRAY_SIZE - 1;

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
                        sizeof(int) * ARRAY_SIZE,
                        data,
                        0,
                        NULL,
                        NULL);

    printf("After sorting:\n");

    for(int i=0;i<ARRAY_SIZE;i++)
        printf("%d ", data[i]);

    printf("\n");

    clReleaseMemObject(buffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(source);

    return 0;
}