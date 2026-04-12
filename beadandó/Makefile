UNAME := $(shell uname)
ifeq ($(UNAME), Darwin)
    OPENCL_FLAGS = -framework OpenCL
else
    OPENCL_FLAGS = -lOpenCL
endif

all: gauss_ocl gen

gauss_ocl:
	gcc main.c src/kernel_loader.c -o gauss_ocl -Iinclude $(OPENCL_FLAGS) -lm

gen:
	gcc matrix_generator.c -o gen

clean:
	rm -f gauss_ocl gen *.txt
