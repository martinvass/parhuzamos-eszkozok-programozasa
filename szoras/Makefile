CC      = clang
CFLAGS  = -std=c11 -O2 -Wall -Wextra -Iinclude -DCL_SILENCE_DEPRECATION
LDFLAGS = -framework OpenCL

TARGET  = szoras
OBJS    = main.o src/cl_utils.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

main.o: main.c include/cl_utils.h
	$(CC) $(CFLAGS) -c main.c -o main.o

src/cl_utils.o: src/cl_utils.c include/cl_utils.h
	$(CC) $(CFLAGS) -c src/cl_utils.c -o src/cl_utils.o

clean:
	rm -f $(OBJS) $(TARGET)

.PHONY: all clean