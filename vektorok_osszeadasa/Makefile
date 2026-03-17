CC      := clang
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Iinclude
LDFLAGS := -framework OpenCL

SRC     := main.c src/vector_add.c src/cl_utils.c
OBJ     := $(SRC:.c=.o)
BIN     := vector_add

.PHONY: all clean run

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(BIN)

run: $(BIN)
	./$(BIN)