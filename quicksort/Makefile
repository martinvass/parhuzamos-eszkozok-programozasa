CC      := clang
CFLAGS  := -std=c11 -O2 -Wall -Wextra -Iinclude
LDFLAGS := -framework OpenCL

SRC     := main.c src/quicksort.c
OBJ     := $(SRC:.c=.o)
BIN     := quicksort

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