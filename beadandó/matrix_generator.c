#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Hasznalat: %s N seed\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    int seed = atoi(argv[2]);
    srand(seed);

    printf("%d\n", N);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", (double)rand() / RAND_MAX);
        }
        printf("\n");
    }
    for (int i = 0; i < N; i++) {
        printf("%f ", (double)rand() / RAND_MAX);
    }
    printf("\n");
    return 0;
}
