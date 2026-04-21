#!/bin/bash
# ============================================================
# Mérési script — Gauss-elimináció: Szekvenciális vs OpenCL
# Futtatás: chmod +x measure.sh && ./measure.sh
# ============================================================

set -e
export LC_ALL=C

echo "=== Forditas ==="
gcc matrix_generator.c -o gen
gcc main.c src/kernel_loader.c -o gauss_ocl -Iinclude -framework OpenCL -lm

if [ ! -f gauss_seq.c ]; then
cat > gauss_seq.c << 'EOF'
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "Hasznalat: %s matrix.txt [-b]\n", argv[0]); return 1; }
    char *file = argv[1];
    int bench = 0;
    for (int i = 2; i < argc; i++)
        if (strcmp(argv[i], "-b") == 0) bench = 1;

    FILE *f = fopen(file, "r");
    if (!f) { perror("fopen"); return 1; }

    int N;
    fscanf(f, "%d", &N);
    float *A = malloc(N * N * sizeof(float));
    float *b = malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            fscanf(f, "%f", &A[i * N + j]);
    for (int i = 0; i < N; i++)
        fscanf(f, "%f", &b[i]);
    fclose(f);

    struct timespec t1, t2;
    if (bench) clock_gettime(CLOCK_MONOTONIC, &t1);

    for (int k = 0; k < N; k++) {
        for (int i = k + 1; i < N; i++) {
            float factor = A[i * N + k] / A[k * N + k];
            for (int j = k; j < N; j++)
                A[i * N + j] -= factor * A[k * N + j];
            b[i] -= factor * b[k];
        }
    }

    float *x = malloc(N * sizeof(float));
    for (int i = N - 1; i >= 0; i--) {
        float sum = b[i];
        for (int j = i + 1; j < N; j++)
            sum -= A[i * N + j] * x[j];
        x[i] = sum / A[i * N + i];
    }

    if (!bench)
        for (int i = 0; i < N; i++)
            printf("%d %.6f\n", i, x[i]);

    if (bench) {
        clock_gettime(CLOCK_MONOTONIC, &t2);
        double elapsed = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec) / 1e9;
        printf("SEQ_TIME %f\n", elapsed);
    }

    free(A); free(b); free(x);
    return 0;
}
EOF
fi
gcc gauss_seq.c -o gauss_seq -lm

echo "=== Meresek indulnak ==="

SIZES="500 1000 1500 2000 2500 3000 3500 4000 4500 5000 10000"
SEED=42
RUNS=3

OUTFILE="meresek.csv"
echo "N;SEQ_avg;OCL_avg;Speedup" > $OUTFILE

for N in $SIZES; do
    echo ""
    echo "--- N = $N ---"

    ./gen $N $SEED > matrix_${N}.txt
    echo "  Matrix generalva (${N}x${N})"

    SEQ_VALS=""
    OCL_VALS=""

    for RUN in $(seq 1 $RUNS); do
        SEQ_TIME=$(./gauss_seq matrix_${N}.txt -b | grep SEQ_TIME | awk '{print $2}')
        OCL_TIME=$(./gauss_ocl matrix_${N}.txt -b | grep OCL_TIME | awk '{print $2}')
        echo "  Run $RUN: SEQ=${SEQ_TIME}s  OCL=${OCL_TIME}s"
        SEQ_VALS="$SEQ_VALS $SEQ_TIME"
        OCL_VALS="$OCL_VALS $OCL_TIME"
    done

    SEQ_AVG=$(echo $SEQ_VALS | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.4f", s/NF}')
    OCL_AVG=$(echo $OCL_VALS | awk '{s=0; for(i=1;i<=NF;i++) s+=$i; printf "%.4f", s/NF}')
    SPEEDUP=$(awk "BEGIN {printf \"%.2f\", $SEQ_AVG / $OCL_AVG}")

    echo "  Atlag: SEQ=${SEQ_AVG}s  OCL=${OCL_AVG}s  Speedup=${SPEEDUP}x"
    echo "${N};${SEQ_AVG};${OCL_AVG};${SPEEDUP}" >> $OUTFILE

    rm -f matrix_${N}.txt
done

echo ""
echo "=== Kesz! ==="
echo "Eredmenyek: $OUTFILE"
echo ""
cat $OUTFILE

echo ""
echo "=== Elmeleti illesztes (f(N) = a*N^3) ==="

# 'a' konstans kiszámítása a 10000-es mérésből
SEQ_10000=$(grep "^10000;" $OUTFILE | cut -d';' -f2)
A_CONST=$(awk "BEGIN {printf \"%.15e\", $SEQ_10000 / (10000.0^3)}")
echo "a = $A_CONST"
echo ""

FITFILE="illesztes.csv"
echo "N;SEQ_mert;SEQ_elmeleti;Elteres_mp;Elteres_pct" > $FITFILE

while IFS=';' read -r N SEQ OCL SPDUP; do
    [ "$N" = "N" ] && continue   # fejléc kihagyása
    ELMELETI=$(awk "BEGIN {printf \"%.4f\", $A_CONST * ($N ^ 3)}")
    DIFF=$(awk "BEGIN {printf \"%.4f\", $SEQ - $ELMELETI}")
    PCT=$(awk "BEGIN {if ($SEQ > 0) printf \"%.2f\", ($SEQ - $ELMELETI) / $SEQ * 100; else print \"0.00\"}")
    echo "  N=$N  Mert=${SEQ}s  Elmeleti=${ELMELETI}s  Elteres=${PCT}%"
    echo "${N};${SEQ};${ELMELETI};${DIFF};${PCT}" >> $FITFILE
done < $OUTFILE

echo ""
echo "Illesztesi eredmenyek: $FITFILE"