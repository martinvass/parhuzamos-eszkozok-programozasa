#include "vector_add.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static void vector_add_cpu(const float* a, const float* b, float* out, size_t n) {
  for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
}

static float frand01(void) {
  return (float)rand() / (float)RAND_MAX; // [0,1]
}

int main(int argc, char** argv) {
  size_t n = 1 << 20; // default: 1,048,576 elem
  if (argc >= 2) {
    long v = atol(argv[1]);
    if (v > 0) n = (size_t)v;
  }

  srand((unsigned)time(NULL));

  float* a = (float*)malloc(n * sizeof(float));
  float* b = (float*)malloc(n * sizeof(float));
  float* out = (float*)malloc(n * sizeof(float));
  float* ref = (float*)malloc(n * sizeof(float));

  if (!a || !b || !out || !ref) {
    fprintf(stderr, "malloc failed\n");
    free(a); free(b); free(out); free(ref);
    return 1;
  }

  for (size_t i = 0; i < n; ++i) {
    a[i] = 100.0f * (frand01() - 0.5f);
    b[i] = 100.0f * (frand01() - 0.5f);
  }

  int rc = vector_add(a, b, out, n);
  if (rc != 0) {
    fprintf(stderr, "vector_add failed: %d\n", rc);
    free(a); free(b); free(out); free(ref);
    return 2;
  }

  vector_add_cpu(a, b, ref, n);

  // Ellenőrzés toleranciával (float)
  float max_abs_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float e = fabsf(out[i] - ref[i]);
    if (e > max_abs_err) max_abs_err = e;
  }

  const float eps = 1e-6f;
  printf("n=%zu, max_abs_err=%.9g, %s\n", n, max_abs_err, (max_abs_err <= eps ? "OK" : "MISMATCH"));

  free(a); free(b); free(out); free(ref);
  return (max_abs_err <= eps) ? 0 : 3;
}