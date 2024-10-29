#include <R.h>
#include <Rdefines.h>
#include <Rinternals.h>
#include <Rmath.h>
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

#define M_PI 3.14159265358979323846
#define USE_FC_LEN_T

#ifndef FCONE
#define FCONE
#endif

// #define DEBUG_PRINT
// #define DEBUGME

void print_array(double *data, int i, int j, const char *lab);

void print_int_array(int *data, int i, int j, const char *lab);

void locateNA(double *vec, int *NAindices, int *positions, int len);

int numberofNA(double *vec, int *NAindices, int *positions, int len);

void reduce_array(double *array_full, int dim0, int dim1,
                  double *array_reduced, int *pos, int len);
