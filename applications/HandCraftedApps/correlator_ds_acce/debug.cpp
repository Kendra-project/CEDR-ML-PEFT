#include "debug.h"

void PRINT_ARRAY(dash_cmplx_flt_type *array, int size, char *array_name, char* func_name, int iter) {
  printf("[%s %d] ========================================= DEBUG PRINT ===============================================\n", func_name, iter);
  printf("[%s %d] Value in %s is\n", func_name, iter, array_name);
  for (int i=0; i < size; i++){
    printf("(%f, %f i), ", array[i].re, array[i].im);
  }
  printf("\n[%s %d] =================================================================================================\n", func_name, iter);
}