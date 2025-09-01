#include <string.h>
#include <gsl/gsl_fft_complex_float.h>
#include <gsl/gsl_fft_complex.h>
#include "dash.h"

void gsl_fft_wrapper(dash_cmplx_flt_type* input, dash_cmplx_flt_type* output, size_t size, bool isForwardTransform) {
//  printf("[fft] Running a %lu-Pt %s on the CPU\n", (*size), *isForwardTransform ? "FFT" : "IFFT");
  
  // Note: we copy to a new buffer here because the API doesn't state that we modify the input
  dash_cmplx_flt_type *data = (dash_cmplx_flt_type*) malloc((size) * sizeof(dash_cmplx_flt_type));
  memcpy(data, input, size * sizeof(dash_cmplx_flt_type));
  
  int check;
  // Our base floating point type is 4-byte (float)
  if (sizeof(dash_re_flt_type) == 4) {
    if (isForwardTransform) {
      check = gsl_fft_complex_float_radix2_forward((float*)(data), 1, size);
    } else {
      check = gsl_fft_complex_float_radix2_inverse((float*)(data), 1, size);
    }
  } 
  // Otherwise it's 8-byte (double)
  else {
    if (isForwardTransform) {
      check = gsl_fft_complex_radix2_forward((double*)(data), 1, size);
    } else {
      check = gsl_fft_complex_radix2_inverse((double*)(data), 1, size);
    }
  }

  if (check != 0) {
    fprintf(stderr, "[libdash] Failed to complete DASH_FFT_flt_cpu using libgsl with message %d!\n", check);
    free(data);
    return;
  }

  memcpy(output, data, size * sizeof(dash_cmplx_flt_type));
  free(data);
  return;
}
