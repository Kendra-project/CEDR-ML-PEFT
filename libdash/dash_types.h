#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#define CONV_2D_MIN 0
#define CONV_2D_MAX 32767

typedef int16_t dash_re_int16_type;

typedef short dash_re_int_type;

typedef struct dash_cmplx_int_type {
dash_re_int_type im;
dash_re_int_type re;
} dash_cmplx_int_type;

typedef float dash_re_flt_type;

typedef struct dash_cmplx_flt_type {
dash_re_flt_type re;
dash_re_flt_type im;
} dash_cmplx_flt_type;

typedef struct cedr_barrier {
  pthread_cond_t* cond;
  pthread_mutex_t* mutex;
  uint32_t* completion_ctr;
  uint32_t* completion;
} cedr_barrier_t;

typedef enum zip_op {
  ZIP_ADD = 0,
  ZIP_SUB = 1,
  ZIP_MULT = 2,
  ZIP_DIV = 3,
} zip_op_t;

#ifdef __cplusplus
} // Close 'extern "C"'
#endif
