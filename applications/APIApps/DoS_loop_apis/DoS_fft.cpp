#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "dash.h"
#include <ctime>     // For time()


#define MAX_UINT32 4294967295
#define LOOP_COUNT 512


int main(void) {
  size_t FFT_SIZE = 512;
  bool   is_fwd   = true;
  dash_cmplx_flt_type **fft_inputs  = (dash_cmplx_flt_type**) malloc(sizeof(dash_cmplx_flt_type*));
  dash_cmplx_flt_type **fft_outputs = (dash_cmplx_flt_type**) malloc(sizeof(dash_cmplx_flt_type*));

  printf("Starting DoS loop of FFTs...\n");

  fft_inputs[0]  = (dash_cmplx_flt_type*) malloc(FFT_SIZE*sizeof(dash_cmplx_flt_type));
  fft_outputs[0] = (dash_cmplx_flt_type*) malloc(FFT_SIZE*sizeof(dash_cmplx_flt_type));

  /*
  for (size_t i = 0; i < NUM_FFTS; i++) {
    for (size_t j = 0; j < FFT_SIZE; j++) {
      if (j == 0) {
        fft_inputs[i][j] = {.re = 1.0f, .im = 0.0f};
      } else {
        fft_inputs[i][j] = {.re = 0.0f, .im = 0.0f};
      }
      //printf("The address of element (%ld, %ld) is: %p\n", i, j, &fft_inputs[i][j]);
    }
  }
  //printf("The address of the 7th row in this array is: %p\n", &(fft_inputs[7]));
  */

  // Seed the random number generator with the current time
  //srand(time(0));

  // Generate a random integer between 1000000 and 300000
  //int DoS_COUNT = rand() % 100000 + 100000;
  int DoS_COUNT = 15000;
  printf("FFT DoS_COUNT: %d\n",DoS_COUNT*LOOP_COUNT);
  int iter=0;
  for(int j = 0 ; j < DoS_COUNT; j++){
  //while(true){
    //printf("Iter: %d\n", iter++);

    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    uint32_t completion = LOOP_COUNT;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr, .completion = &completion};

    for(int i = 0; i < LOOP_COUNT; i++){
      DASH_FFT_flt_nb(&fft_inputs[0], &fft_outputs[0], &FFT_SIZE, &is_fwd, &barrier);
    }
  
    pthread_mutex_lock(barrier.mutex);
    if (completion_ctr != LOOP_COUNT) {
      pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);
  }
/*
  printf("All %ld FFTs have been completed! Printing results...\n", NUM_FFTS);

  for (size_t i = 0; i < NUM_FFTS; i++) {
    printf("FFT %ld: ", i);
    for (size_t j = 0; j < FFT_SIZE; j++) {
      printf("(%f, %f)", fft_outputs[i][j].re, fft_outputs[i][j].im);
      if (j != FFT_SIZE - 1) {
        printf(", ");
      }
    }
    printf("\n");
  }
*/

  printf("FFT DoS is complete .. \n");
  free(fft_inputs[0]);
  free(fft_outputs[0]);
  free(fft_inputs);
  free(fft_outputs);
  

  return 0;
}
