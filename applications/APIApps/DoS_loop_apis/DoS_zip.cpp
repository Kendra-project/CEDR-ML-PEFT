#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "dash.h"

#define MAX_UINT32 4294967295
#define LOOP_COUNT 128


int main(void) {
  size_t ZIP_SIZE = 256; 
  zip_op_t ZIP_MULT;
  dash_cmplx_flt_type **zip_inputs  = (dash_cmplx_flt_type**) malloc(sizeof(dash_cmplx_flt_type*));
  dash_cmplx_flt_type **zip_outputs = (dash_cmplx_flt_type**) malloc(sizeof(dash_cmplx_flt_type*));

  printf("Starting DoS loop of ZIPs...\n");

  zip_inputs[0]  = (dash_cmplx_flt_type*) malloc(ZIP_SIZE*sizeof(dash_cmplx_flt_type));
  zip_outputs[0] = (dash_cmplx_flt_type*) malloc(ZIP_SIZE*sizeof(dash_cmplx_flt_type));

  // Seed the random number generator with the current time
  //srand(time(0));

  // Generate a random integer between 1000000 and 300000
  //int DoS_COUNT = rand() % 200000 + 100000;
  int DoS_COUNT = 100000;
  printf("ZIP DoS_count : %d\n",DoS_COUNT* LOOP_COUNT);
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
      DASH_ZIP_flt_nb(&zip_inputs[0], &zip_inputs[0], &zip_outputs[0], &ZIP_SIZE, &ZIP_MULT, &barrier);
    }
  
    pthread_mutex_lock(barrier.mutex);
    if (completion_ctr != LOOP_COUNT) {
      pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);
  }

  free(zip_inputs[0]);
  free(zip_outputs[0]);
  free(zip_inputs);
  free(zip_outputs);
  printf("ZIP DoS is complete .. \n");

  return 0;
}
