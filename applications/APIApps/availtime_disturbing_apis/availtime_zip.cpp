#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include "dash.h"
#include <cstdlib>
#include <ctime>
#include <signal.h>
#define MAX_UINT32 4294967295





int main(void) {
  // Seed the random number generator with the current time
  srand(static_cast<unsigned int>(time(0)));

  size_t ZIP_SIZE = 150000000;//rand() % 100000000 + 100000000; // it is divided by 10 for ZCU102
  int ROUNDS = 27;//(rand() % 20 + 20) ; it is 25 for ZCU102
  printf("availtime zip_COUNT: %d\n",ROUNDS); 	
  printf("Starting very large size ZIPs (%ld)...\n",ZIP_SIZE);
  for(int i = 0; i < ROUNDS; i++){   
  zip_op_t ZIP_MULT;
  dash_cmplx_flt_type **zip_inputs  = (dash_cmplx_flt_type**) malloc(sizeof(dash_cmplx_flt_type*));
  dash_cmplx_flt_type **zip_outputs = (dash_cmplx_flt_type**) malloc(sizeof(dash_cmplx_flt_type*));



  zip_inputs[0]  = (dash_cmplx_flt_type*) malloc(ZIP_SIZE*sizeof(dash_cmplx_flt_type));
  zip_outputs[0] = (dash_cmplx_flt_type*) malloc(ZIP_SIZE*sizeof(dash_cmplx_flt_type));


 // for(int j = 0 ; j < 10; j++){
  //while(true){
  
  		int LOOP_COUNT = 1;

  	


		  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
		  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
		  uint32_t completion_ctr = 0;
		  uint32_t completion = LOOP_COUNT;
		  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr, .completion = &completion};

		  for(int i = 0; i < LOOP_COUNT; i++){
		  	//printf("Iter: %d\n", iter++);
		    DASH_ZIP_flt_nb(&zip_inputs[0], &zip_inputs[0], &zip_outputs[0], &ZIP_SIZE, &ZIP_MULT, &barrier);
		  }
		
		  pthread_mutex_lock(barrier.mutex);
		  if (completion_ctr != LOOP_COUNT) {
		    pthread_cond_wait(barrier.cond, barrier.mutex);
		  }
		  pthread_mutex_unlock(barrier.mutex);
		  //DASH_ZIP_flt(zip_inputs[0], zip_inputs[0],zip_outputs[0], ZIP_SIZE, ZIP_MULT);
   
//  }

  free(zip_inputs[0]);
  free(zip_outputs[0]);
  free(zip_inputs);
  free(zip_outputs);
  }
  printf("availtime zip is done ..: %d\n",ROUNDS); 	
  return 0;
}
