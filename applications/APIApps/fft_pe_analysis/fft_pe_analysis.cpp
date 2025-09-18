#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include "dash.h"
#include <string.h>




int main(void) {
  const size_t NUM_FFTS = 1;
  bool is_fwd = true;

  for (size_t log2_size = 4; log2_size <= 10; log2_size++) {
    size_t FFT_SIZE = std::pow(2, log2_size);

    dash_cmplx_flt_type **fft_inputs  = (dash_cmplx_flt_type**) calloc(NUM_FFTS, sizeof(dash_cmplx_flt_type*));
    dash_cmplx_flt_type **fft_outputs = (dash_cmplx_flt_type**) calloc(NUM_FFTS, sizeof(dash_cmplx_flt_type*));

    printf("Initializing input for %ld FFTs with size %ld...\n", NUM_FFTS, FFT_SIZE);

    for (size_t i = 0; i < NUM_FFTS; i++) {
      fft_inputs[i]  = (dash_cmplx_flt_type*) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
      fft_outputs[i] = (dash_cmplx_flt_type*) calloc(FFT_SIZE, sizeof(dash_cmplx_flt_type));
    }

    for (size_t i = 0; i < NUM_FFTS; i++) {
      for (size_t j = 0; j < FFT_SIZE; j++) {
        if (j == 0) {
          fft_inputs[i][j] = {.re = 1.0f, .im = 0.0f};
        } else {
          fft_inputs[i][j] = {.re = 0.0f, .im = 0.0f};
        }
      }
    }



    // Repeat the execution multiple times and take the average execution time
    const int NUM_REPEATS = 100000;
    double total_time = 0.0;
    for (int repeat = 0; repeat < NUM_REPEATS; repeat++) {
      auto start_time = std::chrono::high_resolution_clock::now();
		  pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
		  pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
		  uint32_t completion_ctr = 0;
		  uint32_t completion = NUM_FFTS;
		  cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr, .completion = &completion};

		  for(int i = 0; i < NUM_FFTS; i++){
		    DASH_FFT_flt_nb(&fft_inputs[0], &fft_outputs[0], &FFT_SIZE, &is_fwd, &barrier);
		  }
		
		  pthread_mutex_lock(barrier.mutex);
		  if (completion_ctr != NUM_FFTS) {
		    pthread_cond_wait(barrier.cond, barrier.mutex);
		  }
		  pthread_mutex_unlock(barrier.mutex);
		
      

      auto end_time = std::chrono::high_resolution_clock::now();
      auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
      total_time += elapsed_time.count();
    }
    double average_time = total_time / NUM_REPEATS;

    printf("All %ld FFTs with size %ld have been completed! Average execution time: %f microseconds\n",NUM_FFTS, FFT_SIZE, average_time);

    for (size_t i = 0; i < NUM_FFTS; i++) {
      free(fft_inputs[i]);
      free(fft_outputs[i]);
      
     }
    

		}     
   
	 return 0;
	 
}	 
