#include "dash.h"
#include <stdlib.h>

int main(int argc, char *argv[]){

    size_t size = 512;
    int iter_count = 512;
    bool trans = true;

    dash_cmplx_flt_type *input_arr = (dash_cmplx_flt_type *) malloc (sizeof (dash_cmplx_flt_type) * size);
    dash_cmplx_flt_type *output_arr = (dash_cmplx_flt_type *) malloc (sizeof (dash_cmplx_flt_type) * size);

    for(int i = 0; i < size; i++){
        input_arr[i].re = i;
        input_arr[i].im = 0;
    }

    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    uint32_t completion = iter_count;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr, .completion = &completion};

    for(int iter = 0; iter < iter_count; iter++){
        DASH_FFT_flt_nb(&input_arr, &output_arr, &size, &trans, &barrier);
    }

    pthread_mutex_lock(barrier.mutex);
    if (completion_ctr != iter_count) {
        pthread_cond_wait(barrier.cond, barrier.mutex);
    }
    pthread_mutex_unlock(barrier.mutex);

}