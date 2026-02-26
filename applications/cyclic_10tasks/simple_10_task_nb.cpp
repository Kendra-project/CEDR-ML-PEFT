#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>
#include <stddef.h>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include <cstring> 
#include <pthread.h> // Required for barriers
#include "dash.h"

// --- BUFFER SIZES ---
#define M_SIZE 128
#define V_SIZE (M_SIZE * M_SIZE)

// --- GLOBAL BUFFERS ---
// Buffers to hold data for edges (T_Source -> T_Dest)
float T1_out[V_SIZE];
float T2_out[V_SIZE];
float T3_out[V_SIZE];
float T4_out[V_SIZE];
dash_cmplx_flt_type T5_out_complex[V_SIZE]; // FFT output
float T5_out_real[V_SIZE];
float T6_out[V_SIZE];
dash_cmplx_flt_type T7_out_complex[V_SIZE]; // ZIP output
float T8_out[V_SIZE];
int T9_out_int[V_SIZE]; // Transpose output
float T10_out[V_SIZE];

// Inputs
float T1_in[V_SIZE];
float Aux_Kern[9];
float Aux_Vec[V_SIZE];
dash_cmplx_flt_type Aux_Cmplx[V_SIZE];

// --- HELPER FUNCTIONS FOR BARRIERS ---
cedr_barrier_t create_barrier() {
    cedr_barrier_t b;
    b.mutex = new pthread_mutex_t;
    b.cond = new pthread_cond_t;
    b.completion_ctr = new uint32_t(0);
    pthread_mutex_init(b.mutex, NULL);
    pthread_cond_init(b.cond, NULL);
    return b;
}

void wait_for_barrier(cedr_barrier_t b) {
    pthread_mutex_lock(b.mutex);
    while (*(b.completion_ctr) == 0) {
        pthread_cond_wait(b.cond, b.mutex);
    }
    pthread_mutex_unlock(b.mutex);
}

void run_custom_dag() {
    printf("--- Generating DAG from Figure (Weighted Mapping) ---\n");

    // Initialize Barriers
    cedr_barrier_t b_T1 = create_barrier();
    cedr_barrier_t b_T2 = create_barrier();
    cedr_barrier_t b_T3 = create_barrier();
    cedr_barrier_t b_T4 = create_barrier();
    cedr_barrier_t b_T5 = create_barrier();
    cedr_barrier_t b_T6 = create_barrier();
    cedr_barrier_t b_T7 = create_barrier();
    cedr_barrier_t b_T8 = create_barrier();
    cedr_barrier_t b_T9 = create_barrier();
    cedr_barrier_t b_T10 = create_barrier();

    size_t size_m = M_SIZE;
    size_t size_v = V_SIZE;
    size_t size_k3 = 3; 
    bool fwd = true;
    zip_op_t op_add = ZIP_ADD;

    // =========================================================================
    // LEVEL 1: T1 (Entry)
    // =========================================================================
    // T1: DASH_CONVOLVE_2D (Weight: 26)
    printf("[Host] Submitting T1 (CONV)...\n");
    float *p_T1_in = T1_in; 
    float *p_T1_kern = Aux_Kern;
    float *p_T1_out = T1_out;
    DASH_CONVOLVE_2D_flt_nb(&p_T1_in, &p_T1_kern, &p_T1_out, &size_m, &size_m, &size_k3, &b_T1);

    wait_for_barrier(b_T1);

    // =========================================================================
    // LEVEL 2: T2, T3, T4, T5, T6 (Run in Parallel)
    // =========================================================================
    printf("[Host] Submitting T2-T6 Batch...\n");

    // T2: DASH_SOFTMAX (Weight: 19)
    // Input: T1_out
    float *p_T2_in = T1_out;
    float *p_T2_out = T2_out;
    DASH_SOFTMAX_flt_nb(&p_T2_in, &p_T2_out, &size_v, &b_T2);

    // T3: DASH_DENSE_MMUL (Weight: 34 - HEAVIEST)
    // Input: T1_out
    float *p_T3_in = T1_out; 
    float *p_T3_out = T3_out;
    DASH_DENSE_MMUL_flt_nb(&p_T3_in, &p_T3_in, &p_T3_out, &size_m, &size_m, &size_m, &b_T3);

    // T4: DASH_SPARSE_VMUL (Weight: 7 - LIGHTEST)
    // Input: T1_out
    float *p_T4_in = T1_out;
    float *p_T4_vec = Aux_Vec;
    float *p_T4_out = T4_out;
    DASH_SPARSE_VMUL_flt_nb(&p_T4_in, &p_T4_vec, &p_T4_out, &size_m, &b_T4);

    // T5: DASH_FFT (Weight: 30 - HEAVY)
    // Input: T1_out (Needs cast to complex in real app, simulating here)
    dash_cmplx_flt_type *p_T5_in = (dash_cmplx_flt_type*)T1_out;
    dash_cmplx_flt_type *p_T5_out = T5_out_complex;
    DASH_FFT_flt_nb(&p_T5_in, &p_T5_out, &size_m, &fwd, &b_T5);

    // T6: DASH_RELU (Weight: 22)
    // Input: T1_out
    float *p_T6_in = T1_out;
    float *p_T6_out = T6_out;
    DASH_RELU_flt_nb(&p_T6_in, &p_T6_out, &size_v, &b_T6);

    // =========================================================================
    // LEVEL 3: T7, T8, T9
    // =========================================================================
    
    // T7: DASH_ZIP (Weight: 23)
    // Depends on T2 and T3
    wait_for_barrier(b_T2);
    wait_for_barrier(b_T3);
    printf("[Host] Submitting T7 (ZIP)...\n");
    dash_cmplx_flt_type *p_T7_in1 = (dash_cmplx_flt_type*)T2_out;
    dash_cmplx_flt_type *p_T7_in2 = (dash_cmplx_flt_type*)T3_out;
    dash_cmplx_flt_type *p_T7_out = T7_out_complex;
    DASH_ZIP_flt_nb(&p_T7_in1, &p_T7_in2, &p_T7_out, &size_v, &op_add, &b_T7);

    // T8: DASH_CONVOLVE_2D (Weight: 29 - HEAVY)
    // Depends on T2, T4, T6
    wait_for_barrier(b_T4);
    wait_for_barrier(b_T6);
    // (T2 already waited for above)
    printf("[Host] Submitting T8 (CONV)...\n");
    float *p_T8_in = T2_out; // Using T2 output as primary input source
    float *p_T8_kern = Aux_Kern;
    float *p_T8_out = T8_out;
    DASH_CONVOLVE_2D_flt_nb(&p_T8_in, &p_T8_kern, &p_T8_out, &size_m, &size_m, &size_k3, &b_T8);

    // T9: DASH_TRANSPOSE (Weight: 14 - LIGHT)
    // Depends on T4, T5
    wait_for_barrier(b_T5);
    // (T4 already waited for)
    printf("[Host] Submitting T9 (Transpose)...\n");
    int *p_T9_in = (int*)T5_out_complex; // Using T5 output
    int *p_T9_out = T9_out_int;
    DASH_TRANSPOSE_int_nb(&p_T9_in, &p_T9_out, &size_m, &size_m, &b_T9);

    // =========================================================================
    // LEVEL 4: T10 (Exit)
    // =========================================================================
    
    // T10: DASH_RELU (Weight: 20 - MEDIUM)
    // Depends on T7, T8, T9
    wait_for_barrier(b_T7);
    wait_for_barrier(b_T8);
    wait_for_barrier(b_T9);
    
    printf("[Host] Submitting T10 (RELU)...\n");
    float *p_T10_in = T8_out; // Using T8 output as main
    float *p_T10_out = T10_out;
    DASH_RELU_flt_nb(&p_T10_in, &p_T10_out, &size_v, &b_T10);

    wait_for_barrier(b_T10);
    printf("--- Execution Complete ---\n");
}

int main() {
    srand(42); 
    run_custom_dag();
    return 0;
}