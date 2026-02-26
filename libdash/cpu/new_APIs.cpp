#include "dash.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <pthread.h>

// Helper macros to clean up the code
#if defined(__cplusplus)
extern "C" {
#endif

// This is the link to the CEDR runtime
#if !defined(CPU_ONLY)
extern void enqueue_kernel(const char* kernel_name, const char* precision_name, unsigned int n_vargs, ...);
#endif

// --- HELPER FUNCTIONS (Internal logic) ---
// Note: These remain mostly unchanged, just used by the _cpu wrappers

#define NNZ 3277

void gen_vector(float *vector_x, size_t size) {
    for (size_t i = 0; i < size; i++) {
        vector_x[i] = rand() % 100;
    }
}

void gen_sparsematrix(float *M, size_t size_row, size_t size_col) {
    int count = 0;
    for (size_t i = 0; i < size_row; i++) {
        for (size_t j = 0; j < size_col; j++) {
            if (count < NNZ) {
                M[i * size_col + j] = rand() % 100;
                count++;
            } else {
                M[i * size_col + j] = 0;
            }
        }
    }
}

// ============================================================================
// 1. TRANSPOSE
// ============================================================================

// CPU Implementation
void DASH_TRANSPOSE_int_cpu(int **input, int **output, size_t *rows, size_t *cols) {
    size_t r = *rows;
    size_t c = *cols;
    if (r != c) { printf("[Error] Non-square.\n"); return; }

    for (size_t i = 0; i < r; i++) {
        for (size_t j = 0; j < c; j++) {
            (*output)[i * c + j] = (*input)[j * r + i];
        }
    }
}

// Non-Blocking Wrapper (The Interface to CEDR)
void DASH_TRANSPOSE_int_nb(int **input, int **output, size_t *rows, size_t *cols, cedr_barrier_t *barrier) {
#if defined(CPU_ONLY)
    DASH_TRANSPOSE_int_cpu(input, output, rows, cols);
    if (barrier) (*(barrier->completion_ctr))++;
#else
    // "int" is precision, 5 is number of args (4 data args + 1 barrier)
    enqueue_kernel("DASH_TRANSPOSE", "int", 5, input, output, rows, cols, barrier);
#endif
}

// Blocking Wrapper (Application calls this)
void DASH_TRANSPOSE_int(int *input, int *output, size_t rows, size_t cols) {
#if defined(CPU_ONLY)
    size_t r=rows, c=cols;
    int *in=input, *out=output;
    DASH_TRANSPOSE_int_cpu(&in, &out, &r, &c);
#else
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    
    pthread_mutex_lock(barrier.mutex);
    DASH_TRANSPOSE_int_nb(&input, &output, &rows, &cols, &barrier);
    while (completion_ctr != 1) { pthread_cond_wait(barrier.cond, barrier.mutex); }
    pthread_mutex_unlock(barrier.mutex);
#endif
}

// ============================================================================
// 2. DENSE MMUL
// ============================================================================

void DASH_DENSE_MMUL_flt_cpu(float **A, float **B, float **C, size_t *M, size_t *K, size_t *N) {
    // Generate data locally for simulation if needed, or assume data is passed in
    // For standard MMUL, we assume A and B are populated.
    size_t m=*M, k=*K, n=*N;
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            (*C)[i * n + j] = 0;
            for (size_t l = 0; l < k; l++) {
                (*C)[i * n + j] += (*A)[i * k + l] * (*B)[l * n + j];
            }
        }
    }
}

void DASH_DENSE_MMUL_flt_nb(float **A, float **B, float **C, size_t *M, size_t *K, size_t *N, cedr_barrier_t *barrier) {
#if defined(CPU_ONLY)
    DASH_DENSE_MMUL_flt_cpu(A, B, C, M, K, N);
    if (barrier) (*(barrier->completion_ctr))++;
#else
    enqueue_kernel("DASH_DENSE_MMUL", "flt", 7, A, B, C, M, K, N, barrier);
#endif
}

void DASH_DENSE_MMUL_flt(float *A, float *B, float *C, size_t M, size_t K, size_t N) {
#if defined(CPU_ONLY)
    DASH_DENSE_MMUL_flt_cpu(&A, &B, &C, &M, &K, &N);
#else
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);
    DASH_DENSE_MMUL_flt_nb(&A, &B, &C, &M, &K, &N, &barrier);
    while (completion_ctr != 1) { pthread_cond_wait(barrier.cond, barrier.mutex); }
    pthread_mutex_unlock(barrier.mutex);
#endif
}

// ============================================================================
// 3. SPARSE VMUL
// ============================================================================

void DASH_SPARSE_VMUL_flt_cpu(float **M, float **vx, float **vy, size_t *size) {
    size_t s = *size;
    // Note: In a real scenario, M should ideally be in CSR format here.
    // We stick to your logic of dense-as-sparse for simulation.
    // Logic: vy = M * vx
    for (size_t i = 0; i < s; i++) {
        float sum = 0.0f;
        for (size_t j = 0; j < s; j++) {
            sum += (*M)[i * s + j] * (*vx)[j];
        }
        (*vy)[i] = sum;
    }
}

void DASH_SPARSE_VMUL_flt_nb(float **M, float **vx, float **vy, size_t *size, cedr_barrier_t *barrier) {
#if defined(CPU_ONLY)
    DASH_SPARSE_VMUL_flt_cpu(M, vx, vy, size);
    if (barrier) (*(barrier->completion_ctr))++;
#else
    enqueue_kernel("DASH_SPARSE_VMUL", "flt", 5, M, vx, vy, size, barrier);
#endif
}

void DASH_SPARSE_VMUL_flt(float *matrix_a, float *vector_x, float *vector_y, size_t size) {
#if defined(CPU_ONLY)
    DASH_SPARSE_VMUL_flt_cpu(&matrix_a, &vector_x, &vector_y, &size);
#else
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);
    DASH_SPARSE_VMUL_flt_nb(&matrix_a, &vector_x, &vector_y, &size, &barrier);
    while (completion_ctr != 1) { pthread_cond_wait(barrier.cond, barrier.mutex); }
    pthread_mutex_unlock(barrier.mutex);
#endif
}

// ============================================================================
// 4. CONV 2D
// ============================================================================

void DASH_CONVOLVE_2D_flt_cpu(float **input, float **kernel, float **output, size_t *W, size_t *H, size_t *S) {
    size_t width = *W;
    size_t F = *S;
    // Simplified convolution logic
    for (size_t i = 0; i < width - F + 1; i++) {
        for (size_t j = 0; j < width - F + 1; j++) {
            float sum = 0;            
            for (size_t fi = 0; fi < F; fi++) {
                for (size_t fj = 0; fj < F; fj++) {
                    sum += (*input)[(i + fi) * width + (j + fj)] * (*kernel)[fi * F + fj];
                }
            }
            (*output)[i * (width - F + 1) + j] = sum;
        }
    }
}

void DASH_CONVOLVE_2D_flt_nb(float **input, float **kernel, float **output, size_t *W, size_t *H, size_t *S, cedr_barrier_t *barrier) {
#if defined(CPU_ONLY)
    DASH_CONVOLVE_2D_flt_cpu(input, kernel, output, W, H, S);
    if (barrier) (*(barrier->completion_ctr))++;
#else
    enqueue_kernel("DASH_CONVOLVE_2D", "flt", 7, input, kernel, output, W, H, S, barrier);
#endif
}

void DASH_CONVOLVE_2D_flt(float *input, float *kernel, float *output, size_t input_W, size_t input_H, size_t kernel_S) {
#if defined(CPU_ONLY)
    DASH_CONVOLVE_2D_flt_cpu(&input, &kernel, &output, &input_W, &input_H, &kernel_S);
#else
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);
    DASH_CONVOLVE_2D_flt_nb(&input, &kernel, &output, &input_W, &input_H, &kernel_S, &barrier);
    while (completion_ctr != 1) { pthread_cond_wait(barrier.cond, barrier.mutex); }
    pthread_mutex_unlock(barrier.mutex);
#endif
}

// ============================================================================
// 5. SOFTMAX
// ============================================================================

void DASH_SOFTMAX_flt_cpu(float **input, float **output, size_t *size) {
    size_t s = *size;
    if (s == 0) return;
    float max_val = (*input)[0];
    for (size_t i = 1; i < s; i++) {
        if ((*input)[i] > max_val) max_val = (*input)[i];
    }
    float denom = 0.0f;
    for (size_t i = 0; i < s; i++) {
        float val = expf(((*input)[i] / max_val) - 1.0f);
        (*output)[i] = val;
        denom += val;
    }
    if (fabsf(denom) > 1e-6) {
        for (size_t i = 0; i < s; i++) (*output)[i] /= denom;
    }
}

void DASH_SOFTMAX_flt_nb(float **input, float **output, size_t *size, cedr_barrier_t *barrier) {
#if defined(CPU_ONLY)
    DASH_SOFTMAX_flt_cpu(input, output, size);
    if (barrier) (*(barrier->completion_ctr))++;
#else
    enqueue_kernel("DASH_SOFTMAX", "flt", 4, input, output, size, barrier);
#endif
}

void DASH_SOFTMAX_flt(float *input, float *output, size_t size) {
#if defined(CPU_ONLY)
    DASH_SOFTMAX_flt_cpu(&input, &output, &size);
#else
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);
    DASH_SOFTMAX_flt_nb(&input, &output, &size, &barrier);
    while (completion_ctr != 1) { pthread_cond_wait(barrier.cond, barrier.mutex); }
    pthread_mutex_unlock(barrier.mutex);
#endif
}

// ============================================================================
// 6. RELU
// ============================================================================

void DASH_RELU_flt_cpu(float **input, float **output, size_t *size) {
    size_t s = *size;
    for (size_t i = 0; i < s; i++) {
        (*output)[i] = fmaxf(0.0f, (*input)[i]);
    }
}

void DASH_RELU_flt_nb(float **input, float **output, size_t *size, cedr_barrier_t *barrier) {
#if defined(CPU_ONLY)
    DASH_RELU_flt_cpu(input, output, size);
    if (barrier) (*(barrier->completion_ctr))++;
#else
    enqueue_kernel("DASH_RELU", "flt", 4, input, output, size, barrier);
#endif
}

void DASH_RELU_flt(float *input, float *output, size_t size) {
#if defined(CPU_ONLY)
    DASH_RELU_flt_cpu(&input, &output, &size);
#else
    pthread_cond_t cond = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    uint32_t completion_ctr = 0;
    cedr_barrier_t barrier = {.cond = &cond, .mutex = &mutex, .completion_ctr = &completion_ctr};
    pthread_mutex_lock(barrier.mutex);
    DASH_RELU_flt_nb(&input, &output, &size, &barrier);
    while (completion_ctr != 1) { pthread_cond_wait(barrier.cond, barrier.mutex); }
    pthread_mutex_unlock(barrier.mutex);
#endif
}

#if defined(__cplusplus)
}
#endif