#include "../libdash/platform.h"
#include "../libdash/dma/dma.h"
#include <plog/Log.h>

#include "string.h"
#include <stdio.h>
#define NUM_FEATURES 42

static volatile unsigned int* dma_control_base_addr;
static volatile unsigned int* udmabuf_base_addr;
static uint64_t udmabuf_phys_addr;

void __attribute__((constructor))  setup_ae(void) {
    /* Set up AXI DMA */
    printf(" setup_ae(): Calling init_dma()\n");
    #ifndef AUTOENCODER_DMA_CTRL_BASE_ADDRS
         #error "Autoencoder DMA based address defined as zero. It doesn't look like it's present."
    #endif
    dma_control_base_addr = init_dma(AUTOENCODER_DMA_CTRL_BASE_ADDRS[0]);
    LOG_DEBUG << "setup_ae(): Calling reset_dma()";
    reset_dma(dma_control_base_addr);

    /* Set up U-DMA buffer */
    printf("setup_ae(): Calling init_udmabuf()\n");
    init_udmabuf(AUTOENCODER_UDMABUF_NUM, AUTOENCODER_UDMABUF_SIZE, &udmabuf_base_addr, &udmabuf_phys_addr);
    LOG_DEBUG << "setup_ae(): Didn't crash yet.";
}

void __attribute__((destructor))  teardown_ae(void) {
    close_udmabuf(udmabuf_base_addr, AUTOENCODER_UDMABUF_SIZE);
    close_dma(dma_control_base_addr);
}




void autoencoder_kernel(float* inputs, float* outputs, size_t samples_num) {
    // Determine the physical address offset for UDMA buffer
    volatile unsigned int *dma_control_base = dma_control_base_addr;
    volatile unsigned int *udmabuf_base = udmabuf_base_addr;
    uint64_t udmabuf_phys = udmabuf_phys_addr;
    // Copy input data to UDMA buffer
    for (size_t i = 0; i < (samples_num); i++) {
      printf("sample input %d ",i);
      for (size_t j =0; j < NUM_FEATURES; j++){

        printf("%f, ",inputs[i*NUM_FEATURES + j]);
      }
      printf("\n");
    }  
    memcpy((unsigned int*)(udmabuf_base), inputs, NUM_FEATURES * samples_num * sizeof(float));
 
    setup_rx(dma_control_base, udmabuf_phys + (NUM_FEATURES * samples_num * sizeof(float)), NUM_FEATURES * samples_num * sizeof(float));


    setup_tx(dma_control_base, udmabuf_phys, NUM_FEATURES * samples_num * sizeof(float));

    // Perform autoencoder computation here
    for (size_t i = 0; i < (samples_num); i++) {
      printf("sample output before memcpy and dma wait %d ",i);
      for (size_t j =0; j < NUM_FEATURES; j++){

        printf("%f, ",outputs[i*NUM_FEATURES + j]);
      }
      printf("\n");
    }
    // Wait for DMA to complete before reading output
    dma_wait_for_rx_complete(dma_control_base_addr);
     // Perform autoencoder computation here
    for (size_t i = 0; i < (samples_num); i++) {
      printf("sample output before memcpy and after dma wait %d ",i);
      for (size_t j =0; j < NUM_FEATURES; j++){

        printf("%f, ",outputs[i*NUM_FEATURES + j]);
      }
      printf("\n");
    }
    // Copy output data from UDMA buffer
    memcpy(outputs, (unsigned int*) &udmabuf_base[NUM_FEATURES * samples_num ], NUM_FEATURES * samples_num * sizeof(float));
    for (size_t i = 0; i < (samples_num); i++) {
      printf("sample output after memcpy %d ",i);
      for (size_t j =0; j < NUM_FEATURES; j++){

        printf("%f, ",outputs[i*NUM_FEATURES + j]);
      }
      printf("\n");
    } 
}

