#include <complex.h>
#include <unistd.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include "dma.h"
#include "conv_2d.h"

#include <string.h>
#include <time.h>

#if DASH_PLATFORM != DASH_ZCU102_2FFT_2MMULT_1ZIP_1CONV2D_HWSCHEDULER
  #error "CONV 2D is not supported on this platform!"
#else

static volatile unsigned int* conv_2d_control_base_addr[NUM_CONV_2DS];
static volatile unsigned int* dma_control_base_addr[NUM_CONV_2DS];
static volatile pix_t*        udmabuf_base_addr;
static uint64_t               udmabuf_phys_addr;

void __attribute__((constructor)) setup_conv_2d(void) {
  LOG("[conv_2d]\n");
  LOG("[conv_2d] Running Conv2D constructor\n");

  for (unsigned int i = 0; i < NUM_CONV_2DS; i++) {
    LOG("[conv_2d] Initializing CONV_2D DMA at 0x%x\n", CONV_2D_DMA_CTRL_BASE_ADDRS[i]);
    dma_control_base_addr[i] = init_dma(CONV_2D_DMA_CTRL_BASE_ADDRS[i]);
    reset_dma(dma_control_base_addr[i]);

    LOG("[conv_2d] Initializing CONV_2D Control at 0x%x\n", CONV_2D_CONTROL_BASE_ADDRS[i]);
    conv_2d_control_base_addr[i] = init_conv_2d(CONV_2D_CONTROL_BASE_ADDRS[i]);
  }

  LOG("[conv_2d] Initializing udmabuf\n");
  init_udmabuf(CONV_2D_UDMABUF_NUM, CONV_2D_UDMABUF_SIZE, (volatile unsigned int**)&udmabuf_base_addr, &udmabuf_phys_addr);

  LOG("[conv_2d] CONV_2D constructor complete!\n");
}

void __attribute__((destructor)) teardown_conv_2d(void) {
  LOG("[conv_2d] Running Conv2D destructor\n");
  close_udmabuf((volatile unsigned int*)udmabuf_base_addr, CONV_2D_UDMABUF_SIZE);

  for (unsigned int i = 0; i < NUM_CONV_2DS; i++) {
    close_conv_2d(conv_2d_control_base_addr[i]);
    close_dma(dma_control_base_addr[i]);
  }

  LOG("[conv_2d] Conv2D destructor complete!\n");
}

void conv_2d_accel(const pix_t* src, pix_t* dst, const float* mask, float factor, short bias, unsigned int mask_width, unsigned short width, unsigned short height, unsigned short stride, uint8_t rsrc_idx) {
  if (mask_width%2==0){
    printf("[ERROR] Using even width mask terminating the CONV accelerator execution!\n");
    return;
  }
  const size_t input_size  = width * height * sizeof(pix_t);
  const size_t output_size = width * height * sizeof(pix_t);
  const size_t padding = (input_size % BUS_SIZE) ? (BUS_SIZE - (input_size % BUS_SIZE)) : 0;
  float mask_arr[CONV_2D_MAX_MASK * CONV_2D_MAX_MASK];

  // Format mask for the accelerator
  memset(&mask_arr[0], 0, sizeof(float) * CONV_2D_MAX_MASK * CONV_2D_MAX_MASK);
  const unsigned int zero_fill = (int) ((CONV_2D_MAX_MASK - mask_width) / 2);
  unsigned int mask_index = 0;
  // Pad top and bottom
  for (int i = 0; i < zero_fill; i++) {
    for (int j = 0; j < CONV_2D_MAX_MASK; j++) {
      mask_arr[i * CONV_2D_MAX_MASK + j] = 0;
      mask_arr[(CONV_2D_MAX_MASK - 1 - i) * CONV_2D_MAX_MASK + j] = 0;
    }
  }
  for (int i = zero_fill; i < CONV_2D_MAX_MASK - zero_fill; i++) {
    // Pad sides
    for (int j = 0; j < zero_fill; j++) {
      mask_arr[i * CONV_2D_MAX_MASK + j] = 0;
      mask_arr[i * CONV_2D_MAX_MASK + (CONV_2D_MAX_MASK - 1 - j)] = 0;
    }
    for (int j = zero_fill; j < CONV_2D_MAX_MASK - zero_fill; j++) {
      mask_arr[i * CONV_2D_MAX_MASK + j] = mask[mask_index++];
    }
  }

  // Set up AXI-LITE registers
  LOG("[conv_2d-%u] Setting up AXI-LITE register 1.\n", rsrc_idx);
  conv_2d_write_fl(conv_2d_control_base_addr[rsrc_idx], CONV_2D_FACTOR, factor);
  LOG("[conv_2d-%u] Setting up AXI-LITE register 2.\n", rsrc_idx);
  conv_2d_write_16(conv_2d_control_base_addr[rsrc_idx], CONV_2D_BIAS,   bias);
  LOG("[conv_2d-%u] Setting up AXI-LITE register 3.\n", rsrc_idx);
  conv_2d_write_16(conv_2d_control_base_addr[rsrc_idx], CONV_2D_WIDTH,  width);
  LOG("[conv_2d-%u] Setting up AXI-LITE register 4.\n", rsrc_idx);
  conv_2d_write_16(conv_2d_control_base_addr[rsrc_idx], CONV_2D_HEIGHT, height);
  LOG("[conv_2d-%u] Setting up AXI-LITE register 5.\n", rsrc_idx);
  conv_2d_write_16(conv_2d_control_base_addr[rsrc_idx], CONV_2D_STRIDE, stride);

  // Fill in mask memory
  LOG("[conv_2d-%u] Copying mask.\n", rsrc_idx);
  LOG("[conv_2d-%u] First memcpy.\n", rsrc_idx);
  memcpy((void*)((char*)conv_2d_control_base_addr[rsrc_idx] + CONV_2D_MASK), &mask_arr[0], sizeof(float) * (MASK_AREA - 1));
  LOG("[conv_2d-%u] Writing last float.\n", rsrc_idx);
  conv_2d_write_fl(conv_2d_control_base_addr[rsrc_idx], (CONV_2D_MASK + ((MASK_AREA - 1) * sizeof(float))), mask_arr[MASK_AREA - 1]);

  // Set up MM2S
  LOG("[conv_2d-%u] Setting up MM2S.\n", rsrc_idx);
  if (input_size % BUS_SIZE) {
    const size_t remain = input_size % BUS_SIZE;
    memcpy((pix_t*) &udmabuf_base_addr[0], src, input_size - remain);
    for (unsigned int i = 0; i < (remain / sizeof(pix_t)); i++) {
      udmabuf_base_addr[((input_size - remain) / sizeof(pix_t)) + i] = src[((input_size - remain) / sizeof(pix_t)) + i];
    }
  } else {
    memcpy((pix_t*) &udmabuf_base_addr[0], src, input_size);
  }

  setup_rx(dma_control_base_addr[rsrc_idx], udmabuf_phys_addr + input_size + padding, output_size);

  // Initiate stream
  LOG("[conv_2d-%u] Starting accelerator.\n", rsrc_idx);
  setup_tx(dma_control_base_addr[rsrc_idx], udmabuf_phys_addr, input_size);
  conv_2d_write_32(conv_2d_control_base_addr[rsrc_idx], CONV_2D_AP_CTRL, CONV_2D_AP_START);
  LOG("[conv_2d-%u] Just wrote to control register.\n", rsrc_idx);

  // Receive S2MM
  LOG("[conv_2d-%u] Waiting for RX.\n", rsrc_idx);
  dma_wait_for_tx_complete(dma_control_base_addr[rsrc_idx]);
  dma_wait_for_rx_complete(dma_control_base_addr[rsrc_idx]);
  if (output_size % BUS_SIZE) {
    const size_t remain = output_size % BUS_SIZE;
    memcpy(dst, (pix_t*) &udmabuf_base_addr[width * height + (padding / sizeof(pix_t))], output_size - remain);
    for (unsigned int i = 0; i < remain / sizeof(pix_t); i++) {
      dst[width * height - (remain / sizeof(pix_t)) + i] = udmabuf_base_addr[2 * width * height + ((padding - remain) / sizeof(pix_t)) + i];
    }
  } else {
    memcpy(dst, (pix_t*) &udmabuf_base_addr[width * height + (padding / sizeof(pix_t))], output_size);
  }
}

extern "C" void DASH_CONV_2D_int_conv_2d(dash_re_int16_type **input, int *height, int *width, dash_re_flt_type **mask, int *mask_size, dash_re_int16_type **output, uint8_t resource_idx) {
  const float factor = 1; // TODO
  const short bias = 0;   // TODO
  const short stride = 0; // TODO: Not sure about this one, but it's supposed to be a multiple of 64

  conv_2d_accel(*input, *output, *mask, factor, bias, *mask_size, *width, *height, stride, resource_idx);

}

#if defined(__CONV_2D_ENABLE_MAIN)
void CONV_2D_ref(pix_t **input, const unsigned int *height, const unsigned int *width, float **mask, const unsigned int *mask_size, pix_t **output) {
  int i, j, k, l;
  int s, w;
  int z;

  float sum;

  z = (*mask_size) / 2;

  for (i = 0; i < (*height); i++) {
    for (j = 0; j < (*width); j++) {
      sum = 0.0;
      for (k = 0; k < (*mask_size); k++) {
        for (l = 0; l < (*mask_size); l++) {
          s = i + k - z;
          w = j + l - z;
          if ((s >= 0 && s < (*height)) && (w >= 0 && w < (*width))) {
            sum += (*input)[(*width) * s + w] * (*mask)[(*mask_size) * k + l];
          }
        }
      }
      (*output)[i * (*width) + j] = (sum > CONV_2D_MAX ? CONV_2D_MAX : (sum < CONV_2D_MIN ? CONV_2D_MIN : sum));
    }
  }
}

int main(int argc, char** argv) {
  const float factor = 1;
  const short bias = 0;
  const short stride = 64;
  const unsigned int height = 960;
  const unsigned int width = 540;
  const unsigned int mask_size = 11;

  bool failed = false;
  unsigned int fail_cntr = 0;
  unsigned int zero_cntr = 0;
  unsigned int max_cntr = 0;
  unsigned int cntr = 0;
  unsigned int max_diff = 0;

  struct timespec curr_timespec {};
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  long long start_time, end_time;

  // Allocate memory
  LOG("[conv_2d] Allocating memory.\n");
  pix_t *input = (pix_t*)malloc(height * width * sizeof(pix_t));
  pix_t *output_acc = (pix_t*)malloc(height * width * sizeof(pix_t));
  pix_t *output_ref = (pix_t*)malloc(height * width * sizeof(pix_t));
  float *mask = (float*)malloc(mask_size * mask_size * sizeof(float));

  // Fill with test data
  LOG("[conv_2d] Filling with test data.\n");
  for (unsigned int i = 0; i < height; i++) {
    for (unsigned int j = 0; j < width; j++) {
      if ((i >= 8) && (i <= 16) && (j >= 8) && (j <= 16))
        input[i * width + j] = 5;
      else
        input[i * width + j] = 1;
    }
  }
  for (unsigned int i = 0; i < mask_size; i++) {
    for (unsigned int j = 0; j < mask_size; j++) {
      if ((j == 4) && (i >= 4) && (i <= 6))
        mask[i * mask_size + j] = 1;
      else if ((j == 6) && (i >= 4) && (i <= 6))
        mask[i * mask_size + j] = 1;
      else
        mask[i * mask_size + j] = 1;
    }
  }

  // Do accelerator convolution
  LOG("[conv_2d] Calling convolution accelerator.\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  start_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
  conv_2d_accel(input, output_acc, mask, factor, bias, mask_size, width, height, stride, 0);
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  end_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
  LOG("[conv_2d] Accelerator took %lld ns.\n", (end_time - start_time));

  // Do CPU convolution
  LOG("[conv_2d] Calling convolution reference.\n");
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  start_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
  CONV_2D_ref(&input, &height, &width, &mask, &mask_size, &output_ref);
  clock_gettime(CLOCK_MONOTONIC_RAW, &curr_timespec);
  end_time = curr_timespec.tv_nsec + curr_timespec.tv_sec * SEC2NANOSEC;
  LOG("[conv_2d] CPU took %lld ns.\n", (end_time - start_time));

  // Check outputs
  for (unsigned int i = 0; i < height * width; i++) {
    if (width * height < 1024) {
      LOG("[conv_2d] index: %d, accelerator: %d, reference: %d\n", i, output_acc[i], output_ref[i]);
    }
    if (output_acc[i] != output_ref[i]) {
      failed = true;
      fail_cntr++;
      if (abs(output_acc[i] - output_ref[i]) > max_diff)
        max_diff = abs(output_acc[i] - output_ref[i]);
    } else if (output_ref[i] == 0) {
      zero_cntr++;
    } else if (output_ref[i] == CONV_2D_MAX) {
      max_cntr++;
    }
    cntr++;
  }

  if (failed)
    LOG("[conv_2d] Standalone test failed with %d wrong out of %d, %d zeros, %d maxes, and a max difference of %d!\n", fail_cntr, cntr, zero_cntr, max_cntr, max_diff);
  else 
    LOG("[conv_2d] Standalone test passed!\n");

  free(input);
  free(output_acc);
  free(output_ref);
  free(mask);
}
#endif
#endif
