#pragma once

#include "dash_types.h"
#include "platform.h"

#include <stdint.h>

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#if DASH_PLATFORM != DASH_ZCU102_2FFT_2MMULT_1ZIP_1CONV2D_HWSCHEDULER
  #error "CONV 2D is not supported on this platform!"
#else

#define CONV_2D_MAX_IMAGE_WIDTH     1920
#define CONV_2D_MAX_IMAGE_HEIGHT    1080

#define CONV_2D_MAX_MASK  11
#define MASK_AREA (CONV_2D_MAX_MASK * CONV_2D_MAX_MASK)

// Size of array in bytes divided by size of first element in bytes == number of elements in array
#define NUM_CONV_2DS (sizeof(CONV_2D_CONTROL_BASE_ADDRS) / sizeof(CONV_2D_CONTROL_BASE_ADDRS[0]))

#define UDMABUF_PARTITION_SIZE (CONV_2D_UDMABUF_SIZE / NUM_CONV_2DS)
#define REQUIRED_BUFFER_SIZE (2 * CONV_2D_MAX_IMAGE_WIDTH * CONV_2D_MAX_IMAGE_HEIGHT * sizeof(int16_t))
static_assert(UDMABUF_PARTITION_SIZE >= REQUIRED_BUFFER_SIZE, "Current udmabuf size is too small to support this many CONV_2Ds!");

// Transfers must be multiples of bus size or smaller
#define BUS_SIZE  8

// AXI-LITE ap_ctrl masks
#define CONV_2D_AP_START  0x01 // RW
#define CONV_2D_AP_DONE   0x02 // R
#define CONV_2D_AP_IDLE   0x04 // R
#define CONV_2D_AP_READY  0x08 // R
#define CONV_2D_AUTRST    0x80 // RW
// AXI-LITE address offsets
#define CONV_2D_AP_CTRL   0x00 // 32 bits
#define CONV_2D_FACTOR    0x10 // 32 bits
#define CONV_2D_BIAS      0x18 // 16 bits 
#define CONV_2D_WIDTH     0x20 // 16 bits
#define CONV_2D_HEIGHT    0x28 // 16 bits
#define CONV_2D_STRIDE    0x30 // 16 bits
#define CONV_2D_MASK      512

// Time conversion factors
#define SEC2NANOSEC   1000000000
#define USEC2NANOSEC  1000
#define NANOSEC2USEC  1000

//#define __DASH_CONV_2D_DEBUG__

#ifdef LOG
#undef LOG
#endif

#ifdef __DASH_CONV_2D_DEBUG__
#define LOG(...) printf(__VA_ARGS__)
#else
#define LOG(...)
#endif

typedef dash_re_int16_type pix_t;

void conv_2d_accel(const pix_t* src, pix_t* dst, const char* coeffs, float factor, short bias, unsigned short width, unsigned short height, unsigned short stride, uint8_t resource_idx);

volatile unsigned int* init_conv_2d(unsigned int ctrl_base_addr) {
  int fd;
  volatile unsigned int* virtual_addr;

  if (ctrl_base_addr == 0x00000000) {
    LOG("[conv_2d] Trying to initialize CONV_2D, but its base address is 0, I don't think one is available!\n");
    exit(1);
  }

  LOG("[conv_2d] Initializing CONV_2D at control address 0x%x\n", ctrl_base_addr);
  // Open device memory in order to get access to DMA control slave
  fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd < 0) {
    LOG("[conv_2d] Can't open /dev/mem. Exiting ...\n");
    exit(1);
  }

  // Obtain virtual address to DMA control slave through mmap
  virtual_addr = (volatile unsigned int *)mmap(nullptr,
                                            getpagesize(),
                                            PROT_READ | PROT_WRITE,
                                            MAP_SHARED,
                                            fd,
                                            ctrl_base_addr);

  if (virtual_addr == MAP_FAILED) {
    // TODO: does mmap set errno? might be nice to perror here
    LOG("[ctrl_base_addr] Can't obtain memory map to CONV_2D control slave. Exiting ...\n");
    exit(1);
  }

  close(fd);
  return virtual_addr;
}

void inline conv_2d_write_16(volatile unsigned int *base, unsigned int offset, unsigned short data) { 
  *(unsigned short*)(((unsigned char*)base) + offset) = data; 
}

void inline conv_2d_write_32(volatile unsigned int *base, unsigned int offset, unsigned int data) { 
  *(unsigned int*)(((unsigned char*)base) + offset) = data; 
}

void inline conv_2d_write_fl(volatile unsigned int *base, unsigned int offset, float data) { 
  *(float*)(((unsigned char*)base) + offset) = data; 
}

void inline close_conv_2d(volatile unsigned int* virtual_addr) {
  munmap((unsigned int*)virtual_addr, getpagesize());
}

#endif
