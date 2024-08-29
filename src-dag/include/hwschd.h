#pragma once

#include "platform.h"

#define HWSCHD_P    4
#define HWSCHD_D    512
#define HWSCHD_W    16

#if ((HWSCHD_P % 2) != 0)
#warning "Scheduler input is not a multiple of 64 bits!"
#endif

#define HWSCHD_PE_NOT_SUPPORTED 0x80000000 // MSB will exclude PE from mapping

typedef struct __attribute__ ((__packed__)) task_enq {
    uint32_t tid;
    uint32_t tinfo;
    uint32_t runtimes[HWSCHD_P];
} task_enq_t;

typedef struct __attribute__ ((__packed__)) task_deq {
    uint32_t tid;
    uint32_t decision;
} task_deq_t;

bool setup_hwschd();
bool teardown_hwschd();
void hwschd_kern(unsigned int* avail_times, task_enq_t* ready_queue, task_deq_t* schedule, unsigned int num);
