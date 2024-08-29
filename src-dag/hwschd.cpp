#include "platform.h"
#include "dma.h"
#include "hwschd.h"
#include <plog/Log.h>

#include "string.h"
#include <stdio.h>

static volatile unsigned int* dma_control_base_addr;
static volatile unsigned int* udmabuf_base_addr;
static uint64_t udmabuf_phys_addr;

bool setup_hwschd() {
    /* Set up AXI DMA */
    LOG_DEBUG << "setup_hwschd(): Calling init_dma()";
    if (!HWSCHD_DMA_CTRL_BASE_ADDR) {
        LOG_ERROR << "Hardware scheduler DMA based address defined as zero. It doesn't look like it's present.";
        return false;
    }
    dma_control_base_addr = init_dma(HWSCHD_DMA_CTRL_BASE_ADDR);
    LOG_DEBUG << "setup_hwschd(): Calling reset_dma()";
    reset_dma(dma_control_base_addr);

    /* Set up U-DMA buffer */
    LOG_DEBUG << "setup_hwschd(): Calling init_udmabuf()";
    init_udmabuf(HWSCHD_UDMABUF_NUM, HWSCHD_UDMABUF_SIZE, &udmabuf_base_addr, &udmabuf_phys_addr);
    LOG_DEBUG << "setup_hwschd(): Didn't crash yet.";
    return true;
}

bool teardown_hwschd() {
    close_udmabuf(udmabuf_base_addr, HWSCHD_UDMABUF_SIZE);
    close_dma(dma_control_base_addr);
    return true;
}

void hwschd_kern(unsigned int* avail_times, task_enq_t* ready_queue, task_deq_t* schedule, unsigned int num) {

#if ((HWSCHD_P % 2) != 0)
    volatile unsigned int* base_addr = (volatile unsigned int*)((char*)udmabuf_base_addr + sizeof(unsigned int));
    const uint64_t phys_addr = udmabuf_phys_addr + sizeof(unsigned int);
#else
    volatile unsigned int* base_addr = udmabuf_base_addr;
    const uint64_t phys_addr = udmabuf_phys_addr;
#endif

    LOG_DEBUG << "hwschd_kern: num: " << num;

    /* Determine packet sizes */
    const size_t update_size = HWSCHD_P * sizeof(uint32_t);
    const size_t queue_size = num * sizeof(task_enq_t);
    const size_t input_size = update_size + queue_size;
    const size_t output_size = num * sizeof(task_deq_t);

    /* Prepare transfer */
    volatile task_enq_t* hwschd_input = (volatile task_enq_t*) base_addr;
#if ((HWSCHD_P % 2) != 0)
    volatile task_deq_t* hwschd_output;
    if (num & 1) { // Odd number of tasks
        hwschd_output = (volatile task_deq_t*) ((char*) base_addr + input_size + sizeof(unsigned int));
    } else { // Even number of tasks
        hwschd_output = (volatile task_deq_t*) ((char*) base_addr + input_size);
    }
#else
    volatile task_deq_t* hwschd_output = (volatile task_deq_t*) ((char*) base_addr + input_size);
#endif

#if ((HWSCHD_P % 2) != 0)
    // Copy availability time for first PE, and then memcpy the rest
    *((unsigned int*)hwschd_input) = avail_times[0];
    memcpy((task_enq_t*) ((char*) hwschd_input + sizeof(task_enq_t)), &avail_times[1], update_size - sizeof(unsigned int));
#else
    memcpy((task_enq_t*) hwschd_input, avail_times, update_size);
#endif

#if ((HWSCHD_P % 2) != 0)
    if (num > 1) {
        if (num & 1) { // Odd number of tasks
            memcpy((task_enq_t*) ((char*) hwschd_input + update_size), ready_queue, queue_size - sizeof(task_enq_t));
            ((task_enq_t*) ((char*) hwschd_input + update_size + queue_size - sizeof(task_enq_t)))->tid = ready_queue[num - 1].tid;
            ((task_enq_t*) ((char*) hwschd_input + update_size + queue_size - sizeof(task_enq_t)))->tinfo = ready_queue[num - 1].tinfo;
            for (unsigned int i = 0; i < HWSCHD_P; i++) {
                *((uint32_t*) ((char*) hwschd_input + update_size + queue_size - sizeof(task_enq_t) + (2 + i) * sizeof(uint32_t))) = *((uint32_t*)((char*) &ready_queue[0] + (num - 1) * sizeof(task_enq_t) + (2 + i) * sizeof(uint32_t)));
            }
        } else { // Even number of tasks
            memcpy((task_enq_t*) ((char*) hwschd_input + update_size), ready_queue, queue_size);
        }
    } else { // 1 task
        ((task_enq_t*) ((char*) hwschd_input + update_size))->tid = ready_queue[0].tid;
        ((task_enq_t*) ((char*) hwschd_input + update_size))->tinfo = ready_queue[0].tinfo;
        for (unsigned int i = 0; i < HWSCHD_P; i++) {
            *((uint32_t*) ((char*) hwschd_input + update_size + (2 + i) * sizeof(uint32_t))) = *((uint32_t*)((char*) &ready_queue[0] + (2 + i) * sizeof(uint32_t)));
        }
    }
#else
    memcpy((task_enq_t*) ((char*) hwschd_input + update_size), ready_queue, queue_size);
#endif

    LOG_DEBUG << "hwschd_kern: Calling setup_rx()";

#if ((HWSCHD_P % 2) != 0)
    if (num & 1) { // Odd number of tasks
        setup_rx(dma_control_base_addr, phys_addr + input_size + sizeof(unsigned int), output_size);
    } else { // Even number of tasks
        setup_rx(dma_control_base_addr, phys_addr + input_size, output_size);
    }
#else
    setup_rx(dma_control_base_addr, phys_addr + input_size, output_size);
#endif

    LOG_DEBUG << "hwschd_kern: Calling setup_tx()";

    /* Send ready queue */
    setup_tx(dma_control_base_addr, phys_addr, input_size);

    /* Receive scheduling decisions */
    dma_wait_for_rx_complete(dma_control_base_addr);
    memcpy(schedule, (task_deq_t*)hwschd_output, output_size);
}
