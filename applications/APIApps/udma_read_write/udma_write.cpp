#include <fcntl.h>
#include <cstdint>

#include <unistd.h>
#include <sys/mman.h>
#include <cstdio>
#include <cstdlib>

#if defined(ZIP)
	#define ACC 2
#elif defined(GEMM)
	#define ACC 1
#else
	#define ACC 0
#endif

int main(void){
    uint32_t buf_num = ACC;
    size_t udmabuf_size = 1048576;

    int fd, phys_addr_fd;
    volatile unsigned int* virtual_addr = nullptr;
    uint64_t phys_addr = 0;
    char udmabuf_name[10];
    char udmabuf_dev_path[100];
    char udmabuf_sys_path[100];
    char *attr = (char*) calloc(1024, sizeof(char));

    snprintf(udmabuf_name, sizeof(udmabuf_name)/sizeof(udmabuf_name[0]), "udmabuf%d", buf_num);
    snprintf(udmabuf_dev_path, sizeof(udmabuf_dev_path)/sizeof(udmabuf_dev_path[0]), "/dev/udmabuf%d", buf_num);
    snprintf(udmabuf_sys_path, sizeof(udmabuf_dev_path)/sizeof(udmabuf_dev_path[0]), "/sys/class/u-dma-buf/udmabuf%d/phys_addr", buf_num);

    //printf("[dma] udmabuf name: %s, udmabuf dev path: %s, udmabuf sys path: %s\n", udmabuf_name, udmabuf_dev_path, udmabuf_sys_path);

    //printf("[dma] Opening file descriptor to /dev/%s\n", udmabuf_name);
    fd = open(udmabuf_dev_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("[dma] Can't open %s. Exiting ...\n", udmabuf_dev_path);
        exit(1);
    }

    virtual_addr = (volatile unsigned int*) mmap(nullptr,
                                                udmabuf_size,
                                                PROT_READ | PROT_WRITE,
                                                MAP_SHARED,
                                                fd,
                                                0);
    close(fd);

    //printf("1:%u\n", *virtual_addr);
    for (int j=0; j<2000000; j++){ // Number of times to read udma
        for (int i=0; i<512; i++){
            ((float*)virtual_addr)[i] = 0.0;
        }
    }

    munmap((unsigned int*) virtual_addr, udmabuf_size);

}
