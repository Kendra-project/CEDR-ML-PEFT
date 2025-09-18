ARCH ?= aarch64

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build-arm/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME_READ = udma_read
APP_NAME_WRITE = udma_write
SOURCES_READ = udma_read.cpp
SOURCES_WRITE = udma_write.cpp
ifeq ($(ARCH),aarch64)
        CXX = aarch64-linux-gnu-g++
        CC = aarch64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../build-arm/libdash
else
$(error ARCH has to be aarch64, not supported for $(ARCH))
endif

ifeq (ZIP, $(findstring ZIP, $(CFLAGS)))
	ACC = zip
else
	ifeq (GEMM, $(findstring GEMM, $(CFLAGS)))
		ACC = gemm
	else
		ACC = fft
		ifneq (FFT, $(findstring FFT, $(CFLAGS)))
			CFLAGS += -DFFT
		endif
	endif
endif
$(info Building for $(ACC))

read: $(SOURCES_READ)
	$(CXX) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_READ) -o $(APP_NAME_READ)-$(ACC)-$(ARCH).so
	$(CXX) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_READ) -l:libdash.a -lm -o $(APP_NAME_READ)-$(ACC)-$(ARCH).out

write: $(SOURCES_WRITE)
	$(CXX) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_WRITE) -o $(APP_NAME_WRITE)-$(ACC)-$(ARCH).so
	$(CXX) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_WRITE) -l:libdash.a -lm -o $(APP_NAME_WRITE)-$(ACC)-$(ARCH).out

clean:
	-rm $(APP_NAME_READ)-*.so
	-rm $(APP_NAME_READ)-*.out
	-rm $(APP_NAME_WRITE)-*.so
	-rm $(APP_NAME_WRITE)-*.out
