ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = radar_correlator

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../../build-arm/libdash
endif

ifeq ($(ARCH), arm32)
        CXX = /usr/bin/arm-linux-gnueabihf-g++-10
        CC = /usr/bin/arm-linux-gnueabihf-gcc-10
        RUNTIME_LIBDIR = ../../../build-arm32/libdash
endif

LIBS = -L $(RUNTIME_LIBDIR)

non-api:
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_non_api.c -l:libdash.a -lm -o $(APP_NAME)_non_api-$(ARCH).out

api:
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(APP_NAME)_fft.c -o $(APP_NAME)_fft-$(ARCH).so

zip:
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(APP_NAME)_zip.c -o $(APP_NAME)_zip-$(ARCH).so
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_zip.c -l:libdash.a -lm -o $(APP_NAME)_zip-$(ARCH).out

.PHONY: standalone
standalone: $(APP_NAME)_fft-$(ARCH).out
$(APP_NAME)_fft-$(ARCH).out:
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_fft.c -l:libdash.a -lm -o $(APP_NAME)_fft-$(ARCH).out

clean:
	-rm $(APP_NAME)*.so
	-rm $(APP_NAME)*.out
