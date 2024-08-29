ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = pulse_doppler
SOURCES = pulse_doppler.c
SOURCES_NB = pulse_doppler_nb.c
SOURCES_PHASED_NB = pulse_doppler_phased_nb.c
SOURCES_NB_ZIP = pulse_doppler_nb_zip.c

ifeq ($(ARCH), aarch64)
        CXX = aarch64-linux-gnu-g++
        CC = aarch64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../build-arm/libdash
else ifeq ($(ARCH), riscv64)
        CXX = riscv64-linux-gnu-g++
		CC = riscv64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../build_riscv/libdash
endif

default: blocking nonblocking

zip: $(SOURCES_NB_ZIP)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NB_ZIP) -o $(APP_NAME)-nb-zip-$(ARCH).so
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_NB_ZIP) -l:libdash.a -lm -o $(APP_NAME)-nb-zip-$(ARCH).out

blocking: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

nonblocking: $(SOURCES_NB)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NB) -o $(APP_NAME)-nb-$(ARCH).so

phased: $(SOURCES_PHASED_NB)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_PHASED_NB) -o $(APP_NAME)-phased-nb-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -lm -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
