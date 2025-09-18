ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = spectre_v1
SOURCES = spectre_taishan.c

ifeq ($(ARCH), aarch64)
        CXX = aarch64-linux-gnu-g++
        CC = aarch64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../build-arm/libdash
endif

$(APP_NAME)-$(ARCH).so: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -lm -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
