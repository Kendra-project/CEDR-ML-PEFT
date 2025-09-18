ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = radar_correlator
SOURCES = radar_correlator.c
SOURCES_ZIP = radar_correlator_zip.c
SOURCES_NO_API = radar_correlator_no_api.c
ifeq ($(ARCH),aarch64)
        CXX = aarch64-linux-gnu-g++
        CC = aarch64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../build-arm/libdash
endif

$(APP_NAME)-$(ARCH).so: $(SOURCES)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

noapi: $(SOURCES_NO_API)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NO_API) -o $(APP_NAME)-noapi-$(ARCH).so
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_NO_API) -l:libdash.a -lm -o $(APP_NAME)-noapi-$(ARCH).out
zip: $(SOURCES_ZIP)
	$(CC) $(CFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_ZIP) -o $(APP_NAME)-zip-$(ARCH).so
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_ZIP) -l:libdash.a -lm -o $(APP_NAME)-zip-$(ARCH).out

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CC) $(CFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES)  -l:libdash.a -lm -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
