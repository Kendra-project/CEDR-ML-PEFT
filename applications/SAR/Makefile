ARCH ?= x86

LIBDASH_INC ?= ../../libdash
LIBDASH_LIB ?= ../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../build-arm/libdash
	LIBS = -L $(RUNTIME_LIBDIR)
endif

ifeq ($(ARCH), arm32)
    CXX = /usr/bin/arm-linux-gnueabihf-g++
    CC = /usr/bin/arm-linux-gnueabihf-gcc
    RUNTIME_LIBDIR = ../../build-arm32/libdash
	LIBS = -L $(RUNTIME_LIBDIR)
endif

APP_NAME = SAR
SOURCES = SAR.cpp
NB_SOURCES = SAR_nb.cpp
NB_SOURCES_STATIC_INPUT = SAR_nb_static_input.cpp

non-api: $(APP_NAME)_non_api.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_non_api.cpp -lm -lgsl -o $(APP_NAME)_non_api-$(ARCH).out

api: $(APP_NAME)_api.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(APP_NAME)_api.cpp -o $(APP_NAME)_api-$(ARCH).so
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(APP_NAME)_api.cpp -l:libdash.a -lm -o $(APP_NAME)_api-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
