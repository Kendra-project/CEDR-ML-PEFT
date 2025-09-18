ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

APP_NAME = SAR
SOURCES = SAR.cpp
SOURCES_NO_API=SAR-noapi.cpp
NB_SOURCES = SAR_nb.cpp

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../../../build-arm/libdash
endif

default: blocking nonblocking

noapi: $(SOURCES_NO_API)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NO_API) -o $(APP_NAME)-noapi-$(ARCH).so
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_NO_API) -l:libdash.a -o $(APP_NAME)-noapi-$(ARCH).out	
blocking: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

nonblocking: $(NB_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(NB_SOURCES) -o $(APP_NAME)_nb-$(ARCH).so

standalone_nb: $(APP_NAME)_nb-$(ARCH).out
$(APP_NAME)_nb-$(ARCH).out: $(NB_SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(NB_SOURCES) -l:libdash.a -o $(APP_NAME)_nb-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
