ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

ifeq ($(ARCH), aarch64)
        CXX = aarch64-linux-gnu-g++
        CC = aarch64-linux-gnu-gcc
        RUNTIME_LIBDIR = ../../../build-arm/libdash
endif

APP_NAME = temporal_mitigation
SOURCES = main.cpp adjoint.cpp alternateinverse.cpp determinant.cpp display.cpp divide.cpp getcofactor.cpp hermitian.cpp imagpart.cpp inverse.cpp mmadd.cpp mmult.cpp mmult4.cpp mmult64.cpp mmultiply.cpp msub.cpp realpart.cpp scalableinverse.cpp
SOURCES_NO_API = main-noapi.cpp adjoint.cpp alternateinverse.cpp determinant.cpp display.cpp divide.cpp getcofactor.cpp hermitian.cpp imagpart.cpp inverse.cpp mmadd.cpp mmult.cpp mmult4.cpp mmult64.cpp mmultiply.cpp msub.cpp realpart.cpp scalableinverse.cpp
SOURCES_ZIP = main_zip.cpp adjoint.cpp alternateinverse.cpp determinant.cpp display.cpp divide.cpp getcofactor.cpp hermitian.cpp imagpart.cpp inverse.cpp mmadd.cpp mmult.cpp mmult4.cpp mmult64.cpp mmultiply.cpp msub.cpp realpart.cpp scalableinverse.cpp

$(APP_NAME)-$(ARCH).so: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES) -o $(APP_NAME)-$(ARCH).so

zip: $(SOURCES_ZIP)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_ZIP) -o $(APP_NAME)-$(ARCH).so
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_ZIP) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

noapi: $(SOURCES_NO_API)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCES_NO_API) -o $(APP_NAME)-noapi-$(ARCH).so
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES_NO_API) -l:libdash.a -o $(APP_NAME)-noapi-$(ARCH).out
	
.PHONY: standalone
standalone: $(APP_NAME)-$(ARCH).out
$(APP_NAME)-$(ARCH).out: $(SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCES) -l:libdash.a -o $(APP_NAME)-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
