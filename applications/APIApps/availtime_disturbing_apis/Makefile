ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	LIBDASH_LIB = ../../../../build-arm/libdash
endif

ZIP_APP_NAME = availtime_zip
ZIP_SOURCES = availtime_zip.cpp

all: zip

zip: $(ZIP_APP_NAME)-$(ARCH).so $(ZIP_APP_NAME)-$(ARCH).out

$(ZIP_APP_NAME)-$(ARCH).so: $(ZIP_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(ZIP_SOURCES) -o $(ZIP_APP_NAME)-$(ARCH).so

standalone: $(ZIP_APP_NAME)-$(ARCH).out
$(ZIP_APP_NAME)-$(ARCH).out: $(ZIP_SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(ZIP_SOURCES) -l:libdash.a -o $(ZIP_APP_NAME)-$(ARCH).out

clean:
	-rm $(ZIP_APP_NAME)-*.so
	-rm $(ZIP_APP_NAME)-*.out
