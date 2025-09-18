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

FFT_APP_NAME = DoS_fft
FFT_SOURCES = DoS_fft.cpp

ZIP_APP_NAME = DoS_zip
ZIP_SOURCES = DoS_zip.cpp

all: fft zip

fft: $(FFT_APP_NAME)-$(ARCH).so $(FFT_APP_NAME)-$(ARCH).out

zip: $(ZIP_APP_NAME)-$(ARCH).so $(ZIP_APP_NAME)-$(ARCH).out

$(ZIP_APP_NAME)-$(ARCH).so: $(ZIP_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(ZIP_SOURCES) -o $(ZIP_APP_NAME)-$(ARCH).so

$(ZIP_APP_NAME)-$(ARCH).out: $(ZIP_SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(ZIP_SOURCES) -l:libdash.a -o $(ZIP_APP_NAME)-$(ARCH).out

$(FFT_APP_NAME)-$(ARCH).so: $(FFT_SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(FFT_SOURCES) -o $(FFT_APP_NAME)-$(ARCH).so

standalone: $(FFT_APP_NAME)-$(ARCH).out
$(FFT_APP_NAME)-$(ARCH).out: $(FFT_SOURCES)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(FFT_SOURCES) -l:libdash.a -o $(FFT_APP_NAME)-$(ARCH).out
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(ZIP_SOURCES) -l:libdash.a -o $(ZIP_APP_NAME)-$(ARCH).out
clean:
	-rm $(FFT_APP_NAME)-*.so
	-rm $(FFT_APP_NAME)-*.out
	-rm $(ZIP_APP_NAME)-*.so
	-rm $(ZIP_APP_NAME)-*.out	
