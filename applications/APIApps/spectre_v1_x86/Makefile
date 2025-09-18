ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash
INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../../build-arm/libdash
	LIBDASH_LIB = ../../../build-arm/libdash
	INCLUDES = -I $(LIBDASH_INC)
	LIBS = -L $(LIBDASH_LIB)	
endif
CC = g++
CFLAGS = -std=c++11 -Wall -O2 
#-march=armv8-a+crc+simd -mtune=cortex-a72 -DSDL_DISABLE_IMMINTRIN_H" CXXFLAGS="-O2 -march=armv8-a+crc+simd -mtune=cortex-a72 -DSDL_DISABLE_IMMINTRIN_H
SOURCE = main.cpp
APP_NAME = spectre_v1

shared_obj:
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(SOURCE) -o $(APP_NAME)-$(ARCH).so
standalone: $(SOURCE)
	#$(CC) $(SOURCE) $(CFLAGS) -o $(APP_NAME)-$(ARCH).out
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(INCLUDES) $(LIBS) $(SOURCE) -l:libdash.a -o $(APP_NAME)-$(ARCH).out
clean: 
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
