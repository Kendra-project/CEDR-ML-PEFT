ARCH ?= x86

LIBDASH_INC ?= ../../../libdash
LIBDASH_LIB ?= ../../../build/libdash

INCLUDES = -I $(LIBDASH_INC)
LIBS = -L $(LIBDASH_LIB)

ifeq ($(ARCH), aarch64)
	CXX = aarch64-linux-gnu-g++
	CC = aarch64-linux-gnu-gcc
	RUNTIME_LIBDIR = ../../../build-arm/libdash
	LIBS = -L $(RUNTIME_LIBDIR)
endif

ifeq ($(ARCH), arm32)
    CXX = /usr/bin/arm-linux-gnueabihf-g++
    CC = /usr/bin/arm-linux-gnueabihf-gcc
    RUNTIME_LIBDIR = ../../../build-arm32/libdash
	LIBS = -L $(RUNTIME_LIBDIR)
endif

APP_NAME = TX

C_SOURCES = ../src/crc.c ../src/txData.c ../src/baseband_lib.c ../src/viterbi.c \
	../src/interleaver_deintleaver.c ../src/scrambler_descrambler.c ../src/qpsk_Mod_Demod.c \
	../src/CyclicPrefix.c ../src/Preamble_ST_LG.c ../src/diag.c ../src/rfInf.c ../src/pilot.c
CPP_SOURCES = ../src/TX.cpp ../src/fft_hs.cpp
CPP_SOURCES_NB = ../src/TX_nb.cpp ../src/fft_hs.cpp
CPP_SOURCES_NB_STATIC_INPUT = ../src/TX_nb_static_input.cpp

SOURCES = $(C_SOURCES) $(CPP_SOURCES)
SOURCES_NB = $(C_SOURCES) $(CPP_SOURCES_NB)
SOURCES_NB_STATIC_INPUT = $(C_SOURCES) $(CPP_SOURCES_NB_STATIC_INPUT)

non-api: ../src/$(APP_NAME)_non_api.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(C_SOURCES) ../src/$(APP_NAME)_non_api.cpp -lm -lgsl -lgslcblas -o $(APP_NAME)_non_api-$(ARCH).out

api: ../src/$(APP_NAME)_api.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -shared -fPIC $(C_SOURCES) ../src/$(APP_NAME)_api.cpp -o $(APP_NAME)_api-$(ARCH).so
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(LIBS) $(C_SOURCES) ../src/$(APP_NAME)_api.cpp -l:libdash.a -lm -o $(APP_NAME)_api-$(ARCH).out

clean:
	-rm $(APP_NAME)-*.so
	-rm $(APP_NAME)-*.out
