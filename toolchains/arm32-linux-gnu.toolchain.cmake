# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET(CMAKE_SYSTEM_NAME Linux)
SET(CMAKE_SYSTEM_PROCESSOR arm)
SET(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++-10)
SET(CMAKE_AR /usr/bin/arm-linux-gnueabihf-ar)
SET(CMAKE_RANLIB /usr/bin/arm-linux-gnueabihf-ranlib)
SET(CMAKE_ADDR2LINE /usr/bin/arm-linux-gnueabihf-addr2line)
SET(CMAKE_AR /usr/bin/arm-linux-gnueabihf-ar)
SET(CMAKE_CXX_COMPILER_AR /usr/bin/arm-linux-gnueabihf-gcc-ar)
SET(CMAKE_CXX_COMPILER_RANLIB /usr/bin/arm-linux-gnueabihf-gcc-ranlib)
SET(CMAKE_C_COMPILER_AR /usr/bin/arm-linux-gnueabihf-gcc-ar)
SET(CMAKE_C_COMPILER_RANLIB /usr/bin/arm-linux-gnueabihf-gcc-ranlib)
SET(CMAKE_LINKER /usr/bin/arm-linux-gnueabihf-ld)
SET(CMAKE_NM /usr/bin/arm-linux-gnueabihf-nm)
SET(CMAKE_OBJCOPY /usr/bin/arm-linux-gnueabihf-objcopy)
SET(CMAKE_OBJDUMP /usr/bin/arm-linux-gnueabihf-objdump)
SET(CMAKE_RANLIB /usr/bin/arm-linux-gnueabihf-ranlib)
SET(CMAKE_READELF /usr/bin/arm-linux-gnueabihf-readelf)
SET(CMAKE_STRIP /usr/bin/arm-linux-gnueabihf-strip)
SET(GIT_EXECUTABLE /usr/bin/git)

# Rather than make everyone recompile libgsl/etc all the time, we can just version control libraries here...
set(CMAKE_SYSTEM_LIBRARY_PATH
  ${CMAKE_CURRENT_LIST_DIR}/arm32-linux-gnu/lib
  /usr/arm32-linux-gnu/lib
)
