cmake_minimum_required(VERSION 3.18)
project(CEDR VERSION 1.0
        DESCRIPTION "A \"Pretty Good\" (tm) Linux userspace runtime for DAG-based applications"
        LANGUAGES CXX)

include(GNUInstallDirs)

# Figure out the variation to use when building CEDR
set(CEDR_TYPES DAG API)
set(CEDR_TYPE "API" CACHE STRING "Variation of CEDR to build (DAG-based or API-based)")
set(CACHE CEDR_TYPE PROPERTY STRINGS ${CEDR_TYPES})

if (NOT CEDR_TYPE IN_LIST CEDR_TYPES)
  message(FATAL_ERROR "CEDR_TYPE must be one of ${CEDR_TYPES}")
endif()

option(UsePAPI "Include support for performance-counter profiling of applications via PAPI" FALSE)
option(UsePERF "Include support for performance-counter profiling of applications via PERF" TRUE)
option(EnableProfiling "Add compiler-level flags for profiling the underlying CEDR runtime when using the DEBUG build")
option(AnomalyDetection "Enable feature gathering for anomaly detection" True)
option(UsePERF "Include support for performance-counter profiling of applications via PERF" True)
option(TaskLogging "Enable collecting task and schedule decisions-level information" FALSE)
option(EnableCUPTI "Include support for performance-counter profiling of applications via CUPTI" FALSE)
set(CMAKE_CXX_STANDARD 11)

# Flags that are used if CMake is told to generate a debug build (i.e. with -DCMAKE_BUILD_TYPE:STRING=DEBUG)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb -D_GLIBCXX_DEBUG")

if (EnableProfiling)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fprofile-instr-generate -fcoverage-mapping")
endif ()

message(STATUS "Building CEDR with CEDR_TYPE=${CEDR_TYPE}, UsePAPI=${UsePAPI}, UsePERF=${UsePERF}, EnableProfiling=${EnableProfiling}, Target=${CMAKE_SYSTEM_PROCESSOR}")

if (${CEDR_TYPE} STREQUAL DAG)
  set(CEDR_SRCDIR "${CMAKE_CURRENT_SOURCE_DIR}/src-dag")
elseif (${CEDR_TYPE} STREQUAL API)
  set(CEDR_SRCDIR "${CMAKE_CURRENT_SOURCE_DIR}/src-api")
  # Need exported.txt to ensure that enqueue_kernel is exported in the main CEDR binary for child applications to use
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--dynamic-list=${CEDR_SRCDIR}/exported.txt")
endif()

# Build the CEDR-independent version of libdash
set(CEDR_BUILD 0)
add_subdirectory(libdash libdash)

# Build the CEDR-targeted version of libdash
set(CEDR_BUILD 1)
add_subdirectory(libdash libdash-rt)

# Baseline CEDR Sources that are required for all the versions
# Each version should customize this (add to it) as needed in their respective "if" section below
set(CEDR_SOURCES ${CEDR_SRCDIR}/main.cpp ${CEDR_SRCDIR}/config_manager.cpp ${CEDR_SRCDIR}/performance_monitor.cpp 
                 ${CEDR_SRCDIR}/threads.cpp ${CEDR_SRCDIR}/dag_parse.cpp ${CEDR_SRCDIR}/runtime.cpp 
                 ${CEDR_SRCDIR}/scheduler.cpp ${CEDR_SRCDIR}/ipc.cpp)



# Define the main executables and include paths/libraries
add_executable(cedr ${CEDR_SOURCES})
add_executable(sub_dag ${CEDR_SRCDIR}/ipc.cpp ${CEDR_SRCDIR}/sub_dag.cpp)
add_executable(kill_daemon ${CEDR_SRCDIR}/ipc.cpp ${CEDR_SRCDIR}/kill_daemon.cpp)

target_include_directories(cedr PRIVATE extern/include ${CEDR_SRCDIR}/include)
target_link_libraries(cedr PRIVATE pthread dl rt)

target_include_directories(sub_dag PRIVATE extern/include ${CEDR_SRCDIR}/include)
target_link_libraries(sub_dag PRIVATE pthread dl rt)

target_include_directories(kill_daemon PRIVATE extern/include ${CEDR_SRCDIR}/include)
target_link_libraries(kill_daemon PRIVATE pthread dl rt)

if (UsePAPI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSEPAPI")
  target_link_libraries(cedr PRIVATE papi)
endif ()

if (UsePERF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSEPERF")
endif ()

if (AnomalyDetection)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DANOMALY_DETECTION")
endif()

if (TaskLogging)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DTASK_LOGGING")
endif()

if (EnableCUPTI)
  project(LANGUAGES CXX CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCUPTI_LOGGING")
endif()

# Enable additional targets for running clang-format/clang-tidy
# https://www.labri.fr/perso/fleury/posts/programming/using-clang-tidy-and-clang-format.html
file(GLOB_RECURSE
        ALL_CXX_SOURCE_FILES
        ${PROJECT_SOURCE_DIR}/${CEDR_SRCDIR}/*.[ch]pp
        ${PROJECT_SOURCE_DIR}/include/*.[ch]pp)

# Add clang-format target if executable is found
find_program(CLANG_FORMAT "clang-format")
if (CLANG_FORMAT)
    add_custom_target(
            clang-format
            COMMAND clang-format
            -i
            -style=file
            ${ALL_CXX_SOURCE_FILES}
    )
endif ()

# Add clang-tidy target if executable is found
find_program(CLANG_TIDY "clang-tidy")
if (CLANG_TIDY)
    add_custom_target(
            clang-tidy
            COMMAND clang-tidy
            ${ALL_CXX_SOURCE_FILES}
            -config=''
            --
            -std=c++${CMAKE_CXX_STANDARD}
            -I${PROJECT_SOURCE_DIR}/extern/include
            -I${PROJECT_SOURCE_DIR}/include
            -I${PROJECT_SOURCE_DIR}/${CEDR_SRCDIR}/include
    )
endif ()
