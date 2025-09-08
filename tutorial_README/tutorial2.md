# Class Tutorial #2: FPGA Integration and Accelerator Testing

Table of Contents:
1. [Start with an existing FPGA image containing FFT and ZIP accelerators](#1-hardware-information)
   1. Cross Compilation:
      * [Build CEDR for the FPGA](#111-build-cedr-for-the-fpga)
      * [Build sample application for the FPGA](#112-application-cross-compilation)
   2. [Access the FPGA hardware and testing CEDR on the FPGA using CPUs only](#12-running-cedr-on-aup-zu3)
2. [Add a new FFT accelerator to CEDR](#2-fft-acceleretor-integration)
   1. [Rebuild CEDR with the updated configuration](#21-building-cedr-with-fft)
   2. [Test the FFT accelerator with CEDR](#22-testing-fft-on-aup-zu3)
      * [Validate execution using Gantt charts](#221-gantts-from-aup-zu3-experiments)

## 1. Hardware Information

For this tutorial, you will begin with an FPGA image that already includes FFT and ZIP accelerators. You do not need to generate the image yourself before the tutorial—everything required is provided. However, if you’re interested, you can explore the design files and even rebuild the image on your own.

  * FPGA Image: [GitHub here](https://github.com/UA-RCL/Hardware-Images/tree/AUP-ZU3-2fft2zip).
  * Block Diagram: [View here](https://github.com/UA-RCL/Hardware-Images/blob/AUP-ZU3-2fft2zip/vivado/fft2xzip2x.pdf).
  * Target Board: We will use the [AUP-ZU3 board](https://www.realdigital.org/hardware/aup-zu3) for this tutorial.

### 1.1.1. Build CEDR for the FPGA

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

Moving on to the aarch64-based build for AUP-ZU3 FPGA with accelerators. We'll start by building CEDR itself. This time we will use the [toolchain](/toolchains/aarch64-linux-gnu.toolchain.cmake) file for cross-compilation. If you are on Ubuntu 22.04, the toolchain requires running inside the Docker container.
Run the following commands from the repository root folder to build CEDR for the AUP-ZU3 board:

```bash
mkdir build-arm
cd build-arm
cmake --toolchain=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j $(nproc)
```

This will create an executable file for `cedr`, `sub_dag`, `kill_deamon`, and `libdash-rt` for aarch64 platforms. We can check the platform type of an executable using the `file` command:

```bash
file cedr
```
```
cedr: ELF 64-bit LSB pie executable, ARM aarch64, version 1 (GNU/Linux), dynamically linked, interpreter /lib/ld-linux-aarch64.so.1, BuildID[sha1]=40df8392a1e9c1cb31478e800ea1c7d4955349be, for GNU/Linux 3.7.0, not stripped
```

Next, we can build our application using cross-compilation for ARM. 

### 1.1.2. Application Cross-compilation

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

First, navigate to [applications/api_example](/applications/api_example/) folder. Then run the following command to build the executable for aarch64:

```bash
cd ../applications/api_example
ARCH=aarch64 make api
file api_example_api-aarch64.so
```
```
api_example_api-aarch64.so:  ELF 64-bit LSB shared object, ARM aarch64, version 1 (SYSV), dynamically linked, BuildID[sha1]=f86ca08b59c2769a26d2b236731e163ee8e0ba1f, not stripped
```

After verifying the file is compiled for the correct platform, copy the file and inputs to the build directory:

```bash
# Assuming your CEDR build folder is in the root directory and named "build-arm"
cp api_example_api-aarch64.{so,out} ../../build-arm
```

### 1.2. Running CEDR on AUP-ZU3

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

Now, change your working directory to the `build-arm` directory. Before going into the AUP-ZU3, first copy the [daemon_config.json](/daemon_config.json) file to the `build-arm` directory.

```bash
cd ../../build-arm
cp ../daemon_config.json ./
ls
```

Now we will copy our files to AUP-ZU3 and enter the associated password when prompted. Assuming using Docker, copy the `CEDR/build-arm` folder to the host machine. Although we only need a handful of files to run CEDR and the sample application, we will copy everything under `build-arm`. Running on the host machine:

```bash
docker cp <container_name>:/root/repository/CEDR/build-arm ./
```

Before copying files to the FPGA board, make sure you reserve space on the Appointment Sheet shared on D2L. After reserving the space SSH to the board on a different terminal (or using tools like [PuTTY](https://www.chiark.greenend.org.uk/~sgtatham/putty/latest.html)) and create a folder for your group.

```bash
ssh petalinux@<aup-zu3-IP> # You need to be in UA ENGR VPN to access!
# Enter password when prompted
mkdir ece506_<group-id> # Create a folder for your group
cd ece506_<group-id> # Change to your groups folder
```

After creating a folder for the group, copy your files to the created folder on the host machine you have been working on.

```bash
scp -r <path_to_CEDR_repo_folder>/build-arm petalinux@<aup-zu3-IP>:/home/petalinux/ece506_<group-id> # <path_to_CEDR_repo_folder> can also be the folder you use `docker cp` in. Can find the full path using `pwd` command. 
```

Typing `ls` on the terminal with an SSH connection, you should see all the files you copied on the AUP-ZU3 board.

```bash
ls # Should be in the ~/ece506_<group-id> folder
```

Before running CEDR, let's take a look at the [daemon_config.json](/daemon_config.json) file.

```json
"Worker Threads": {
        "cpu": 3,
        "fft": 0,
        "gemm": 0,
        "gpu": 0
 },
```
Execution of CEDR is the same as the x86_64 version. In one terminal, launch CEDR (or directly as a background process):

```bash
./cedr -c ./daemon_config.json &
```

In another terminal, we will submit 1 instance of `api_example_api-aarch64.so` using `sub_dag` and check the outputs:

```bash
./sub_dag -a ./api_example_api-aarch64.so -n 1 -p 0
```

Now kill CEDR by running `./kill_deamon` and check the `resource_name` fields of all the APIs using the 3 CPUs:

```bash
./kill_daemon
cat ./log_dir/experiment0/timing_trace.log
```

We can see that all FFT and ZIP APIs are only using CPUs (`cpu1`, `cpu2`, and `cpu3`) for execution.

## 2. FFT Accelerator Integration

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

Update the CEDR header file ([src-api/include/header.hpp](/src-api/include/header.hpp)) to include fft as a resource by updating the following lines:

<pre>
enum resource_type { cpu = 0, mmult = 1, gpu = 2, <b>fft = 3, NUM_RESOURCE_TYPES = 4 </b>};
static const char *resource_type_names[] = {"cpu", "gemm", "gpu"<b>, "fft"</b>};
static const std::map<std::string, resource_type> resource_type_map = {
 {resource_type_names[(uint8_t) resource_type::cpu], resource_type::cpu},
 {resource_type_names[(uint8_t) resource_type::mmult], resource_type::mmult},
 {resource_type_names[(uint8_t) resource_type::gpu], resource_type::gpu}<b>,
 {resource_type_names[(uint8_t) resource_type::fft], resource_type::fft}</b>};
</pre>

These lines make sure the functions with `_fft` suffix are grabbed when CEDR starts and used when schedulers assign tasks to the FFT accelerator.

### Accelerator Source File

Add accelerator implementation of the FFT to [libdash/fft](/libdash/). For simplicity, we copy the existing implementation.
```bash
cp -r ../original_files/fft ../libdash/
```

In the `libdash/fft/fft.cpp`, there is a `DASH_FFT_flt_fft` function call. The `DASH_FFT_flt_fft` functions handle input type conversion and call another function called `fft_accel`, which handles the data transfers and signals the FFT accelerator to start execution upon returning from `fft_accel`. Output conversions are handled before finalizing the API call. Looking deeper into the `fft_accel` function:

* ***config_fft(fft_control_base, log2(fft_size)):***: Sets the FFT size as `log2(fft_size)` for forward FFT
* ***config_ifft(fft_control_base, log2(fft_size)):***: Sets the FFT size as `log2(fft_size)` for backward FFT
* ***memcpy((unsigned int\*) udmabuf_base, input, fft_size \* sizeof(fft_cmplx_type)):***: Sends the first input to UDMA for FFT accelerator to use
* ***setup_rx(dma_control_base, udmabuf_phys + (fft_size \* sizeof(fft_cmplx_type)), fft_size \* sizeof(fft_cmplx_type)):***: Sets the return address for FFT accelerator to put the output once it is ready
* ***setup_tx(dma_control_base, udmabuf_phys, fft_size \* sizeof(fft_cmplx_type)):***: Sets the input address for FFT accelerator to start reading the input from and start its execution
* ***dma_wait_for_rx_complete(dma_control_base):***: Waits for FFT accelerator to complete writing the output to the given UDMA address with `setup_rx` function
* ***memcpy(output, (unsigned int\*) &udmabuf_base[2 \* fft_size], fft_size \* sizeof(fft_cmplx_type)):***: Copies the output back to the memory location given as output to this function (move from UDMA to virtual memory of the given pointer)

These are some of the main steps for adding a new accelerator to CEDR.

We also need to ensure that `CMakeLists.txt` in the `libdash` folder searches for FFT as an accelerator. Update following line (19) in [libdash/CMakeLists.txt](/libdash/CMakeLists.txt)
```CMake
 set(ALL_LIBDASH_MODULES GEMM GPU FFT)
```

### 2.1. Building CEDR with FFT

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

In the `build-arm` folder, run the following steps to rebuild CEDR with ZIP as an accelerator.

```bash
rm -rf ./* # This can be skipped, used for showing a fresh start of cmake with a new accelerator
cmake -DLIBDASH_MODULES="FFT" --toolchain=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j
```

Now, verify the functions with the `_fft` suffix that are used for the FFT accelerator.

```bash
nm -D libdash-rt/libdash-rt.so | grep -E '*_fft$'
```

<pre>
000065c5 T <b>DASH_FFT_flt_fft</b>
</pre>

### 2.2. Testing FFT on AUP-ZU3

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

Repeat the steps taken in the [running CEDR on AUP-ZU3 section](#12-running-cedr-on-aup-zu3) to copy updated files to the FPGA.

```bash
docker cp <container_name>:/root/repository/CEDR/build-arm ./ # Copy files out of Docker if needed
scp -r <path_to_CEDR_repo_folder>/build-arm petalinux@<aup-zu3-IP>:/home/petalinux/ece506_<group-id> # <path_to_CEDR_repo_folder> can also be the folder you use `docker cp` in. Can find the full path using `pwd` command. 
```

This time, before running CEDR, we need to enable the FFT accelerator in the [daemon_config.json](/daemon_config.json) file. By the time this tutorial was written, we had 2 FFT accelerators available in the FPGA image. We can enter any number between 0 and 2 in the corresponding fields of the [daemon_config.json](/daemon_config.json) file. Change the file with the following `Worker Threads` setup:

```json
"Worker Threads": {
        "cpu": 3,
        "fft": 2,
        "gemm": 0,
        "gpu": 0
 },
```

Run CEDR on the FPGA by following the same steps outlined in the [running CEDR on AUP-ZU3 section](#12-running-cedr-on-aup-zu3) on the terminal with the SSH connection, but with a minor modification: use `sudo` to run all commands. Since we need to access UDMA driver files, we need `root` access to use the accelerators.

```bash
sudo ./cedr -c ./daemon_config.json # This time do not use & otherwise you won't be able to enter the password for the `sudo`
# After CEDR starts, you can do `Ctrl+z` and type `bg` to proceed in the same terminal
sudo ./sub_dag -a ./api_example_api-aarch64.so -n 1 -p 0
sudo ./kill_daemon
```

After terminating CEDR by running `./kill_deamon`, check the `resource_name` fields of all the APIs in the `timing_tace.log` for FFTs:

```bash
head -n 10 ./log_dir/experiment1/timing_trace.log
```

We can see that all the resources available for FFT execution (`cpu1`, `cpu2`, `cpu3`, `fft1`, and `fft2`) are being used for FFT API executions.

#### 2.2.1. Gantts from AUP-ZU3 Experiments:

[Return to top](#class-tutorial-2-fpga-integration-and-accelerator-testing)

After running these experiments, you can copy the log files back from AUP-ZU3 to your host machine and plot the Gantts using the same setup as before. The Gantt of `experiment0` will only show 3 CPUs being used, while `experiment1` will show 3 CPUs and 2FFTs in use.

```bash
# On the host machine
mkdir <path_to_CEDR_repo_folder>/build-arm/arm-logs
scp -r petalinux@<aup-zu3-IP>:/home/petalinux/ece506_<group-id>/build-arm/log_dir/* <path_to_CEDR_repo_folder>/build-arm/arm-logs # Instead of `log_dir/*` you can specifiy to copy select results like `log_dir/experiment0`

# If needed, copy the logs to the Docker container to run the Python scripts!

# Assuming you are in the `build-arm` folder
python3 ../scripts/gantt_k-nk.py arm-logs/experiment0/timing_trace.log
cp gantt.png gantt_cpu.png
python3 ../scripts/gantt_k-nk.py arm-logs/experiment1/timing_trace.log
cp gantt.png gantt_fft.png
```