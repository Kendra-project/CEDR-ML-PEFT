cd build-arm
cmake -DLIBDASH_MODULES="FFT GEMM" --toolchain=../toolchains/aarch64-linux-gnu.toolchain.cmake ..
make -j
ls
cd ../applications/APIApps/radar_correlator
ls
ARCH=aarch64 make 
ls
file radar_correlator-aarch64.so 
exit
