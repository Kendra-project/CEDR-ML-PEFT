  rm -rf build
  mkdir build
  cd build/
  cmake ../
  make -j $(nproc)
  cd ..
  cd applications/api_example
  cp api_example_api-x86.so ../../build
  cd ../../build
  cp ../daemon_config.json ./