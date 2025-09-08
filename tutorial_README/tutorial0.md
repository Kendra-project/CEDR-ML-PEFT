# Tutorial #0: Getting Started

In this tutorial, we will review the [requirements](#tutorial-requirements) for completing the subsequent tutorials and [verify](#building-cedr-for-x86) that all requirements are met by building CEDR for x86.

## Tutorial Requirements
- Ubuntu-based Linux machine or ability to run a [docker image](https://hub.docker.com/r/uofarcl/cedr/tags)
  * Choose one of the supported environments:
    * [Docker-based setup (Start on any platform without root access)](#option-1-docker-based-instructions-linux-windows-and-mac)
    * [Native Linux-based setup (Requires root access)](#option-2-linux-native-instructions-requires-root-access)
- CEDR Source Files: [CEDR repository for this tutorial](https://github.com/UA-RCL/CEDR/), checked out to the `tutorial` branch.


### Option 1: Docker-based Instructions (Linux, Windows, and Mac)
Install Docker based on the host machine platform using the [link](https://docs.docker.com/engine/install/#desktop).
Pull the latest [Docker container](https://hub.docker.com/r/uofarcl/cedr/tags) with all dependencies installed. 
Open a terminal in your CEDR folder and run the Docker image using the following command: 
```
docker run -it --name cedr_tutorial uofarcl/cedr:ece506 /bin/bash
```

***After runing the Docker container you can move to [Building CEDR for x86](#building-cedr-for-x86). Following few lines shows simple `docker` comments that would be useful for the following tutorials.***

We will need to copy files from the container to the host machine. Use one of these alternatives for this: 
  * Mount a volume while running Docker using a folder on the host machine that has read and write permissions for `other` users: 
```bash
docker run -it --name cedr_tutorial -v <host-folder>:/root/repository/share uofarcl/cedr:ece506 /bin/bash
```
  * Using `docker cp` to copy files from the container to the host: 
```bash
docker ps -a # Find the Container ID and Name
docker cp cedr_tutorial:/root/repository/CEDR ./
```
  * If you need to detach from Docker at any time, you can use `Crtl+p` and `Ctrl+q` to detach and to re-attach use: 
```bash
docker exec -it cedr_tutorial /bin/bash
```
  * Using `docker start` to restart a stopped Docker container: 
```bash
docker start -ai cedr_tutorial
```
  * Save your changes locally to start a new container with saved changes: 
```bash
docker commit cedr_tutorial <new_image_name>:<new_tag> # Save changes in the current running image to create a new container
docker run -it --name my_cedr_dev <new_image_name>:<new_tag> /bin/bash # Start a new image with the updated changes
```

### Option 2: Linux-native instructions (Requires root access)
Install git using the following command:
```bash
sudo apt-get install git-all
```

Clone CEDR from GitHub using one of the following methods:
  * Clone with ssh:
```bash
git clone -b tutorial git@github.com:UA-RCL/CEDR.git
```
  * Clone with https:
```bash
git clone -b ece506 https://github.com/UA-RCL/CEDR.git
```

Change your working directory to the cloned CEDR folder
```bash
cd CEDR
```

Install all the required dependencies using the following command (this will take some time):
```bash
sudo bash install_dependencies.sh
```

## Building CEDR for x86:

Navigate to the [root directory](https://github.com/UA-RCL/CEDR/tree/ece506) and create a build folder
```bash
mkdir build
```
Change the current directory to build
```bash
cd build
```
Call cmake and make to create CEDR executables for x86. You can leave `$(nproc)` empty (`make -j`) or set it equal to the number of cores you have on the host machine (i.e., `make -j 12` for a 12 CPU core machine).
```bash
cmake ../
make -j $(nproc)
```
At this point, there are 4 important files that should be compiled:
 - *cedr:* CEDR runtime daemon
 - *sub_dag:* CEDR job submission process
 - *kill_daemon:* CEDR termination process
 - *libdash-rt/libdash-rt.so:* Shared object used by CEDR for API calls

Look into [dash.h](https://github.com/UA-RCL/CEDR/tree/ece506/libdash/dash.h) under [libdash](https://github.com/UA-RCL/CEDR/tree/ece506/libdash) folder and see available API calls.


## Hardware Information


All the files to recreate the hardware image and the related Petalinux can be found below. You don't need to create these before the tutorial, but you can view the image that will be used and recreate it yourself if you're interested.

Hardware Images: The FPGA image used in this tutorial is available [here](https://github.com/UA-RCL/Hardware-Images/tree/AUP-ZU3-2fft2zip).
 * Block Diagram of the FPGA image can be found [here](https://github.com/UA-RCL/Hardware-Images/blob/AUP-ZU3-2fft2zip/vivado/fft2xzip2x.pdf).
 * We will be using AUP-ZU3 boards. Specification can be found [here](https://www.realdigital.org/hardware/aup-zu3).