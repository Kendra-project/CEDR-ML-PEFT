FROM ubuntu:18.04

WORKDIR /root

ENV DEBIAN_FRONTEND=noninteractive

# Baseline stuff that would basically be on any actual user's machine but not many barebones containers
RUN apt update && apt install -y software-properties-common lsb-release wget git-all

RUN git clone -b tutorial https://github.com/UA-RCL/CEDR.git /root/repository/CEDR

WORKDIR /root/repository/CEDR

RUN chmod +x install_dependencies.sh && ./install_dependencies.sh
RUN pip3 install --no-cache-dir -r requirements.txt

ENV DEBIAN_FRONTEND=dialog

WORKDIR /root/repository
