#!/bin/bash
cwd=$(pwd)
cd ~
git clone https://gitlab.com/libeigen/eigen.git
cd eigen
git checkout 3.4.0
mkdir build
cd build
cmake ..
sudo make -j$(nproc) install
cd "$cwd"
mkdir build
cd build
cmake ..
make -j
cd "$cwd"

sudo apt-get install libncurses5-dev
sudo apt-get install libglew-dev
sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
sudo apt-get install ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev
sudo apt-get install libdc1394-22-dev libraw1394-dev
sudo apt-get install libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev
sudo apt-get install libboost-all-dev libopenblas-dev
sudo apt-get install libbluetooth-dev
sudo apt install libpcl-dev
sudo apt -y install libxext-dev libxfixes-dev libxrender-dev libxcb1-dev libx11-xcb-dev libxcb-glx0-dev
sudo apt -y install libxkbcommon-dev libxcb-keysyms1-dev libxcb-image0-dev libxcb-shm0-dev libxcb-icccm4-dev libxcb-sync0-dev libxcb-xfixes0-dev libxcb-shape0-dev libxcb-randr0-dev libxcb-render-util0-dev
chmod +x opencv3.4.16Install.sh
bash opencv3.4.16Install.sh

cd ~
git clone https://github.com/nlohmann/json.git
cd json
mkdir build
cd build
cmake ..
sudo make -j$(nproc) install
cd ~
git clone https://github.com/lava/matplotlib-cpp.git
cd matplotlib-cpp
mkdir build
cd build
cmake ..
sudo make -j$(nproc) install
