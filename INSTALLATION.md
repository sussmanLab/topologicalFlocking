# INSTALLATION {#install}


# Sample compilation from a clean install of Ubuntu 24.04

Starting from a fresh copy of Ubuntu 24.04 as an example, the main branch  can be compiled by first installing Cmake, boost, and CGAL. Start in a directory that you don't mind adding some tar files and software folders to, and then run the following commands:

##  clang, CMake, Boost and CGAL:

    $ sudo apt-get install clang
    $ sudo  apt install  cmake  
    $ wget https://boostorg.jfrog.io/artifactory/main/release/1.84.0/source/boost_1_84_0.tar.gz
    $ tar xf boost_1_84_0.tar.gz
    $ cd boost_1_84_0
    $ sudo ./bootstrap.sh
    $ sudo ./b2 install
    $ cd ..
    $ wget https://github.com/CGAL/cgal/releases/download/v5.6/CGAL-5.6.tar.xz
    $ tar xf CGAL-5.6.tar.xz
    $ cd CGAL-5.6
    $ cmake .
    $ make install
    $ cd ..

## Other required packages:

Other dependencies can be install via  apt-get. We need netcdf-cxx (for which we are now using the updated 4.3.X verions), which itself requires things like zlib, hdf5, and netcdf),  and the CGAL headers will need  gmp and mpfr. An MPI package  is needed as well. The following commands will do the trick:

    $ sudo add-apt-repository universe
    $ sudo apt-get update
    $ sudo apt-get install zlib1g-dev libhdf5-dev libnetcdf-dev  netcdf-bin libnetcdf-c++4-dev libgmp-dev libmpfr-dev mpich

### Cuda on WSL2

This code needs cuda, and you should look up your own OS' installation instructions for it. On WSL2, you can do this:

    $ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
    $ sudo dpkg -i cuda-keyring_1.1-1_all.deb
    $ sudo apt-get update
    $ sudo apt-get -y install cuda-toolkit-12-6
    $ sudo apt-get install nvidia-cuda-toolkit
    $ sudo apt install libeigen3-dev 
    $ sudo apt-get install libsfml-dev

and then edit your path appropriately.
