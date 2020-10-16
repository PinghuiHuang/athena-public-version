#!/bin/bash

#python configure.py --prob disk --coor cylindrical --eos adiabatic --flux hllc --ndustfluids=0 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob collision_multipledust --ndustfluids=1 --nghost=2 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -h5double

#python configure.py --prob collision_multipledust --ndustfluids=5 --nghost=2 --cxx=g++ -debug -hdf5 --hdf5_path=/usr/local

#python configure.py --prob shock_multipledust --ndustfluids=3 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob disk --coor cylindrical --eos adiabatic --flux hllc --ndustfluids=3 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob shock_tube --ndustfluids=0 --eos general/hydrogen -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob kh --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -debug

#python configure.py -shear --prob=ssheet --flux=hlle --eos=isothermal --ndustfluids=1 -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob=visc_dustfluids -mpi --ndustfluids=1 -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -debug

#python configure.py --prob=hb3 -shear -b --eos=isothermal -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -debug

#python configure.py --prob=hb3_dustfluids -shear -b --eos=isothermal --ndustfluids=1 -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -debug

python configure.py -shear --prob=nsh_dust --ndustfluids=1 --eos=isothermal -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -debug

make clean
make -j 10
