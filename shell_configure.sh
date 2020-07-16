#!/bin/bash

#module load hdf5
#module swap gnu8 intel
#module swap impi mvapich2


#python configure.py --prob disk --coor cylindrical --eos adiabatic --flux hllc --ndustfluids=1 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

python configure.py --prob collision_multipledust --ndustfluids=1 --nghost=3 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob shock_multipledust --ndustfluids=3 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob disk --coor cylindrical --eos adiabatic --flux hllc --ndustfluids=3 -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi


#python configure.py --prob shock_tube --ndustfluids=0 --eos general/hydrogen -mpi -debug -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob kh --ndustfluids=1 --nghost=2 -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi -debug

#python configure.py -shear --prob=ssheet --flux=hlle --eos=isothermal --ndustfluids=1 -mpi -hdf5 --hdf5_path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob=visc_dustfluids -mpi --ndustfluids=0 --coord=cylindrical

make clean
make -j 10
