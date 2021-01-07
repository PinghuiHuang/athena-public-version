hell_configure.sh/bin/bash

hdf5path=/usr/lib/x86_64-linux-gnu/hdf5/openmpi

#python configure.py --prob disk --coor=cylindrical --eos=adiabatic --flux=hllc --ndustfluids=0 -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob dusty_soundwave --ndustfluids=4 --eos=isothermal --flux=roe -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob linear_wave --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob disk_dustdrift --coor=cylindrical --ndustfluids=1 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=disk_original --coor=spherical_polar --ndustfluids=0 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=dust_collision --ndustfluids=1 --nghost=2 -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=dusty_shock --ndustfluids=3 --eos=isothermal --flux=roe -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=disk --coor cylindrical --eos adiabatic --flux=hllc --ndustfluids=3 -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=shock_tube --ndustfluids=0 --eos general/hydrogen -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=dusty_kh --ndustfluids=1 --nghost=2 -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py -shear --prob=ssheet --flux=hlle --eos=isothermal --ndustfluids=1 -mpi -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=dust_diffusion -mpi -debug --ndustfluids=1 -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py --prob=hb3 -shear -b --eos=isothermal -mpi -debug -hdf5 --hdf5_path=${hdf5path} -h5double

#python configure.py -shear --prob=nsh_dust --ndustfluids=1 --eos=isothermal --flux=roe -mpi -hdf5 --hdf5_path=${hdf5path} -debug -h5double

python configure.py -shear --prob=dust_streaming --ndustfluids=1 --eos=isothermal --flux=roe -mpi -hdf5 --hdf5_path=${hdf5path} -debug -h5double

make clean
make -j 16
