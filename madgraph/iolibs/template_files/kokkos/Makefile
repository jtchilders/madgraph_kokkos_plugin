KOKKOSPATH_CUDA ?= /home/jchilders/git/kokkos/install_v100
KOKKOSPATH_OMP ?= /home/jchilders/git/kokkos/install_omp
KOKKOSPATH_INTEL ?= $(KOKKOS_HOME)
KOKKOSPATH_HIP ?= /home/jchilders/git/kokkos/install_mi50

MODELLIB = model_%(model)s

INCDIR=../../src
CXXFLAGS=-O3 -ffast-math -I$(INCDIR) --std=c++17 -lineinfo
LDFLAGS=

CUDA_ARCH_NUM ?= 70
NVCC=$(KOKKOSPATH_CUDA)/bin/nvcc_wrapper
CUDA_CXXFLAGS=$(CXXFLAGS) -I$(KOKKOSPATH_CUDA)/include -arch=compute_$(CUDA_ARCH_NUM) --expt-extended-lambda --expt-relaxed-constexpr -use_fast_math --openmp
CUDA_LDFLAGS=$(LDFLAGS) $(KOKKOSPATH_CUDA)/lib64/libkokkoscore.a -lnvToolsExt --openmp

CXX=g++
IPCX=icpx
CLANG=clang++
HIPCC=hipcc

OPENMP_CXXFLAGS=$(CXXFLAGS) -I$(KOKKOSPATH_OMP)/include --openmp
OPENMP_LDFLAGS=$(LDFLAGS) $(KOKKOSPATH_OMP)/lib64/libkokkoscore.a -ldl --openmp

# INTEL_CXXFLAGS=$(CXXFLAGS) -I$(KOKKOSPATH_INTEL)/include -I/soft/restricted/CNDA/sdk/2021.04.30.001/oneapi/compiler/latest/linux/include/sycl -fiopenmp -fopenmp-targets=spir64 -Wno-parentheses -Wno-openmp-mapping
# INTEL_LDFLAGS=$(LDFLAGS)  $(KOKKOSPATH_INTEL)/lib64/libkokkoscore.a -fiopenmp -fopenmp-targets=spir64 -L/soft/restricted/CNDA/sdk/2021.04.30.001/oneapi/compiler/latest/linux/lib/ -lsycl

INTEL_CXXFLAGS=$(CXXFLAGS) -I$(KOKKOSPATH_INTEL)/include -fiopenmp -fopenmp-targets=spir64 -Wno-parentheses -Wno-openmp-mapping
INTEL_LDFLAGS=$(LDFLAGS)  $(KOKKOSPATH_INTEL)/lib64/libkokkoscore.a -fiopenmp -fopenmp-targets=spir64 

HIP_CXXFLAGS=$(CXXFLAGS) -I$(KOKKOSPATH_HIP)/include -fopenmp -fno-gpu-rdc --amdgpu-target=gfx906
HIP_LDFLAGS=$(LDFLAGS) -L $(KOKKOSPATH_HIP)/lib64  -lkokkoscore -ldl -fopenmp 

cuda_exe=ccheck.exe
cuda_obj=ccheck.o
openmp_exe=ocheck.exe
openmp_obj=ocheck.o
intel_exe=icheck.exe
intel_obj=icheck.o
hip_exe=hcheck.exe
hip_obj=hcheck.o
libmodel=../../lib/lib${MODELLIB}.a
intel_libmodel=../../lib/lib${MODELLIB}_intel.a
hip_libmodel=../../lib/lib${MODELLIB}_hip.a

all: cuda openmp intel hip


cuda:
	make -C ../../src cuda
	make $(cuda_exe)
openmp: 
	make -C ../../src openmp
	make $(openmp_exe)
intel:
	make -C ../../src intel
	make $(intel_exe)
hip:
	make -C ../../src hip
	make $(hip_exe)

$(cuda_obj): check.cpp ../../src/rambo.h ../../src/random_generator.h CPPProcess.h
	$(NVCC) -c $< -o $@ $(CUDA_CXXFLAGS) $(CUDA_COMPLEX)

$(cuda_exe): $(cuda_obj) $(libmodel)
	$(NVCC)  $(cuda_obj) -o $@ $(CUDA_LDFLAGS) $(libmodel)

$(openmp_obj): check.cpp ../../src/rambo.h ../../src/random_generator.h CPPProcess.h
	$(CXX) -c $< -o $@ $(OPENMP_CXXFLAGS)

$(openmp_exe): $(openmp_obj) $(libmodel)
	$(CXX) $(openmp_obj) -o $@ $(OPENMP_LDFLAGS) $(libmodel)

$(intel_obj): check.cpp ../../src/rambo.h ../../src/random_generator.h CPPProcess.h
	$(IPCX) -c $< -o $@ $(INTEL_CXXFLAGS)

$(intel_exe): $(intel_obj) $(intel_libmodel)
	$(IPCX) $(intel_obj) -o $@ $(INTEL_LDFLAGS) $(intel_libmodel)

$(hip_obj): check.cpp ../../src/rambo.h ../../src/random_generator.h CPPProcess.h
	$(HIPCC) -c $< -o $@ $(HIP_CXXFLAGS)

$(hip_exe): $(hip_obj) $(hip_libmodel)
	$(HIPCC) $(hip_obj) -o $@ $(HIP_LDFLAGS) $(hip_libmodel)

.PHONY: clean

clean:
	make -C ../../src clean
	rm -f $(cuda_obj) $(cuda_exe) $(openmp_obj) $(openmp_exe) $(intel_exe) $(intel_obj) $(hip_exe) $(hip_obj)

