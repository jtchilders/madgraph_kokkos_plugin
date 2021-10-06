#ifndef MGKOKKOSTYPES_H
#define MGKOKKOSTYPES_H 1

#include "mgKokkosConfig.h"

namespace mgKokkos
{

#ifdef THRUST_COMPLEX
  #include <thrust/complex.h>
  template<typename T>
  using complex = thrust::complex<T>;
#else
  #include "Kokkos_Complex.hpp"
  template<typename T>
  using complex = Kokkos::complex<T>;
#endif

}


#ifdef __CUDACC__
#include <nvToolsExt.h> 
#else

inline void nvtxRangePush(const char* text){
  return;
}

inline void nvtxRangePop(void){
  return;
}



#endif // MGKOKKOSTYPES_H
