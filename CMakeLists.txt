cmake_minimum_required(VERSION 3.13)
project(MKL_Eigs_for_Sparse_Matrices)

set(CMAKE_CXX_STANDARD 14)
set(LOGGER ../Singleton-Logger/sln/Sources)
message(WARNING "mkl root is: " $ENV{MKLROOT})

include_directories($ENV{MKLROOT}/include ${LOGGER})

add_executable(MKL_Eigs_for_Sparse_Matrices main.cpp ${LOGGER}/SingletonLogger.cpp)

# MKL: https://software.intel.com/en-us/articles/intel-mkl-link-line-advisor
# g++ -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_rt -lpthread -lm -ldl -m64 -I${MKLROOT}/include main.cpp
# setting env vars: https://askubuntu.com/a/849954/842444 + https://stackoverflow.com/a/41983223/6712873
add_compile_options(-m64 -I$ENV{MKLROOT}/include)

# static
target_link_libraries(MKL_Eigs_for_Sparse_Matrices "-L$ENV{MKLROOT}/lib/intel64/" -Wl,--start-group libmkl_intel_lp64.a libmkl_gnu_thread.a libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl)

# dynamic
#target_link_libraries(MKL_Eigs_for_Sparse_Matrices "-L$ENV{MKLROOT}/lib/intel64/" -Wl,--no-as-needed -lmkl_rt)
#target_link_libraries(MKL_Eigs_for_Sparse_Matrices -lpthread -lm -ldl)