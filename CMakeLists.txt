cmake_minimum_required(VERSION 3.17)
project(CustomOp CXX)
find_package(CUDA 9.0 REQUIRED)
find_package(TIFF 4.1 REQUIRED)

set(CMAKE_CXX_STANDARD 14)

# TODO get from the environment
set(PYTHON_CMD "/home/ice/.virtualenvs/deep/bin/python")

execute_process(
        COMMAND ${PYTHON_CMD} -c "import nvidia.dali as dali; print(dali.sysconfig.get_lib_dir())"
        OUTPUT_VARIABLE DALI_LIB_DIR)
string(STRIP "${DALI_LIB_DIR}" DALI_LIB_DIR)

execute_process(
        COMMAND ${PYTHON_CMD} -c "import nvidia.dali as dali; print(\" \".join(dali.sysconfig.get_compile_flags()))"
        OUTPUT_VARIABLE DALI_COMPILE_FLAGS)
string(STRIP "${DALI_COMPILE_FLAGS}" DALI_COMPILE_FLAGS)

message("Got >${DALI_LIB_DIR}< >${DALI_COMPILE_FLAGS}<")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DALI_COMPILE_FLAGS}")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${DALI_COMPILE_FLAGS} ")
link_directories(CustomOp "${DALI_LIB_DIR}")

cuda_add_library(CustomOp SHARED file_reader.cc file_file_loader.cc tiff_decoder.cc tiff_libtiff.cc)
target_link_libraries(CustomOp dali dali_operators TIFF::TIFF)

