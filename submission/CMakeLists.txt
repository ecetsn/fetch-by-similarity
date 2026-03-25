cmake_minimum_required(VERSION 3.23)
project(fetch_by_similarity_heongpu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)


find_package(CUDAToolkit REQUIRED)

set(HEONGPU_ROOT $ENV{HOME}/HEonGPU)
set(HEONGPU_BUILD ${HEONGPU_ROOT}/build)
add_subdirectory(${HEONGPU_ROOT} ${HEONGPU_BUILD})

find_package(OpenMP REQUIRED)


include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

set(SHARED_SRCS
    src/utils.cu
    src/slot_replication.cu
    src/running_sums.cu
)

# add_compile_definitions(DEBUG)

add_executable(client_key_generation 
    src/client_key_generation.cu
    ${SHARED_SRCS}
)

# Client DB Encoding and Encryption
add_executable(client_encode_encrypt_db 
    src/client_encode_encrypt_db.cu
    ${SHARED_SRCS}
)

# Client Query Encoding and Encryption
add_executable(client_encode_encrypt_query 
    src/client_encode_encrypt_query.cu 
    ${SHARED_SRCS}
)

# Server Encrypted Computation
add_executable(server_encrypted_compute
    src/server_encrypted_compute.cu
    ${SHARED_SRCS}
)

# Client Decrypt and Decode
add_executable(client_decrypt_decode
    src/client_decrypt_decode.cu
    ${SHARED_SRCS}
)

# Client Postprocess
add_executable(client_postprocess 
    src/client_postprocess.cu
    ${SHARED_SRCS}
)

# Preprocess (currently empty)
add_executable(client_preprocess_dataset src/client_preprocess_dataset.cu)
add_executable(client_preprocess_query src/client_preprocess_query.cu)
add_executable(server_preprocess_dataset src/server_preprocess_dataset.cu)

set(MY_EXECUTABLES
    client_key_generation
    client_encode_encrypt_db
    client_encode_encrypt_query
    server_encrypted_compute
    client_decrypt_decode
    client_postprocess
    client_preprocess_dataset
    client_preprocess_query
    server_preprocess_dataset
)

foreach(EXE ${MY_EXECUTABLES})
    target_link_libraries(${EXE} PRIVATE heongpu CUDA::cudart OpenMP::OpenMP_CXX)
endforeach()


