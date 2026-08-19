target_sources(lonexgrad PRIVATE
    lonex_tensor.cpp
    lonex_backend_base.h
    lonex_backend_register.cpp
    lonex_backend_register.h
    lonex_random.cpp
    lonex_index.cpp
    lonex_print.h
    lonex_shape.cpp
    lonex_storage_base.h
)

# Add backends
add_subdirectory(backend/common)
add_subdirectory(backend/cpu)
if(${LONEXGRAD_BUILD_CUDA_AND_FOUND})
    add_subdirectory(backend/cuda) 
    # If building with CUDA support, some common files need to be built in CUDA 
    #  mode instead to ensure linker can find the host/device methods
    set_source_files_properties(lonex_random.cpp TARGET_DIRECTORY lonexgrad PROPERTIES LANGUAGE CUDA) 
endif()

