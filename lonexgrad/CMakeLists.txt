file(GLOB_RECURSE HEADER_LIST CONFIGURE_DEPENDS "${lonexgrad_SOURCE_DIR}/include/lonexgrad/*.h")

target_sources(lonexgrad PRIVATE ${HEADER_LIST})

target_include_directories(lonexgrad PRIVATE ${PROJECT_SOURCE_DIR}/lonexgrad)
target_include_directories(lonexgrad SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/external/libnop/include)


add_subdirectory(tensor)
add_subdirectory(common)
add_subdirectory(autograd)
add_subdirectory(nn)
add_subdirectory(optim)
add_subdirectory(data)

