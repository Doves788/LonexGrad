include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if (NOT DEFINED lonexgrad_INSTALL_CMAKEDIR)
   set(lonexgrad_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/lonexgrad"
       CACHE STRING "Path to lonexgrad CMake files")
endif ()

install(TARGETS lonexgrad EXPORT lonexgrad_Targets
        RUNTIME COMPONENT lonexgrad_Runtime
        LIBRARY COMPONENT lonexgrad_Runtime
        NAMELINK_COMPONENT lonexgrad_Development
        ARCHIVE COMPONENT lonexgrad_Development
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(DIRECTORY "${lonexgrad_SOURCE_DIR}/include/" "${lonexgrad_BINARY_DIR}/include/"
        TYPE INCLUDE
        COMPONENT lonexgrad_Development)

if (BUILD_SHARED_LIBS)
    set(type shared)
    message(STATUS "Building lonexgrad shared lib")
else ()
    set(type static)
    message(STATUS "Building lonexgrad static lib")
endif ()

install(EXPORT lonexgrad_Targets
        DESTINATION "${lonexgrad_INSTALL_CMAKEDIR}"
        NAMESPACE lonexgrad::
        FILE lonexgrad-${type}-targets.cmake
        COMPONENT lonexgrad_Development)

write_basic_package_version_file(
    lonexgradConfigVersion.cmake
    COMPATIBILITY SameMajorVersion)

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/lonexgradConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/lonexgradConfigVersion.cmake"
        DESTINATION "${lonexgrad_INSTALL_CMAKEDIR}"
        COMPONENT lonexgrad_Development)
