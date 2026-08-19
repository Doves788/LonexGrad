#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "lonexgrad::lonexgrad" for configuration ""
set_property(TARGET lonexgrad::lonexgrad APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(lonexgrad::lonexgrad PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/liblonexgrad.a"
  )

list(APPEND _cmake_import_check_targets lonexgrad::lonexgrad )
list(APPEND _cmake_import_check_files_for_lonexgrad::lonexgrad "${_IMPORT_PREFIX}/lib/liblonexgrad.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
