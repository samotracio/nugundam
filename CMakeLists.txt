cmake_minimum_required(VERSION 3.18...4.3)
project(${SKBUILD_PROJECT_NAME} VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C Fortran)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
# nuGUNDAM is performance-oriented and the distributed Linux builds require OpenMP.
find_package(OpenMP REQUIRED COMPONENTS Fortran)

set(module_name _cflibfor)
set(PYF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/nugundam/cflibfor.pyf")
set(FORTRAN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/nugundam/cflibfor.f90")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "import numpy.f2py; print(numpy.f2py.get_include())"
  RESULT_VARIABLE F2PY_INCLUDE_RESULT
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  ERROR_VARIABLE F2PY_INCLUDE_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT F2PY_INCLUDE_RESULT EQUAL 0 OR NOT IS_DIRECTORY "${F2PY_INCLUDE_DIR}")
  message(FATAL_ERROR
    "Could not locate the NumPy F2PY include directory: ${F2PY_INCLUDE_ERROR}")
endif()

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")

# fortranobject.c includes both Python.h and the NumPy C headers.
# With CMake 4.2+ (CMP0201 NEW), Python::NumPy no longer
# automatically depends on Python::Module.
target_link_libraries(fortranobject PUBLIC
  Python::Module
  Python::NumPy
)

target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

set(F2PY_C_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/${module_name}module.c")
set(F2PY_F90_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/${module_name}-f2pywrappers2.f90")

add_custom_command(
  OUTPUT "${F2PY_C_WRAPPER}" "${F2PY_F90_WRAPPER}"
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py "${PYF_FILE}"
  DEPENDS "${PYF_FILE}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "Generating F2PY wrappers for ${module_name}"
  VERBATIM
)

python_add_library(${module_name} MODULE
  "${F2PY_C_WRAPPER}"
  "${F2PY_F90_WRAPPER}"
  "${FORTRAN_SRC}"
  WITH_SOABI
)

target_link_libraries(${module_name} PRIVATE
  fortranobject
  OpenMP::OpenMP_Fortran
)

# th_A_grp/th_C_grp must give exactly the same counts as th_A/th_C. A fused
# multiply-add rounds differently and could move a pair lying on a bin edge, so
# contraction is disabled, also when FFLAGS adds -march=native. This changes
# nothing on x86-64 with default flags.
if(CMAKE_Fortran_COMPILER_ID STREQUAL "GNU")
  target_compile_options(${module_name} PRIVATE
    $<$<COMPILE_LANGUAGE:Fortran>:-ffp-contract=off>)
endif()

install(TARGETS ${module_name} DESTINATION nugundam)
