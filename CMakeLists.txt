cmake_minimum_required(VERSION 3.16)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)
find_package(OpenMP)

set(module_name _cflibfor)
set(PYF_FILE "${CMAKE_CURRENT_SOURCE_DIR}/src/nugundam/cflibfor.pyf")
set(FORTRAN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/nugundam/cflibfor.f90")

execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import numpy.f2py; print(numpy.f2py.get_include())"
  OUTPUT_VARIABLE F2PY_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_library(fortranobject OBJECT "${F2PY_INCLUDE_DIR}/fortranobject.c")
target_link_libraries(fortranobject PUBLIC Python::NumPy)
target_include_directories(fortranobject PUBLIC "${F2PY_INCLUDE_DIR}")
set_property(TARGET fortranobject PROPERTY POSITION_INDEPENDENT_CODE ON)

set(F2PY_C_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/${module_name}module.c")
set(F2PY_F90_WRAPPER "${CMAKE_CURRENT_BINARY_DIR}/${module_name}-f2pywrappers2.f90")

add_custom_command(
  OUTPUT "${F2PY_C_WRAPPER}" "${F2PY_F90_WRAPPER}"
  COMMAND "${Python_EXECUTABLE}" -m numpy.f2py "${PYF_FILE}"
  DEPENDS "${PYF_FILE}"
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  VERBATIM
)

python_add_library(${module_name} MODULE
  "${F2PY_C_WRAPPER}"
  "${F2PY_F90_WRAPPER}"
  "${FORTRAN_SRC}"
  WITH_SOABI
)

target_link_libraries(${module_name} PRIVATE fortranobject)
if (OpenMP_Fortran_FOUND)
  target_link_libraries(${module_name} PRIVATE OpenMP::OpenMP_Fortran)
endif()

install(TARGETS ${module_name} DESTINATION nugundam)
