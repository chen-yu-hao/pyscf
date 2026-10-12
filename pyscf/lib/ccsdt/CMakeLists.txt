# Copyright 2014-2025 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

add_library(ccsdt SHARED rccsdt.c uccsdt.c uccsdt_t3.c rccsdtq.c uccsdt_q.c uccsdtq.c uccsdtq_blocked.c)
set_target_properties(ccsdt PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(ccsdt ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES} ${CMAKE_DL_LIBS})

option(CCSDT_USE_NATIVE_ARCH
       "Enable -march=native for the local libccsdt build" OFF)
if(CCSDT_USE_NATIVE_ARCH AND CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(ccsdt PRIVATE -march=native)
endif()
