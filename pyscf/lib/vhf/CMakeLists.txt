# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0.
add_library(cvhf SHARED
  fill_nr_s8.c nr_incore.c nr_direct.c optimizer.c nr_direct_dot.c
  time_rev.c r_direct_o1.c rkb_screen.c
  r_direct_dot.c rah_direct_dot.c rha_direct_dot.c
  hessian_screen.c nr_sgx_direct.c nr_sr_vhf.c nr_s8_precheck.c eri_cache.c)
add_dependencies(cvhf cgto np_helper)
target_compile_features(cvhf PRIVATE c_std_11)
# Rename only the direct driver, not the screening/contraction routines.
set_property(SOURCE nr_direct.c APPEND PROPERTY COMPILE_DEFINITIONS
  CVHFnr_direct_drv=CVHFnr_direct_drv_uncached
  CVHFnrs8_prescreen_block=CVHFnrs8_precheck_block
  CVHFnrs8_vj_prescreen_block=CVHFnrs8_vj_precheck_block
  CVHFnrs8_vk_prescreen_block=CVHFnrs8_vk_precheck_block)
set_target_properties(cvhf PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(cvhf cgto np_helper cint ${BLAS_LIBRARIES} ${OPENMP_C_PROPERTIES})
