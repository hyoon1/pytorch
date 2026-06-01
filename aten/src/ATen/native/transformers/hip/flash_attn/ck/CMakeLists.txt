set(CK_SDPA_FMHA_TARGETS ${PYTORCH_ROCM_ARCH})
list(FILTER CK_SDPA_FMHA_TARGETS INCLUDE REGEX "^gfx(9|1[12])")
list(REMOVE_DUPLICATES CK_SDPA_FMHA_TARGETS)

if(NOT CK_SDPA_FMHA_TARGETS)
  message(FATAL_ERROR "CK Tile FMHA requires at least one gfx9/gfx11/gfx12 target in PYTORCH_ROCM_ARCH: ${PYTORCH_ROCM_ARCH}")
endif()

list(JOIN CK_SDPA_FMHA_TARGETS "," CK_SDPA_FMHA_TARGETS_ARG)
message(STATUS "Generating CK Tile FMHA kernels for targets: ${CK_SDPA_FMHA_TARGETS_ARG}")

# Generated blobs live in this source directory in PyTorch's current build flow.
# Remove stale generated instances so changing PYTORCH_ROCM_ARCH does not leave
# old arch-specific blobs in ck_sdpa_sources_hip.
file(GLOB CK_SDPA_STALE_GEN_BLOBS
  "${CMAKE_CURRENT_LIST_DIR}/fmha_*.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/fmha_*.hip")
if(CK_SDPA_STALE_GEN_BLOBS)
  file(REMOVE ${CK_SDPA_STALE_GEN_BLOBS})
endif()

# generate a list of kernels, but not actually emit files at config stage
execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api fwd --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api fwd_splitkv --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_SPLITKV kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api fwd_appendkv --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_APPENDKV kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api bwd --receipt 4 --list_blobs ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# Generate the files for both fwd, fwd_splitkv, fwd_appendkv, and bwd
execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api fwd --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api fwd_splitkv --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_SPLITKV kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api fwd_appendkv --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_APPENDKV kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --targets ${CK_SDPA_FMHA_TARGETS_ARG} --api bwd --receipt 4 --output_dir ${CMAKE_CURRENT_LIST_DIR}
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate BWD kernels.")
endif()

# Change make_kernel to make_kernel_pt for fwd
execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd pass")
endif()

execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd_splitkv pass")
endif()

execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the fwd appendkv pass")
endif()

# Change make_kernel to make_kernel_pt for bwd
execute_process(
  COMMAND bash -c "${CMAKE_CURRENT_LIST_DIR}/add_make_kernel_pt.sh ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt"
  RESULT_VARIABLE ret)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change make_kernel to make_kernel_pt for the bwd pass")
endif()

# Change file extensions to .hip
execute_process(COMMAND bash -c "for file in ${CMAKE_CURRENT_LIST_DIR}/*.cpp; do mv -- \"$file\" \"\${file%.cpp}.hip\"; done"
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to change the generated instances extensions from .cpp to .hpp")
endif()
