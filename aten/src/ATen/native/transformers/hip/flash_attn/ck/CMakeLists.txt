# generate a list of kernels, but not actually emit files at config stage

# The CK tile flash-attn generator only supports a subset of architectures.
# Keep this list aligned with example/ck_tile/01_fmha/generate.py.
set(CK_TILE_FA_SUPPORTED_ARCHS gfx90a gfx942 gfx950 gfx1200 gfx1201)
set(CK_TILE_TARGET_ARGS "")

if(PYTORCH_ROCM_ARCH)
  set(CK_TILE_TARGET_LIST "")
  foreach(arch ${PYTORCH_ROCM_ARCH})
    if(arch IN_LIST CK_TILE_FA_SUPPORTED_ARCHS)
      list(APPEND CK_TILE_TARGET_LIST ${arch})
    else()
      message(WARNING "Skipping CK SDPA generation for unsupported arch '${arch}'")
    endif()
  endforeach()
  if(NOT CK_TILE_TARGET_LIST)
    message(FATAL_ERROR
      "No ROCm architectures remain after filtering with CK_TILE_FA_SUPPORTED_ARCHS; "
      "flash attention CK generation needs one of: ${CK_TILE_FA_SUPPORTED_ARCHS}")
  endif()
  list(REMOVE_DUPLICATES CK_TILE_TARGET_LIST)
  list(JOIN CK_TILE_TARGET_LIST "," CK_TILE_TARGETS)
  set(CK_TILE_TARGET_ARGS --targets ${CK_TILE_TARGETS})
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd --receipt 4 ${CK_TILE_TARGET_ARGS} --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd_splitkv --receipt 4 ${CK_TILE_TARGET_ARGS} --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_splitkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_SPLITKV kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api fwd_appendkv --receipt 4 ${CK_TILE_TARGET_ARGS} --list_blobs ${CMAKE_CURRENT_LIST_DIR}/fwd_appendkv_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of FWD_APPENDKV kernels via Python.")
endif()

execute_process(
  COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py
  --api bwd --receipt 4 ${CK_TILE_TARGET_ARGS} --list_blobs ${CMAKE_CURRENT_LIST_DIR}/bwd_blob_list.txt
  RESULT_VARIABLE ret
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate a list of BWD kernels via Python.")
endif()

# Generate the files for both fwd, fwd_splitkv, fwd_appendkv, and bwd
execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd --receipt 4 ${CK_TILE_TARGET_ARGS} --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
  message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd_splitkv --receipt 4 ${CK_TILE_TARGET_ARGS} --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_SPLITKV kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api fwd_appendkv --receipt 4 ${CK_TILE_TARGET_ARGS} --output_dir ${CMAKE_CURRENT_LIST_DIR}
)

if(ret AND NOT ret EQUAL 0)
    message( FATAL_ERROR "CK Tile FMHA FAILED to generate FWD_APPENDKV kernels.")
endif()

execute_process(COMMAND python3 ${CMAKE_SOURCE_DIR}/third_party/composable_kernel/example/ck_tile/01_fmha/generate.py --api bwd --receipt 4 ${CK_TILE_TARGET_ARGS} --output_dir ${CMAKE_CURRENT_LIST_DIR}
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
