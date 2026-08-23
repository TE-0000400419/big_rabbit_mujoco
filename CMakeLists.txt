cmake_minimum_required(VERSION 3.16)
project(big_rabbit_mujoco LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Big Rabbit は位置制御ループをドライバが持つので、rabbit_mujoco_2 が必要としていた
# SDRDtk / Controllers / PosDOB（加速度制御と外乱オブザーバ）に依存しない。
# パイプライン方式（Invoke で駆動）は維持しつつ、依存は MuJoCo と GLFW だけ。
set(COMMON_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/dep/motion_control/include
)

# CONFIGURE_DEPENDS を付けておくと、ファイルを足したときに cmake の再実行を忘れても
# ビルド時に glob が再評価される（付けないとリンクエラーになって原因が分かりにくい）。
file(GLOB_RECURSE BIG_RABBIT_SIM_SRC_FILES CONFIGURE_DEPENDS "src/*.cc")
file(GLOB_RECURSE BIG_RABBIT_BRIDGE_SRC_FILES CONFIGURE_DEPENDS "dep/motion_control/src/*.cc")

function(resolve_mujoco_dir out_var)
    if(MUJOCO_DIR)
        set(_candidate "${MUJOCO_DIR}")
    else()
        find_package(Python3 COMPONENTS Interpreter REQUIRED)
        execute_process(
            COMMAND "${Python3_EXECUTABLE}" -c "import pathlib, mujoco; print(pathlib.Path(mujoco.__file__).resolve().parent)"
            RESULT_VARIABLE _mujoco_status
            OUTPUT_VARIABLE _mujoco_output
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        if(_mujoco_status EQUAL 0)
            set(_candidate "${_mujoco_output}")
        endif()
    endif()

    if(_candidate AND EXISTS "${_candidate}/include/mujoco/mujoco.h")
        set(${out_var} "${_candidate}" PARENT_SCOPE)
    else()
        set(${out_var} "" PARENT_SCOPE)
    endif()
endfunction()

# ---- 関節 <-> モータ変換の単体テスト（MuJoCo 不要）----
add_executable(test_joint_motor_transform
    dep/motion_control/test/test_joint_motor_transform.cc
)
target_include_directories(test_joint_motor_transform PRIVATE ${COMMON_INCLUDE_DIRS})

# ---- 観測 / 参照 / 推論を単体で叩くツール（MuJoCo 不要。クロス検証で使う）----
add_executable(dump_policy_io
    tools/dump_policy_io.cc
    ${BIG_RABBIT_BRIDGE_SRC_FILES}
)
target_include_directories(dump_policy_io PRIVATE ${COMMON_INCLUDE_DIRS})

# ---- MuJoCo 実行本体 ----
add_executable(big_rabbit_mujoco_sim
    ${BIG_RABBIT_SIM_SRC_FILES}
    ${BIG_RABBIT_BRIDGE_SRC_FILES}
)
target_compile_definitions(big_rabbit_mujoco_sim PRIVATE
    "BIG_RABBIT_SCENE_XML_DEFAULT=\"${CMAKE_SOURCE_DIR}/robotmodel/big_rabbit/scene.xml\""
)
target_include_directories(big_rabbit_mujoco_sim PRIVATE ${COMMON_INCLUDE_DIRS})

resolve_mujoco_dir(BIG_RABBIT_MUJOCO_DIR)
if(NOT BIG_RABBIT_MUJOCO_DIR)
    message(FATAL_ERROR
        "MuJoCo が見つからない。MUJOCO_DIR を明示するか 'pip install mujoco' を入れる。"
    )
endif()

find_library(BIG_RABBIT_MUJOCO_LIB NAMES mujoco PATHS ${BIG_RABBIT_MUJOCO_DIR} NO_DEFAULT_PATH)
if(NOT BIG_RABBIT_MUJOCO_LIB)
    file(GLOB BIG_RABBIT_MUJOCO_LIB_CANDIDATES
        "${BIG_RABBIT_MUJOCO_DIR}/libmujoco.so*"
        "${BIG_RABBIT_MUJOCO_DIR}/libmujoco.dylib"
        "${BIG_RABBIT_MUJOCO_DIR}/libmujoco.a"
    )
    list(LENGTH BIG_RABBIT_MUJOCO_LIB_CANDIDATES _count)
    if(_count GREATER 0)
        list(GET BIG_RABBIT_MUJOCO_LIB_CANDIDATES 0 BIG_RABBIT_MUJOCO_LIB)
    endif()
endif()
if(NOT BIG_RABBIT_MUJOCO_LIB)
    message(FATAL_ERROR "MuJoCo ライブラリが ${BIG_RABBIT_MUJOCO_DIR} に無い。")
endif()

target_include_directories(big_rabbit_mujoco_sim PRIVATE ${BIG_RABBIT_MUJOCO_DIR}/include)

if(UNIX AND NOT APPLE)
    set_target_properties(big_rabbit_mujoco_sim PROPERTIES
        BUILD_RPATH "${BIG_RABBIT_MUJOCO_DIR}"
        INSTALL_RPATH "${BIG_RABBIT_MUJOCO_DIR}"
    )
endif()

find_package(OpenGL REQUIRED)
find_package(glfw3 QUIET)
if(TARGET glfw)
    set(BIG_RABBIT_GLFW_LIB glfw)
elseif(TARGET glfw3)
    set(BIG_RABBIT_GLFW_LIB glfw3)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GLFW3 REQUIRED glfw3)
    target_include_directories(big_rabbit_mujoco_sim PRIVATE ${GLFW3_INCLUDE_DIRS})
    target_link_directories(big_rabbit_mujoco_sim PRIVATE ${GLFW3_LIBRARY_DIRS})
    set(BIG_RABBIT_GLFW_LIB ${GLFW3_LIBRARIES})
endif()

target_link_libraries(big_rabbit_mujoco_sim PRIVATE
    ${BIG_RABBIT_MUJOCO_LIB}
    ${BIG_RABBIT_GLFW_LIB}
    OpenGL::GL
)
