cmake_minimum_required(VERSION 2.8)

project(poissonblend)

if (WIN32)
	#set(OpenCV_STATIC true)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS")
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} "inc")


# Library

add_library(blend 
	inc/blend/poisson_blend.h
	src/poisson_blend.cpp
)
	
target_link_libraries(blend ${OpenCV_LIBRARIES})
	
# Samples

add_executable(inpaint_image_criminisi examples/inpaint_image_criminisi.cpp)
target_link_libraries(inpaint_image_criminisi inpaint ${OpenCV_LIBRARIES})

add_executable(patch_match examples/patch_match.cpp)
target_link_libraries(patch_match inpaint ${OpenCV_LIBRARIES})

# Tests

include_directories("tests")
add_executable(inpaint_tests 
	tests/catch.hpp
	tests/random_image.h
	tests/gradient.cpp
	tests/patch.cpp
    tests/integral.cpp
	tests/criminisi_inpainter.cpp
    tests/template_match_candidates.cpp
	tests/patch_match.cpp
)
target_link_libraries (inpaint_tests inpaint ${OpenCV_LIBRARIES})

# Benchmarks

include_directories("benchmarks")
add_executable(inpaint_benchmarks 
	benchmarks/catch.hpp
	benchmarks/patch.cpp	
)
target_link_libraries (inpaint_benchmarks inpaint ${OpenCV_LIBRARIES})