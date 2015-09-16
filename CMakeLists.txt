cmake_minimum_required(VERSION 2.8)

project(poissonblend)

if (WIN32)
	#set(OpenCV_STATIC true)
	add_definitions("-D_SCL_SECURE_NO_WARNINGS")
endif()

set(EIGEN_INCLUDE_DIR "../eigen" CACHE PATH "Where is the include directory of Eigen located")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR} "inc")


# Library

add_library(blend 
	inc/blend/poisson_blend.h
	src/poisson_blend.cpp
)
	
target_link_libraries(blend ${OpenCV_LIBRARIES})
	
# Samples

add_executable(seamless_cloning examples/seamless_cloning.cpp)
target_link_libraries(seamless_cloning blend ${OpenCV_LIBRARIES})