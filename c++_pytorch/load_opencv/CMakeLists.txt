cmake_minimum_required(VERSION 3.0.0)
project(load_opencv)

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable(load_opencv load_opencv.cpp)
#与opencv库建立连接
target_link_libraries(load_opencv ${OpenCV_LIBS} )