cmake_minimum_required(VERSION 2.8)
project( GT_Evaluation )
set("OpenCV_DIR" "/Users/ylvaselling/Documents/TSBB15\ Computer\ Vision/Projekt/OpenCV/build_opencv")
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

file(GLOB_RECURSE sources src/*.cpp include/*.hpp)
add_executable( GT_Evaluation ${sources} )
target_link_libraries( GT_Evaluation ${OpenCV_LIBS} )
