cmake_minimum_required(VERSION 3.2.0) 
MESSAGE(STATUS "Project: SVM") 
Project(MYSVM)
add_definitions(-std=c++11) 
add_definitions(-g -o2) 
#define cuda,opencv,cudnn 
ADD_DEFINITIONS( -DGPU -DCUDNN ) 
# use opencv 
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/share/OpenCV") 
find_package(OpenCV 3.2.0 REQUIRED) 
if(NOT OpenCV_FOUND) 
    message(WARNING "OpenCV not found!") 
else() 
    include_directories(${OpenCV_INCLUDE_DIRS}) 
endif() 
# headers 
include_directories(${PROJECT_SOURCE_DIR}/inc) 
#sources 
set(SRC 
${PROJECT_SOURCE_DIR}/src/main.cpp 
${PROJECT_SOURCE_DIR}/src/MYSVM.cpp
${PROJECT_SOURCE_DIR}/src/CutPic.cpp
) 

#lib link 
link_directories(${PROJECT_SOURCE_DIR}) 
#build so 
add_executable(MYSVM ${SRC})
target_link_libraries(MYSVM ${OpenCV_LIBS}) 
target_link_libraries(MYSVM -lpthread)
