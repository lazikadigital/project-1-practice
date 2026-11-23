cmake_minimum_required(VERSION 3.16)
project(mass_spring_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_options(/W4)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

add_library(mass_spring_lib
  mass_spring.cpp
  utilities.cpp
)

target_include_directories(mass_spring_lib
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

add_executable(mass_spring_demo
  main.cpp
)

target_link_libraries(mass_spring_demo
  PRIVATE mass_spring_lib
)

# Build from cpp/:
# cd cpp   // Changes into the dedicated directory
# cmake -S . -B build   // Generates Build folder
# cmake --build build   // Compiles / re compiles the code
# ./build/Debug/mass_spring_demo.exe // Runs the executable


# cmake -S . -B build  // Generates Build folder
# cmake --build build // Compiles the code
# ./build/Debug/mass_spring_demo.exe // Runs the executable 
