add_executable(testPBgenerators testPBgenerators_Policy.cpp)
target_link_libraries(testPBgenerators ${RAI_LINK})

add_executable(testPBgeneratorsQ testPBgenerators_Q.cpp)
target_link_libraries(testPBgeneratorsQ ${RAI_LINK})

add_executable(testPBgeneratorsV testPBgenerators_V.cpp)
target_link_libraries(testPBgeneratorsV ${RAI_LINK})

add_executable(testPBgeneratorsSP testPBgenerators_Stochpol.cpp)
target_link_libraries(testPBgeneratorsSP ${RAI_LINK})

add_executable(testPBgeneratorGPU testPBgenerators_multiGPU.cpp)
target_link_libraries(testPBgeneratorGPU ${RAI_LINK})
