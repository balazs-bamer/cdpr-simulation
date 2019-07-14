add_definitions(-std=c++11 -w -O3)


include("../../../../../cmake/tensorflowDependencies/tensorflow.cmake")

add_executable(testDeepDeterministicPolicyGradient testDeepDeterministicPolicyGradient.cpp)
use_tensorflow(testDeepDeterministicPolicyGradient)
target_link_libraries(testDeepDeterministicPolicyGradient graph boost_system boost_filesystem rai_logger glog)

file(COPY resources DESTINATION ${CMAKE_BINARY_DIR}/RAI/include/algorithm/tests/)
file(COPY ../../../../src/tasks/poleBalancing/visualizer DESTINATION ${CMAKE_BINARY_DIR}/RAI/include/algorithm/tests)
