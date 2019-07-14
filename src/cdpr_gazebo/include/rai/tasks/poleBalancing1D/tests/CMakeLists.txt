add_executable(testPoleBalancing1D testPoleBalancing1D.cpp ${QLEARNING_TASKS_SRC})
use_tensorflow(testPoleBalancing1D)
target_link_libraries(testPoleBalancing1D graph glog boost_system boost_filesystem ${RAI_LINK} ${QLEARNING_TASKS_LIBS})