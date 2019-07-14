//
// Created by jhwangbo on 07/08/17.
//

#ifndef RAI_PERFORMANCETESTER_HPP
#define RAI_PERFORMANCETESTER_HPP
#include <rai/memory/Trajectory.hpp>
#include <rai/tasks/common/Task.hpp>
#include <rai/noiseModel/NormalDistributionNoise.hpp>
#include <rai/experienceAcquisitor/TrajectoryAcquisitor_Parallel.hpp>
#include "rai/RAI_core"

namespace rai {

namespace Algorithm {
template<typename Dtype, int StateDim, int ActionDim>
class PerformanceTester {


  typedef Eigen::Matrix<Dtype, StateDim, 1> State;
  typedef Eigen::Matrix<Dtype, StateDim, Eigen::Dynamic> StateBatch;
  typedef Eigen::Matrix<Dtype, ActionDim, 1> Action;

  using Trajectory_ = Memory::Trajectory<Dtype, StateDim, ActionDim>;
  using Task_ = Task::Task<Dtype, StateDim, ActionDim, 0>;
  using Noise_ = Noise::Noise<Dtype, ActionDim>;
  using Acquisitor_ = ExpAcq::TrajectoryAcquisitor_Parallel<Dtype, StateDim, ActionDim>;
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PerformanceTester(){
    Utils::logger->addVariableToLog(2, "PerformanceTester/performance", "");
  }

  void testPerformance(std::vector<Task_ *> &task,
                       std::vector<Noise_ *> &noise,
                       Policy_ *policy,
                       double timeLimit,
                       int testingTrajN,
                       double stepsTaken,
                       int vis_lv,
                       std::string videoFileName) {

    timer->disable();
    testTraj_.resize(testingTrajN);
    for (auto &tra : testTraj_)
      tra.clear();
    for (auto &noise : noise)
      noise->initializeNoise();

    if (vis_lv > 0) {
      task[0]->turnOnVisualization("");
      if (task[0]->shouldRecordVideo())
        task[0]->startRecordingVideo(RAI_LOG_PATH, videoFileName);
    }
    Dtype averageCost = acquisitor_.acquire(task,
                                            policy,
                                            noise,
                                            testTraj_,
                                            timeLimit,
                                            false);
    if (vis_lv > 0) task[0]->turnOffVisualization();
    if (task[0]->shouldRecordVideo())
      task[0]->endRecordingVideo();

    Utils::logger->appendData("PerformanceTester/performance",
                              stepsTaken,
                              averageCost);
    LOG_IF(FATAL, isnan(averageCost)) << "average cost is nan";
    LOG(INFO) << "steps taken " << logger->getData("PerformanceTester/performance")->at(0).back()
              << ", average cost " << logger->getData("PerformanceTester/performance")->at(1).back();

    timer->enable();
  }

 private:
  void sampleBatchOfInitial(StateBatch &initial, std::vector<Task_ *> &task_) {
    for (int trajID = 0; trajID < initial.cols(); trajID++) {
      State state;
      task_[0]->setToInitialState();
      task_[0]->getState(state);
      initial.col(trajID) = state;
    }
  }

  std::vector<Trajectory_> testTraj_;
  Acquisitor_ acquisitor_;
  Policy_ *policy_;

};

}
}

#endif //RAI_PERFORMANCETESTER_HPP
