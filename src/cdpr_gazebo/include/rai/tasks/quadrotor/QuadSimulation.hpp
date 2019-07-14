//
// Created by jhwangbo on 04.02.17.
//

#ifndef RAI_SIMULATEQUAD_HPP
#define RAI_SIMULATEQUAD_HPP
#include "QuadrotorControl.hpp"
#include "MLP_QuadControl.hpp"
#include "rai/function/common/Policy.hpp"
#include <sys/time.h>
#include "math/RAI_math.hpp"


namespace rai{
namespace Task{

template<typename Dtype>
class QuadSimulation {
  using Policy_ = FuncApprox::Policy<Dtype, StateDim, ActionDim>;
  using State = typename QuadrotorControl<Dtype>::State;
  using Action = typename QuadrotorControl<Dtype>::Action;

 public:
  QuadSimulation() {
    task.turnOnVisualization("");
    stateTarget.setZero();
    stateTarget.segment(0,9)<<0,0,0,0,0,0,0,0,0;
    Position initialPosition, target0, target1, target2, target3, target4, target5;
    LinearVelocity initialLinVel;
    AngularVelocity initialAngVel;
    Quaternion quat;
//    double orientationScale_, positionScale_, angVelScale_, linVelScale_;
//    positionScale_ = 0.5;
//    angVelScale_ = 0.15;
//    linVelScale_ = 0.5;
    /// exp 4
//    quat << 0.7242, 0.1203  ,  0.5206,  -0.4358 ;
//    initialPosition << -0.4113 ,   0.8562 ,   2.1539;
//    initialLinVel <<   -0.4086 ,  -0.8474  ,  3.1776;
//    initialAngVel << 0.1405 ,  -7.0643  , -0.4982;
//    RotationMatrix R = rai::Math::quatToRotMat(quat);
//    State initialState;
//    initialState << R.col(0), R.col(1), R.col(2),initialPosition * positionScale_, R * initialAngVel *angVelScale_, R * initialLinVel *linVelScale_;
//    task.setToParticularState(initialState);

    task.setToInitialState();

    target0 <<0,0,0;
    target1 <<0,0,1;
    target2 <<1,0,0;
    target3 <<1,0,1;
    target4 <<1,-1,1;

    traj.push_back(std::pair<Dtype, Position>({Dtype(0.0), target0}));
//    traj.push_back(std::pair<Dtype, Position>({Dtype(3.0), target0}));
//    traj.push_back(std::pair<Dtype, Position>({Dtype(6.0), target1}));
//    traj.push_back(std::pair<Dtype, Position>({Dtype(9.0), target2}));
//    traj.push_back(std::pair<Dtype, Position>({Dtype(12.0), target3}));
//    traj.push_back(std::pair<Dtype, Position>({Dtype(15.0), target4}));
    traj.push_back(std::pair<Dtype, Position>({Dtype(1e9), target0}));
    Utils::logger->addVariableToLog(3, "position", "");
    Utils::logger->addVariableToLog(4, "orientation", "");
    timebegin = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
  }

  void loop(Policy_ &policy){
    time += task.dt();

    gettimeofday(&timevalNow, nullptr);
    double timeStart = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
    State state, state_tp1, state_aug;
    Action action;
    TerminationType termType;
    Dtype cost;
    task.getState(state);
    if(time > traj[timeDiscrete].first){
      timeDiscrete++;
      stateTarget.segment(9, 3) = traj[timeDiscrete].second*0.5;
      task.changeTarget(traj[timeDiscrete].second);
    }

    std::cout<<"error norm "<<(state.segment(9, 3) - stateTarget.segment(9, 3)).norm()*2.0<<std::endl<<std::endl;
    state_aug = state - stateTarget;
    policy.forward(state_aug, action);
    task.takeOneStep(action, state_tp1, termType, cost);
    gettimeofday(&timevalNow, nullptr);
    double timeEnd = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
    usleep(std::max((task.dt() - timeEnd + timeStart) * 1e6, 0.0));
  }

  void loop(MLP_QuadControl& policy){
    time += task.dt();

    gettimeofday(&timevalNow, nullptr);
    double timeStart = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
    Action thrust;
    if(time > traj[timeDiscrete].first){
      timeDiscrete++;
      task.changeTarget(traj[timeDiscrete].second);
    }

    RotationMatrix Rot;
    Quaternion quat;
    Position posi;
    AngularVelocity anglvel;
    LinearVelocity linvel;
    task.getOrientation(quat);
    task.getPosition(posi);
    task.getAngvel(anglvel);
    task.getLinvel(linvel);

    Utils::logger->appendData("position", posi.data());
    Utils::logger->appendData("orientation", quat.data());

    std::cout<<"\r error norm "<<(posi - traj[timeDiscrete].second).norm()<< "position "<<posi.transpose()<< std::flush;
    rai::Utils::timer->startTimer("mlp");
    policy.forward(quat, posi, anglvel, linvel, traj[timeDiscrete].second, thrust);
    rai::Utils::timer->stopTimer("mlp");
    task.stepSim(thrust);
    gettimeofday(&timevalNow, nullptr);
    double timeEnd = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
    usleep(std::max((task.dt() - timeEnd + timeStart) * 1e6, 0.0));
  }

  void randomInit(){
    task.init();
    Position position = stateTarget.segment(9, 3);
    task.translate(position);
  }

 private:
  QuadrotorControl<Dtype> task;
  timeval timevalNow;
  State stateTarget;
  double timebegin;
  int timeDiscrete = 0;
  double time = 0;
  std::vector<std::pair<Dtype, Position> > traj;

};

}
}


#endif //RAI_SIMULATEQUAD_HPP
