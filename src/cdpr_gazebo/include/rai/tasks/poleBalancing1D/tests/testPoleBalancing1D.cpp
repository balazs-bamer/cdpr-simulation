#include <iostream>

#include <tasks/poleBalancing1D/PoleBalancing1D.hpp>

using std::cout;
using std::endl;

using Dtype = float;
using PoleBalancing1D = rai::Task::Reaching2D<Dtype>;
using State = PoleBalancing1D::State;
using Action = PoleBalancing1D::Action;
using TerminationType = rai::TerminationType;

int main(){

  PoleBalancing1D pole;
  pole.turnOnVisualization("");

  State state, nextState;
  state << 3.14/4;
  pole.setInitialState(state);
  pole.setToInitialState();

  Action action;
  action << 0.1;

  for(int i = 0; i < 100; ++i){
    Dtype cost;
    TerminationType termType;
    pole.takeOneStep(action, nextState, termType, cost);
    pole.getState(state);
    cout << state << endl;
  }
}