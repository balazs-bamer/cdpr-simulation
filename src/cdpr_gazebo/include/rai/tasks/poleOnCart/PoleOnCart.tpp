/*
 * PoleBalancing.cpp
 *
 *  Created on: Mar 12, 2016
 *      Author: jhwangbo
 */

#include "PoleOnCart.hpp"
#include <unistd.h>
#include "math.h"

namespace rai {
namespace Task {

using std::sin;
using std::cos;

template<typename Dtype>
PoleOnCart<Dtype>::PoleOnCart() {
////// set default parameters//////////////
	this->valueAtTermination_ = 1.5;
	this->discountFactor_ 		= 0.99;
	this->timeLimit_= 15.0;
	this->controlUpdate_dt_   = 0.05;

  q_<<0.0, 0.0; u_<<0.0,0.0;
  q0_<<0.0, 0.2; u0_<<0.0, 0.0;
  stateScale_<< 0.5, 0.6, 0.3, 0.3;
  stateBias_<< 0.0, 0.0, 0.0, 0.0;

/////// adding constraints////////////////////
  State upperStateBound, lowerStateBound;
  upperStateBound<< 3.0, 1.5 * M_PI, 10000.0, 10000.0;
  lowerStateBound<< -3.0, -1.5 * M_PI, -10000.0, -10000.0;
  upperStateBound = upperStateBound.cwiseProduct(stateScale_) + stateBias_;
  lowerStateBound = lowerStateBound.cwiseProduct(stateScale_) + stateBias_;
  this->setBoxConstraints(lowerStateBound, upperStateBound);

///////////////////////////////////////////////
#ifdef VIS_ON
  visualizer_ = new Visualizer;
	this->visualization_ON_ = false;
#endif
}

template<typename Dtype>
PoleOnCart<Dtype>::~PoleOnCart() {
#ifdef VIS_ON
	delete visualizer_;
#endif
}

template<typename Dtype>
void PoleOnCart<Dtype>::takeOneStep(const Action &action_t,
																					State &state_tp1,
																					TerminationType &termType,
																					Dtype &costOUT) {
  double Sq = sin(q_(1)), Cq = cos(q_(1));
	ddq_(0) = 1.0 / (mc_ + mp_ * Sq*Sq ) * ( (action_t(0) * 10.0  - u_(0)*10.0) + mp_ * Sq * (lp_ * u_(1) *u_(1) + g_*Cq));
  ddq_(1) = 1.0 / (lp_ * (mc_+mp_*Sq*Sq)) * (-(action_t(0) * 10.0 - u_(0)*10.0) * Cq - mp_*lp_*u_(1)*u_(1)*Sq*Cq - (mc_+mp_) * g_ *Sq ) - u_(1)*0.6;
	u_ += ddq_ * this->controlUpdate_dt_;
  q_ += u_ * this->controlUpdate_dt_;

  getState(state_tp1);

  if(this->isTerminalState(state_tp1))
    termType = terminalState;

  costOUT = 0.01 * (1.0 + Cq) + 1e-5 * action_t(0) * action_t(0);

  // visualization
#ifdef VIS_ON
  if(this->visualization_ON_){
		visualizer_->drawWorld(q_, this->visualization_info_);
		usleep(this->controlUpdate_dt_ * 1e6/1.5);
	}
#endif
}

template<typename Dtype>
void PoleOnCart<Dtype>::getGradientStateResAct(const State& stateIN, const Action& actionIN, Jacobian& gradientOUT){
  double Sq = sin(q_(1)), Cq = cos(q_(1));
  gradientOUT(2) = 10.0 / (mc_ + mp_ * Sq*Sq ) *this->controlUpdate_dt_;
  gradientOUT(3) = -10.0 / (lp_ * (mc_ + mp_*Sq*Sq)) * Cq * this->controlUpdate_dt_;
  gradientOUT(0) = gradientOUT(2) * this->controlUpdate_dt_;
  gradientOUT(1) = gradientOUT(3) * this->controlUpdate_dt_;
}

template<typename Dtype>
void PoleOnCart<Dtype>::getGradientCostResAct(const State& stateIN, const Action& actionIN, JacobianCostResAct& gradientOUT){
  gradientOUT(0) = 2e-5 * actionIN(0);
}

template <typename Dtype>
void PoleOnCart<Dtype>::setInitialState(const State& state) {
  state4Learning_ = state.cwiseQuotient(stateScale_) + stateBias_;
  q0_ = state4Learning_.topRows(2).template cast<double>();
  u0_ = state4Learning_.bottomRows(2).template cast<double>();
}

template<typename Dtype>
void PoleOnCart<Dtype>::saveOneState4Visualization(std::ofstream& file)  const{
	file << q_.transpose() << '\n';
}

template<typename Dtype>
void PoleOnCart<Dtype>::setToInitialState() {
	q_ = q0_;
  u_ = u0_;
}

template<typename Dtype>
void PoleOnCart<Dtype>::setToParticularState(const State &state){
  state4Learning_ = state.cwiseQuotient(stateScale_) + stateBias_;
  q_ = state4Learning_.topRows(2).template cast<double>();
  u_ = state4Learning_.bottomRows(2).template cast<double>();
}

template<typename Dtype>
void PoleOnCart<Dtype>::getState(State& state){
  state4Learning_ << q_.template cast<Dtype>(), u_.template cast<Dtype>();
	state = (state4Learning_ - stateBias_).cwiseProduct(stateScale_);
}


}} // namespaces
