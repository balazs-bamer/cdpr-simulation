#include <iostream>

#include <noiseModel/OrnsteinUhlenbeckNoise.hpp>
#include <gnuplot_wrapper/gnuplotter.hpp>

using std::endl;
using std::cout;

using Dtype = float;
constexpr int ActionDimension = 1;
using OrnsteinUhlenbeck = rai::Noise::OrnsteinUhlenbeck<Dtype, ActionDimension>;
using Action = Eigen::Matrix<Dtype, ActionDimension, 1>;

int main() {

  OrnsteinUhlenbeck ou(0.15, 0.3);

  std::vector<Dtype> x, noiseSamples;
  for (int i = 0; i < 1000; ++i) {
    x.push_back(i);
    Action action = Action::Zero();
    ou.noisify(action);
    noiseSamples.push_back(action(0, 0));
  }

  Utils::FigProp2D figure1properties;

  Utils::graph->figure(1, figure1properties);
  Utils::graph->appendData(x.data(), noiseSamples.data(), x.size(), "");
  Utils::graph->drawFigure();
}