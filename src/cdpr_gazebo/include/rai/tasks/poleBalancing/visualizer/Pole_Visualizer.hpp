
#include "raiGraphics/RAI_graphics.hpp"
#include "raiGraphics/obj/Mesh.hpp"
#include "raiCommon/math/RAI_math.hpp"
#include "raiCommon/TypeDef.hpp"
#include "raiGraphics/obj/Cylinder.hpp"
#include "raiGraphics/obj/Sphere.hpp"
#include "raiGraphics/obj/Arrow.hpp"


namespace rai {
namespace Vis {

class Pole_Visualizer {

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Pole_Visualizer();

  ~Pole_Visualizer();

  void setTerrain(std::string fileName);
  void drawWorld(HomogeneousTransform &bodyPose, double action);
  rai_graphics::RAI_graphics* getGraphics();


 private:
  rai_graphics::RAI_graphics graphics;
  rai_graphics::object::Cylinder Pole;
  rai_graphics::object::Sphere Dot, origin;

  rai_graphics::object::Mesh arrow;
  HomogeneousTransform defaultPose_;
};

}
}
