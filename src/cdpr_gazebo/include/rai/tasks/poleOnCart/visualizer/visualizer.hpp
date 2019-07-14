/*
 * Copyright (c) 2013, Remo Diethelm, Christian Gehring
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the Autonomous Systems Lab, ETH Zurich nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRight_ HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Remo Diethelm, Christian Gehring
 * BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
*/

#ifndef VISUALIZER_HPP_
#define VISUALIZER_HPP_

#include <Eigen/Dense>
#include <Eigen/Core>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>


class World;
class ContactManager;


class Visualizer
{
 private:
  World* world = nullptr;
  ContactManager* contactManager = nullptr;

  int windowWidth;
  int windowHeight;
  double windowAspectRatio;

  // light
  GLfloat ambient0[4] = {0.1f, 0.1f, 0.1f, 1.0f};
  GLfloat diffuse0[4] = {0.5f, 0.5f, 0.5f, 1.0f};
  GLfloat specular0[4] = {0.7f, 0.7f, 0.7f, 1.0f};
  GLfloat position0[4] = {100.0f, 200.0f, 200.0f, 0.0f };
  std::string filePath;

 public:
  Visualizer();
  virtual ~Visualizer();
  void drawWorld(Eigen::Vector2d angle, std::string info);

 private:
  void updateView();
  void updateLightPositions();
  void updateProjection();
  void DrawCircle(float cx, float cy, float r, int num_segments);

};



#endif
