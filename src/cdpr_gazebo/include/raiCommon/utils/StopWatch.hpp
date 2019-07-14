//
// Created by jhwangbo on 01.05.17.
//

#ifndef RAI_STOPWATCH_HPP
#define RAI_STOPWATCH_HPP
#include <sys/time.h>
#include <cstddef>
#include <vector>
#include <string>
#include <algorithm>

class StopWatch{
 public:
  inline void start(){
    gettimeofday(&timevalNow, nullptr);
    startTime = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec;
  }

  inline double measure(){
    gettimeofday(&timevalNow, nullptr);
    return timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec - startTime;
  }

  inline void start(const std::string& name) {
    names.push_back(name);
    gettimeofday(&timevalNow, nullptr);
    startTimes.push_back(timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec);
  }

  inline double measure(const std::string& name, bool reset) {
    gettimeofday(&timevalNow, nullptr);
    ptrdiff_t pos = find(names.begin(), names.end(), name) - names.begin();
    if(pos == names.size()) return 1e15;
    double elapse = timevalNow.tv_sec + 1e-6 * timevalNow.tv_usec - startTimes[pos];

    if(reset){
      names.erase(names.begin() + pos);
      startTimes.erase(startTimes.begin() + pos);
    }

    return elapse;
  }

 private:
  timeval timevalNow;
  double startTime;
  std::vector<std::string> names;
  std::vector<double> startTimes;

};

#endif //RAI_STOPWATCH_HPP
