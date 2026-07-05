#ifndef _PYTHON_FRAME_H_
#define _PYTHON_FRAME_H_
#include <string>
#include <vector>

extern "C" std::string print_pyframes(int verbose = 1);

extern "C" std::vector<std::string> get_pyframes(int keep = -1);


#endif  // _PYTHON_FRAME_H_
