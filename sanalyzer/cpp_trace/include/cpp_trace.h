#ifndef __CPP_TRACE_H__
#define __CPP_TRACE_H__
#include <string>
#include <vector>

extern "C" bool init_backtrace(const char *filename);
extern "C" std::string print_backtrace(int verbose = 1);
extern "C" std::vector<std::string> get_backtrace(int keep = -1);

#endif // __CPP_TRACE_H__
