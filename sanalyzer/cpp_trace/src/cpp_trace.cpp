#include "cpp_trace.h"
#include <cstdlib>
#include <backtrace.h>
#include <cxxabi.h>

static void *__bt_state = nullptr;

static std::string backtrace_str;
static std::vector<std::string> backtraces;
static int bt_index = 0;
static int bt_keep = -1;

int bt_callback(void *, uintptr_t, const char *filename, int lineno, const char *function) {
  /// demangle function name
  const char *func_name = function;
  int status;
  char *demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);
  if (status == 0) {
    func_name = demangled;
  }

  if (filename && func_name) {
    backtrace_str += "b-" + std::to_string(bt_index) + " " + std::string(filename) + ":" + std::to_string(lineno) + " " + func_name + "\n";
    bt_index++;
  }
  return 0;
}

bool starts_with(const std::string& filename, const std::string& prefix) {
    return filename.compare(0, prefix.size(), prefix) == 0;
}

int get_bt_callback(void *, uintptr_t, const char *filename, int lineno, const char *function) {
  if (bt_keep != -1 && bt_index >= bt_keep) {
    return 0;
  }
  /// demangle function name
  const char *func_name = function;
  int status;
  char *demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);
  if (status == 0) {
    func_name = demangled;
  }

  // skip the backtrace from /usr/local
  if (filename && func_name && !starts_with(filename, "/usr/local")) {
    backtraces.push_back("b-" + std::to_string(bt_index) + " " + std::string(filename) + ":" + std::to_string(lineno) + " " + func_name);
    bt_index++;
  }
  return 0;
}

void bt_error_callback(void *, const char *msg, int errnum) {
  printf("Error %d occurred when getting the stacktrace: %s", errnum, msg);
}

void bt_error_callback_create(void *data, const char *msg, int errnum) {
  printf("Error %d occurred when initializing the stacktrace: %s", errnum, msg);
  bool *status = (bool *)data;
  *status = false;
}

bool init_backtrace(const char *filename) {
  bool status = true;
  __bt_state = backtrace_create_state(filename, 0, bt_error_callback_create, (void *)status);
  return status;
}

std::string print_backtrace(int verbose) {
  if (!__bt_state) { /// make sure init_backtrace() is called
    printf("Make sure init_backtrace() is called before calling print_stack_trace()\n");
    abort();
  }
  backtrace_str.clear();
  bt_index = 0;
  backtrace_full((backtrace_state *)__bt_state, 0, bt_callback, bt_error_callback, nullptr);

  if (verbose) {
    printf("%s\n", backtrace_str.c_str());
    fflush(stdout);
  }
  return backtrace_str;
}

/*
* keep = -1, get all backtraces
*/
std::vector<std::string> get_backtrace(int keep) {
  if (!__bt_state) { /// make sure init_backtrace() is called
    printf("Make sure init_backtrace() is called before calling print_stack_trace()\n");
    abort();
  }
  backtraces.clear();
  bt_index = 0;
  bt_keep = keep;
  backtrace_full((backtrace_state *)__bt_state, 0, get_bt_callback, bt_error_callback, nullptr);

  return backtraces;
}