#ifndef YOSEMITE_UTILS_HELPER_H
#define YOSEMITE_UTILS_HELPER_H

#include <string>
#include <cstddef>
#include <cstdint>

namespace yosemite {

std::string format_size(size_t size);

std::string format_number(uint64_t number);

std::string get_current_date_n_time();

bool check_folder_existance(const std::string &folder);

// Prefix a relative output name with $YOSEMITE_RESULT_DIR (exported by
// bin/accelprof from -o) so that results land where the user asked instead of
// in the current working directory. Absolute names and an unset/empty
// YOSEMITE_RESULT_DIR are returned as-is.
std::string resolve_output_path(const std::string &name);

}   // yosemite

#endif // YOSEMITE_UTILS_HELPER_H
