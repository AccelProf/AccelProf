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

}   // yosemite

#endif // YOSEMITE_UTILS_HELPER_H
