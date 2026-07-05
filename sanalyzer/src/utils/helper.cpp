#include "utils/helper.h"

#include <sstream>
#include <iomanip>
#include <ctime>
#include <sys/stat.h>   // for folder creation

namespace yosemite {

std::string format_size(size_t size) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (size <= 1024) {
        os << size << " bytes";
    } else if (size <= 1048576) {
        os << (size / 1024.0);
        os << " KB";
    } else if (size <= 1073741824ULL) {
        os << size / 1048576.0;
        os << " MB";
    } else if (size <= 1099511627776ULL) {
        os << size / 1073741824.0;
        os << " GB";
    } else {
        os << size / 1099511627776.0;
        os << " TB";
    }
    return os.str();
}

std::string format_number(uint64_t number) {
    std::ostringstream os;
    os.precision(2);
    os << std::fixed;
    if (number <= 1000) {
        os << number;
    } else if (number <= 1000000) {
        os << number / 1000.0;
        os << " K";
    } else if (number <= 1000000000) {
        os << number / 1000000.0;
        os << " M";
    } else if (number <= 1000000000000ULL) {
        os << number / 1000000000.0;
        os << " B";
    } else {
        os << number / 1000000000000.0;
        os << " T";
    }
    return os.str();
}

std::string get_current_date_n_time() {
    // Get current time as time_t
    std::time_t now = std::time(nullptr);
    
    // Convert time_t to tm struct for local time
    std::tm* local_time = std::localtime(&now);
    
    // Create a stringstream to format the date and time
    std::stringstream ss;
    ss << (local_time->tm_year + 1900) << "-"  // Year
       << std::setw(2) << std::setfill('0') << (local_time->tm_mon + 1) << "-"  // Month (tm_mon is 0-11)
       << std::setw(2) << std::setfill('0') << local_time->tm_mday << "_"       // Day of the month
       << std::setw(2) << std::setfill('0') << local_time->tm_hour << "-"       // Hour
       << std::setw(2) << std::setfill('0') << local_time->tm_min << "-"        // Minute
       << std::setw(2) << std::setfill('0') << local_time->tm_sec;              // Second

    // Convert the stringstream to string and return
    return ss.str();
}

bool check_folder_existance(const std::string &folder) {
    // Check if the folder exists
    struct stat info;
    if (stat(folder.c_str(), &info) != 0) {
        // Folder does not exist, create it
        if (mkdir(folder.c_str(), 0777) == 0) {
            fprintf(stdout, "Folder %s created successfully.\n", folder.c_str());
            return true;
        } else {
            fprintf(stderr, "Failed to create folder %s.\n", folder.c_str());
            return false;
        }
    } else if (info.st_mode & S_IFDIR) {
        // Folder exists
        fprintf(stdout, "Folder %s already exists.\n", folder.c_str());
        return true;
    } else {
        fprintf(stderr, "%s exists but is not a directory.\n", folder.c_str());
        return false;
    }
}

}   // yosemite