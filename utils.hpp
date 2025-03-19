#ifndef UTILS_HPP
#define UTILS_HPP
#include <string>
#include <fstream>
#include <chrono>
typedef struct {
    double abs_err;
    double rel_err;
    double x_start;
    double x_end;
    double y_start;
    double y_end;
    int init_steps_x;
    int init_steps_y;
    int max_iter;
} conf_file_t;
conf_file_t parse_conf_file(const std::string& filename);
std::chrono::high_resolution_clock::time_point get_current_time_fenced();
template<class D>
inline long long to_ms(const D& d) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(d).count();
}
#endif