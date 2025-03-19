#ifndef UTILS_HPP
#define UTILS_HPP
#include <string>
#include <fstream>
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
#endif