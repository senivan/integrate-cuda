#include "utils.hpp"

#include <sstream>
#include <unordered_map>
#include <stdexcept>
#include <algorithm>
#include <cctype>
#include <locale>
#include <atomic>
static inline std::string trim(const std::string &s) {
    auto start = s.begin();
    while (start != s.end() && std::isspace(*start)) {
        ++start;
    }
    auto end = s.end();
    while (end != start && std::isspace(*(end - 1))) {
        --end;
    }
    return std::string(start, end);
}

conf_file_t parse_conf_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Can't open config file: " + filename);
    }

    std::unordered_map<std::string, std::string> config;
    std::string line;
    while (std::getline(file, line)) {
        auto commentPos = line.find('#');
        if (commentPos != std::string::npos) {
            line = line.substr(0, commentPos);
        }
        line = trim(line);
        if (line.empty()) continue;  

        auto equalPos = line.find('=');
        if (equalPos == std::string::npos) {
            throw std::runtime_error("Invalid config line (no '=' found): " + line);
        }
        std::string key = trim(line.substr(0, equalPos));
        std::string value = trim(line.substr(equalPos + 1));
        config[key] = value;
    }

    const char* requiredKeys[] = {
        "abs_err", "rel_err", "x_start", "x_end", "y_start", "y_end",
        "init_steps_x", "init_steps_y", "max_iter"
    };
    for (const char* key : requiredKeys) {
        if (config.find(key) == config.end()) {
            throw std::runtime_error("Missing required key in config file: " + std::string(key));
        }
    }

    conf_file_t conf;
    try {
        conf.abs_err       = std::stod(config["abs_err"]);
        conf.rel_err       = std::stod(config["rel_err"]);
        conf.x_start       = std::stod(config["x_start"]);
        conf.x_end         = std::stod(config["x_end"]);
        conf.y_start       = std::stod(config["y_start"]);
        conf.y_end         = std::stod(config["y_end"]);
        conf.init_steps_x  = std::stoi(config["init_steps_x"]);
        conf.init_steps_y  = std::stoi(config["init_steps_y"]);
        conf.max_iter      = std::stoi(config["max_iter"]);
    } catch (const std::exception &ex) {
        throw std::runtime_error("Error parsing config values: " + std::string(ex.what()));
    }
    return conf;
}
std::chrono::high_resolution_clock::time_point get_current_time_fenced() {
    std::atomic_thread_fence(std::memory_order_seq_cst);
    auto res_time = std::chrono::high_resolution_clock::now();
    
    std::atomic_thread_fence(std::memory_order_seq_cst);
    return res_time;
}

