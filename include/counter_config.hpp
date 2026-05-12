#ifndef COUNTER_CONFIG_HPP
#define COUNTER_CONFIG_HPP

#include <string>
#include <fstream>
#include <iostream>
#include <array>
#include "json.hpp"

using json = nlohmann::json;

namespace CounterConfig {
    inline bool parseJsonConfig(const std::string& jsonFilePath, std::vector<std::string>& hwCounters) {
        try {
            std::ifstream jsonFile(jsonFilePath);
            if (!jsonFile.is_open()) {
                std::cerr << "Error: Unable to open JSON configuration file: " << jsonFilePath << std::endl;
                return false;
            }

            json configJson;
            jsonFile >> configJson;

            if (!configJson.contains("counters") || !configJson["counters"].is_array()) {
                std::cerr << "Error: No counters array found in JSON file" << std::endl;
                return false;
            }

            auto& counters = configJson["counters"];

            hwCounters.clear();
            for (size_t i = 0; i < counters.size(); i++) {
                hwCounters.push_back(counters[i].get<std::string>());
            }

            return true;
        }
        catch (const json::exception& e) {
            std::cerr << "JSON parsing error: " << e.what() << std::endl;
            return false;
        }
        catch (const std::exception& e) {
            std::cerr << "Error reading configuration: " << e.what() << std::endl;
            return false;
        }
    }
}

#endif // COUNTER_CONFIG_HPP