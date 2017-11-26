#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>
#include <utility>
#include <vector>

template<typename=void>
std::vector<int> settings_recall_n() {
    return {1, 5};
}

template<typename=void>
std::vector<std::pair<std::string, std::pair<double, double> > > settings_size_ranges() {
    return {
        {"all", {0., 4096.}},
        {"large", {32., 4096.}},
        {"medium", {16., 32.}},
        {"small", {0., 16.}},
    };
}

template<typename=void>
std::vector<std::string> settings_attributes() {
    return {
        "occluded",
        "bgcomplex",
        "distorted",
        "raised",
        "wordart",
        "handwritten",
    };
}

template<typename=void>
int settings_max_det() {
    return 1000;
}

template<typename=void>
double settings_iou_thresh() {
    return .5;
}

#endif
