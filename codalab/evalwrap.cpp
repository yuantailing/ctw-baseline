#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../cppapi/eval_tools.hpp"
#include "settings.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " detection.jsonl <ground_truth.jsonl" << endl;
        exit(1);
    }
    auto load = [](char *file_name, vector<string> &out) {
        ifstream f(file_name);
        assert(f.is_open());
        if (!f.is_open()) {
            cerr << "cannot open " << file_name << endl;
            exit(1);
        }
        out.clear();
        while (!f.eof()) {
            out.push_back(string());
            std::getline(f, out.back());
        }
        f.close();
        while (!out.empty() && out.back().empty())
            out.pop_back();
    };

    vector<string> gt;
    while (cin) {
        gt.push_back(string());
        std::getline(cin, gt.back());
    }
    while (!gt.empty() && gt.back().empty())
        gt.pop_back();
    vector<string> dt;
    load(argv[1], dt);
    string report = detection_mAP(gt, dt, settings_attributes(), settings_size_ranges(), settings_max_det(), settings_iou_thresh(), false, false);
    cout << report << endl;
    return 0;
}
