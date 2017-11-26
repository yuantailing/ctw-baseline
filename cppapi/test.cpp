#include <cassert>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "eval_tools.hpp"
#include "../codalab/settings.hpp"

using namespace std;

int main(int argc, char *argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " ground_truth.jsonl detection.jsonl" << endl;
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
    load(argv[1], gt);
    cerr << "gt.size() = " << gt.size() << endl;
    vector<string> dt;
    load(argv[2], dt);
    cerr << "dt.size() = " << dt.size() << endl;
    clock_t tic = clock();
    string report = detection_mAP(gt, dt, settings_attributes(), settings_size_ranges(), settings_max_det(), settings_iou_thresh(), false, true);
    clock_t toc = clock();
    cout << report.substr(0, 500) << endl;
    cerr << "detection_mAP: " << (double)(toc - tic) / CLOCKS_PER_SEC << " sec" << endl;
    return 0;
}
