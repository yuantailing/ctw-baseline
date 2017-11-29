#ifndef EVAL_TOOLS_HPP
#define EVAL_TOOLS_HPP

#include <algorithm>
#include <cstdarg>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <vector>
#include "rapidjson/include/rapidjson/document.h"
#include "rapidjson/include/rapidjson/stringbuffer.h"
#include "rapidjson/include/rapidjson/writer.h"


struct BBox {
    double x, y, w, h;
    BBox() { }
    BBox(rapidjson::Value const &j):
        x(j[0].GetDouble()),
        y(j[1].GetDouble()),
        w(j[2].GetDouble()),
        h(j[3].GetDouble()) { }
    double iou(BBox const &o) const {
        assert(w >= 0 && h >= 0 && o.w >= 0 && o.h >= 0);
        double A0 = w * h;
        double A1 = o.w * o.h;
        if (0 == A0 || 0 == A1)
            return 0;
        double Nw = std::min<double>(x + w, o.x + o.w) - std::max<double>(x, o.x);
        double Nh = std::min<double>(y + h, o.y + o.h) - std::max<double>(y, o.y);
        double AN = std::max<double>(0., Nw) * std::max<double>(0., Nh);
        return AN / (A0 + A1 - AN);
    }
    double a_in_b(BBox const &o) const {
        assert(w >= 0 && h >= 0 && o.w >= 0 && o.h >= 0);
        double A0 = w * h;
        if (0 == A0)
            return 0;
        double Nw = std::min<double>(x + w, o.x + o.w) - std::max<double>(x, o.x);
        double Nh = std::min<double>(y + h, o.y + o.h) - std::max<double>(y, o.y);
        double AN = std::max<double>(0., Nw) * std::max<double>(0., Nh);
        return AN / A0;
    }
};

struct Attribute {
    int k;
    Attribute(): k(0) { }
    Attribute(rapidjson::Value const &j, std::vector<std::string> const &attrs): k(0) {
        assert(attrs.size() < 15);
        for (rapidjson::Value const &o: j.GetArray()) {
            std::string const &attr = o.GetString();
            int old_k = k;
            for (std::size_t i = 0; i < attrs.size(); i++) {
                if (attrs[i] == attr) {
                    k |= 1 << i;
                }
            }
            assert(old_k != k);
        }
    }
};

template<typename=void>
std::string detection_mAP(
        std::vector<std::string> const &ground_truth,
        std::vector<std::string> const &detection,
        std::vector<std::string> const &attributes,
        std::vector<std::pair<std::string, std::pair<double, double> > > const &size_ranges,
        int max_det,
        double iou_thresh,
        bool proposal=false,
        bool echo=false) {
    auto format = [](std::string const &fmt_str, ...)->std::string {
        int final_n, n = ((int)fmt_str.size()) * 2;
        std::unique_ptr<char[]> formatted;
        std::va_list ap;
        while(1) {
            formatted.reset(new char[n]);
            strcpy(&formatted[0], fmt_str.c_str());
            va_start(ap, fmt_str);
            final_n = std::vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
            va_end(ap);
            if (final_n < 0 || final_n >= n)
                n += std::abs(final_n - n + 1);
            else
                break;
        }
        return formatted.get();
    };

    auto error = [](std::string const &msg)->std::string {
        rapidjson::Document d;
        d.SetObject();
        d.AddMember("error", 1, d.GetAllocator());
        d.AddMember("msg", {msg.c_str(), (rapidjson::SizeType)msg.size(), d.GetAllocator()}, d.GetAllocator());
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        d.Accept(writer);
        return buffer.GetString();
    };

    struct APData {
        struct DMatch {
            int match_status;
            std::size_t i_dt;
            double score;
        };
        int n = 0;
        std::vector<DMatch> dt;
        std::vector<std::pair<int, int> > attributes;
        APData &operator +=(APData const &o) {
            n += o.n;
            for (DMatch const &dm: o.dt)
                dt.push_back(dm);
            if (attributes.size() < o.attributes.size())
                attributes.resize(o.attributes.size(), {0, 0});
            for (std::size_t i = 0; i < o.attributes.size(); i++) {
                attributes[i].first += o.attributes[i].first;
                attributes[i].second += o.attributes[i].second;
            }
            return *this;
        }
        std::pair<double, std::vector<double> > AP_compute() {
            if (0 == n)
                return {-1., {}};
            std::vector<double> acc;
            std::vector<int> rc_inc;
            std::stable_sort(dt.begin(), dt.end(), [](DMatch const &a, DMatch const &b) {
                if (a.score != b.score) return a.score > b.score;
                if (a.i_dt != b.i_dt) return a.i_dt < b.i_dt;
                return a.match_status < b.match_status;
            });
            int match_cnt = 0;
            for (std::size_t i = 0; i < dt.size(); i++) {
                int matched = dt[i].match_status;
                assert(0 == matched || 1 == matched);
                match_cnt += matched;
                acc.push_back((double)match_cnt / (i + 1));
                rc_inc.push_back(matched);
            }
            for (int i = (int)acc.size() - 1; i > 0; i--)
                acc[i - 1] = std::max(acc[i - 1], acc[i]);
            double AP = 0;
            std::vector<double> curve;
            for (std::size_t i = 0; i < acc.size(); i++) {
                double a = acc[i];
                int r = rc_inc[i];
                AP += a * r;
                if (1 == r)
                    curve.push_back(a);
            }
            return {AP / n, curve};
        }
    };

    char const *eval_type = proposal ? "proposals" : "detections";
    std::vector<std::unordered_map<std::string, APData> > m(size_ranges.size());
    std::vector<std::vector<double> > AP_imgs(size_ranges.size());

    if (ground_truth.size() != detection.size())
        return error(format("number of lines not match: %d expected, %d loaded", (int)ground_truth.size(), (int)detection.size()));

    for (std::size_t i = 0; i < ground_truth.size(); i++) {
        if (echo && 0 == i % 200)
            std::cerr << i << " / " << ground_truth.size() << std::endl;
        std::string const &gt_s(ground_truth[i]);
        std::string const &dt_s(detection[i]);

        rapidjson::Document dt_j;
        if (dt_j.Parse<rapidjson::kParseIterativeFlag|rapidjson::kParseFullPrecisionFlag>(dt_s.c_str(), dt_s.size()).HasParseError())
            return error(format("line %d is not legal json", i + 1));
        if (!dt_j.IsObject())
            return error(format("line %d is not json object", i + 1));
        if (!dt_j.HasMember(eval_type))
            return error(format("line %d does not contain key `%s`", i + 1, eval_type));
        if (!dt_j[eval_type].IsArray())
            return error(format("line %d %s is not an array", i + 1, eval_type));
        if (dt_j[eval_type].Size() > (rapidjson::SizeType)max_det)
            return error(format("line %d number of %s exceeds limit (%d)", i + 1, eval_type, max_det));
        int j = 0;
        for (rapidjson::Value const &chr: dt_j[eval_type].GetArray()) {
            if (!chr.IsObject())
                return error(format("line %d %s candidate %d is not an object", i + 1, eval_type, j + 1));
            if (!proposal && !chr.HasMember("text"))
                return error(format("line %d %s candidate %d does not contain key `text`", i + 1, eval_type, j + 1));
            if (!chr.HasMember("score"))
                return error(format("line %d %s candidate %d does not contain key `score`", i + 1, eval_type, j + 1));
            if (!chr.HasMember("bbox"))
                return error(format("line %d %s candidate %d does not contain key `bbox`", i + 1, eval_type, j + 1));
            if (!chr["text"].IsString())
                return error(format("line %d %s candidate %d text is not text-typet", i + 1, eval_type, j + 1));
            if (!chr["score"].IsNumber())
                return error(format("line %d %s candidate %d score is neither int nor float", i + 1, eval_type, j + 1));
            rapidjson::Value const &bbox(chr["bbox"]);
            if (!bbox.IsArray())
                return error(format("line %d %s candidate %d bbox is not an array", i + 1, eval_type, j + 1));
            if (4 != bbox.Size())
                return error(format("line %d %s candidate %d bbox is illegal", i + 1, eval_type, j + 1));
            for (rapidjson::Value const &t: bbox.GetArray())
                if (!t.IsNumber())
                    return error(format("line %d %s candidate %d bbox is illegalt", i + 1, eval_type, j + 1));
            if (bbox[2].GetDouble() <= 0 || bbox[3].GetDouble() <= 0)
                return error(format("line %d %s candidate %d bbox w or h <= 0", i + 1, eval_type, j + 1));
            j += 1;
        }

        struct DT {
            BBox bbox;
            std::string text;
            double score;
        };
        struct GT {
            BBox bbox;
            std::string text;
            Attribute attribute;
        };
        std::vector<DT> dt;
        for (rapidjson::Value const &o: dt_j[eval_type].GetArray()) {
            dt.push_back({
                o["bbox"],
                o["text"].GetString(),
                o["score"].GetDouble(),
            });
        }
        std::stable_sort(dt.begin(), dt.end(), [](DT const &a, DT const &b) {
            return a.score > b.score;
        });

        rapidjson::Document gtobj;
        gtobj.Parse<rapidjson::kParseFullPrecisionFlag>(gt_s.c_str(), gt_s.size());
        std::vector<BBox> ig;
        for (rapidjson::Value const &o: gtobj["ignore"].GetArray())
            ig.push_back(o["bbox"]);
        std::vector<GT> gt;
        for (rapidjson::Value const &block: gtobj["annotations"].GetArray()) {
            for (rapidjson::Value const &chr: block.GetArray()) {
                if (chr["is_chinese"].GetBool()) {
                    gt.push_back({
                        chr["adjusted_bbox"],
                        chr["text"].GetString(),
                        {chr["attributes"], attributes},
                    });
                }
            }
        }

        std::vector<std::vector<std::pair<double, std::size_t> > > dt_matches(dt.size());
        std::vector<bool> dt_ig(dt.size(), false);
        for (std::size_t i_dt = 0; i_dt < dt.size(); i_dt++) {
            DT const &dtchar(dt[i_dt]);
            for (std::size_t i_gt = 0; i_gt < gt.size(); i_gt++) {
                GT const &gtchar(gt[i_gt]);
                if (proposal || dtchar.text == gtchar.text) {
                    double miou = dtchar.bbox.iou(gtchar.bbox);
                    if (miou > iou_thresh) {
                        dt_matches[i_dt].push_back({-miou, i_gt});
                    }
                }
            }
            for (BBox const &igchar: ig) {
                double miou = dtchar.bbox.a_in_b(igchar);
                if (miou > iou_thresh)
                    dt_ig[i_dt] = true;
            }
        }
        for (std::vector<std::pair<double, std::size_t> > &matches: dt_matches)
            std::stable_sort(matches.begin(), matches.end());

        for (std::size_t i_sz = 0; i_sz < size_ranges.size(); i_sz++) {
            std::pair<double, double> const &size_range(size_ranges[i_sz].second);
            auto in_size = [&](BBox const &bbox)->bool {
                double longsize = std::max(bbox.w, bbox.h);
                return size_range.first <= longsize && longsize < size_range.second;
            };

            std::vector<int> dt_matched(dt.size(), 0);
            for (std::size_t i = 0; i < dt.size(); i++) {
                if (!(in_size(dt[i].bbox) && false == dt_ig[i]))
                    dt_matched[i] = 2;
            }
            std::vector<std::pair<int, int> > gt_taken(gt.size(), {0, -1});
            std::size_t n = 0;
            for (std::size_t i = 0; i < gt.size(); i++) {
                if (!in_size(gt[i].bbox))
                    gt_taken[i].first = 2;
                else
                    n += 1;
            }

            for (std::size_t i_dt = 0; i_dt < dt_matches.size(); i_dt++) {
                for (std::pair<double, std::size_t> const &p: dt_matches[i_dt]) {
                    int i_gt = p.second;
                    if (1 != dt_matched[i_dt] && 1 != gt_taken[i_gt].first) {
                        if (0 == gt_taken[i_gt].first) {
                            dt_matched[i_dt] = 1;
                            gt_taken[i_gt] = {1, i_dt};
                        } else {
                            dt_matched[i_dt] = 2;
                        }
                    }
                }
            }
            APData m_img;
            for (std::size_t i_dt = 0; i_dt < dt.size(); i_dt++) {
                DT const &dtchar(dt[i_dt]);
                int match_status = dt_matched[i_dt];
                if (2 != match_status) {
                    m[i_sz][dtchar.text].dt.push_back({match_status, i_dt, dtchar.score});
                    m_img.dt.push_back({match_status, i_dt, dtchar.score});
                }
            }
            std::unordered_set<int> top_dt;
            for (std::size_t i = 0; i < dt.size(); i++) {
                if (top_dt.size() >= n)
                    break;
                if (2 != dt_matched[i])
                    top_dt.insert(i);
            }
            for (std::size_t i = 0; i < gt.size(); i++) {
                GT const &gtchar(gt[i]);
                int taken = gt_taken[i].first;
                int dt_id = gt_taken[i].second;
                if (2 != taken) {
                    APData &thism(m[i_sz][gtchar.text]);
                    thism.n += 1;
                    m_img.n += 1;
                    if (thism.attributes.empty())
                        thism.attributes.resize(1 << attributes.size(), {0, 0});
                    thism.attributes[gtchar.attribute.k].first += 1;
                    if (0 != taken && top_dt.find(dt_id) != top_dt.end())
                        thism.attributes[gtchar.attribute.k].second += 1;
                }
            }
            double AP_img = m_img.AP_compute().first;
            AP_imgs[i_sz].push_back(AP_img);
        }
    }

    rapidjson::Document performance;
    performance.SetObject();
    for (std::size_t i_sz = 0; i_sz < size_ranges.size(); i_sz++) {
        std::string const &szname = size_ranges[i_sz].first;
        int n = 0;
        APData apd;
        apd.attributes.resize(1 << attributes.size(), {0, 0});
        rapidjson::Value texts;
        texts.SetObject();
        std::vector<std::pair<int, std::vector<double> > > mAP_curves;
        for (auto it = m[i_sz].begin(); it != m[i_sz].end(); ++it) {
            std::string const &text(it->first);
            APData &stat(it->second);
            if (0 == stat.n)
                continue;
            n += stat.n;
            std::pair<double, std::vector<double> > p = stat.AP_compute();
            double AP = p.first;
            mAP_curves.push_back(std::make_pair(stat.n, p.second));
            apd += stat;
            rapidjson::Value thistext;
            thistext.SetObject();
            thistext.AddMember("n", stat.n, performance.GetAllocator());
            thistext.AddMember("AP", AP, performance.GetAllocator());
            texts.AddMember({text.c_str(), (rapidjson::SizeType)text.size(), performance.GetAllocator()}, thistext, performance.GetAllocator());
        }
        std::vector<std::pair<double, double> > mAP_curve;
        std::vector<std::size_t> heads(mAP_curves.size(), 0);
        std::pair<int, int> const rc_inf(1, 0);
        auto rc_lt = [](std::pair<int, int> const &a, std::pair<int, int> const &b)->bool {
            return (long long)a.first * (long long)b.second < (long long)b.first * (long long)a.second;
        };
        auto rc_get = [&](std::size_t i)->std::pair<int, int> {
            return heads[i] == mAP_curves[i].second.size() ? rc_inf : std::make_pair(1 + (int)heads[i], mAP_curves[i].first);
        };
        auto cmp_heap = [&](std::size_t a, std::size_t b)->bool {
            return rc_lt(rc_get(b), rc_get(a));
        };
        std::vector<std::size_t> rcheap;
        for (std::size_t i = 0; i < mAP_curves.size(); i++)
            rcheap.push_back(i);
        std::make_heap(rcheap.begin(), rcheap.end(), cmp_heap);
        long double accsum = 0;
        std::for_each(mAP_curves.begin(), mAP_curves.end(), [&](std::pair<int, std::vector<double> > const &p) {
            if (!p.second.empty())
                accsum += p.first * p.second[0];
        });
        while (0 < rcheap.size() && rc_get(rcheap[0]) != rc_inf) {
            std::pair<int, int> rcmin = rc_get(rcheap[0]);
            mAP_curve.push_back(std::make_pair((double)rcmin.first / (double)rcmin.second, (double)(accsum / n)));
            while (!rc_lt(rcmin, rc_get(rcheap[0]))) {
                std::size_t i = rcheap[0];
                std::pop_heap(rcheap.begin(), rcheap.end(), cmp_heap);
                assert(heads[i] < mAP_curves[i].second.size());
                heads[i] += 1;
                long double rc_delta = (long double)mAP_curves[i].second[heads[i] - 1]
                        - (heads[i] < mAP_curves[i].second.size() ? (long double)mAP_curves[i].second[heads[i]] : 0.);
                accsum -= mAP_curves[i].first * rc_delta;
                std::push_heap(rcheap.begin(), rcheap.end(), cmp_heap);
            }
        }
        long double lmAP = 0.;
        rapidjson::Value mAP_jcurve;
        mAP_jcurve.SetArray();
        for (std::size_t i = 0; i < mAP_curve.size(); i++) {
            double lastrc = i == 0 ? 0 : mAP_curve[i - 1].first;
            double rc = mAP_curve[i].first;
            double acc = mAP_curve[i].second;
            lmAP += ((long double)rc - (long double)lastrc) * (long double)acc;
            rapidjson::Value point;
            point.SetArray();
            point.PushBack(rc, performance.GetAllocator());
            point.PushBack(acc, performance.GetAllocator());
            mAP_jcurve.PushBack(point, performance.GetAllocator());
        }
        double mAP = (double)lmAP;

        rapidjson::Value szattrs;
        szattrs.SetArray();
        for (std::pair<int, int> const &p: apd.attributes) {
            rapidjson::Value j;
            j.SetObject();
            j.AddMember("n", p.first, performance.GetAllocator());
            j.AddMember("recall", p.second, performance.GetAllocator());
            szattrs.PushBack(j, performance.GetAllocator());
        }
        std::pair<double, std::vector<double> > AP_all = apd.AP_compute();
        int mAP_micro_n = 0;
        double mAP_micro_sum = 0;
        for (double AP_img: AP_imgs[i_sz])
            if (AP_img >= 0) {
                mAP_micro_n += 1;
                mAP_micro_sum += AP_img;
            }
        rapidjson::Value curve;
        curve.SetArray();
        for (double const &acc: AP_all.second)
            curve.PushBack(acc, performance.GetAllocator());
        rapidjson::Value szperf;
        szperf.SetObject();
        szperf.AddMember("n", n, performance.GetAllocator());
        if (proposal) {
            if (AP_all.first < 0)
                szperf.AddMember("mAP", rapidjson::Value(), performance.GetAllocator());
            else
                szperf.AddMember("mAP", AP_all.first, performance.GetAllocator());
        } else {
            if (0 == n)
                szperf.AddMember("mAP", rapidjson::Value(), performance.GetAllocator());
            else
                szperf.AddMember("mAP", mAP, performance.GetAllocator());
            szperf.AddMember("mAP_curve", mAP_jcurve, performance.GetAllocator());
        }
        szperf.AddMember("attributes", szattrs, performance.GetAllocator());
        szperf.AddMember("texts", texts, performance.GetAllocator());
        if (AP_all.first < 0)
            szperf.AddMember("AP", rapidjson::Value(), performance.GetAllocator());
        else
            szperf.AddMember("AP", AP_all.first, performance.GetAllocator());
        if (0 == mAP_micro_n)
            szperf.AddMember("mAP_micro", rapidjson::Value(), performance.GetAllocator());
        else
            szperf.AddMember("mAP_micro", mAP_micro_sum / mAP_micro_n, performance.GetAllocator());
        szperf.AddMember("AP_curve", curve, performance.GetAllocator());
        performance.AddMember({szname.c_str(), (rapidjson::SizeType)szname.size(), performance.GetAllocator()}, szperf, performance.GetAllocator());
    }
    rapidjson::Document report;
    report.SetObject();
    report.AddMember("error", 0, report.GetAllocator());
    report.AddMember("performance", performance, report.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    report.Accept(writer);
    return buffer.GetString();
}

#endif // EVAL_TOOLS_HPP
