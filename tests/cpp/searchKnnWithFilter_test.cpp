// This is a test file for testing the filtering feature

#include <gtest/gtest.h>
#include "hnswlib/hnswlib.h"
#include <vector>
#include <iostream>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

namespace {

using idx_t = hnswlib::labeltype;

class PickDivisibleIds: public hnswlib::BaseFilterFunctor {
unsigned int divisor = 1;
 public:
    PickDivisibleIds(unsigned int divisor): divisor(divisor) {
        assert(divisor != 0);
    }
    bool operator()(idx_t label_id) {
        return label_id % divisor == 0;
    }
};

class PickNothing: public hnswlib::BaseFilterFunctor {
 public:
    bool operator()(idx_t label_id) {
        return false;
    }
};

class CustomFilterFunctor: public hnswlib::BaseFilterFunctor {
    std::unordered_set<idx_t> allowed_values;

 public:
    explicit CustomFilterFunctor(const std::unordered_set<idx_t>& values) : allowed_values(values) {}

    bool operator()(idx_t id) {
        return allowed_values.count(id) != 0;
    }
};


class searchKnnWithFilter {
public:
    searchKnnWithFilter(hnswlib::BaseFilterFunctor& filter_func, size_t div_num, size_t label_id_start):
        filter_func(filter_func), div_num(div_num), label_id_start(label_id_start) {   
        d = 4;
        n = 100;
        nq = 10;
        k = 10;
        data = std::vector<float>(n * d);
        query = std::vector<float>(nq * d);

        std::mt19937 rng;
        rng.seed(47);
        std::uniform_real_distribution<> distrib;

        for (idx_t i = 0; i < n * d; ++i) {
            data[i] = distrib(rng);
        }
        for (idx_t i = 0; i < nq * d; ++i) {
            query[i] = distrib(rng);
        }

        space = new hnswlib::L2Space(d);
        alg_brute  = new hnswlib::BruteforceSearch<float>(space, 2 * n);
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, 2 * n);

        for (size_t i = 0; i < n; ++i) {
            // `label_id_start` is used to ensure that the returned IDs are labels and not internal IDs
            alg_brute->addPoint(data.data() + d * i, label_id_start + i);
            alg_hnsw->addPoint(data.data() + d * i, label_id_start + i);
        }
    }
    ~searchKnnWithFilter() {
        delete alg_brute;
        delete alg_hnsw;
        delete space;
    }

    void test_code(hnswlib::AlgorithmInterface<float>* alg) {
        for (size_t j = 0; j < nq; ++j) {
            const void* p = query.data() + j * d;
            auto gd = alg->searchKnn(p, k, &filter_func);
            auto res = alg->searchKnnCloserFirst(p, k, &filter_func);
            EXPECT_EQ(gd.size(), res.size());
            if (div_num > 0) {  // Test with filter
                size_t t = gd.size();
                while (!gd.empty()) {
                    EXPECT_EQ(gd.top(), res[--t]);
                    //std::cout << gd.top().second << " " << div_num << std::endl;
                    EXPECT_EQ((gd.top().second % div_num), 0);
                    gd.pop();
                }
            } else {  // Test with nonfilter (div_num=-1)
                EXPECT_EQ(0, gd.size());
            }
        }
    }

    void RunTest() {
        test_code(this->alg_brute);
        test_code(this->alg_hnsw);
    }

    hnswlib::BaseFilterFunctor& filter_func;
    hnswlib::L2Space* space;
    hnswlib::AlgorithmInterface<float>* alg_brute; 
    hnswlib::AlgorithmInterface<float>* alg_hnsw;
    size_t div_num;
    size_t label_id_start;
    std::vector<float> data;
    std::vector<float> query;
    int d;
    idx_t n;
    idx_t nq;
    size_t k;
};

TEST(searchKnnWithFilterTest, DivBy3StartId17) {
    PickDivisibleIds pickIdsDivisibleByThree(3);
    searchKnnWithFilter seach(pickIdsDivisibleByThree, 3, 17);
    seach.RunTest();
}

TEST(searchKnnWithFilterTest, DivBy7StartId17) {
    PickDivisibleIds pickIdsDivisibleByThree(7);
    searchKnnWithFilter seach(pickIdsDivisibleByThree, 7, 17);
    seach.RunTest();
}

TEST(searchKnnWithFilterTest, PickNothing) {
    PickNothing pickNothing;
    searchKnnWithFilter seach(pickNothing, -1, 17);
    seach.RunTest();
}

TEST(searchKnnWithFilterTest, CustomFilterFunctor) {
    CustomFilterFunctor pickIdsDivisibleByThirteen({26, 39, 52, 65});
    searchKnnWithFilter seach(pickIdsDivisibleByThirteen, 13, 21);
    seach.RunTest();
}
}