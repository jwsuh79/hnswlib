#include <gtest/gtest.h>
#include "hnswlib/hnswlib.h"
#include <iostream>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

using idx_t = hnswlib::labeltype;


int getLevel(double real_dist_value, int M) {
        double reverse_size = 1 / log(1.0 * M);
        double r = -log(real_dist_value) * reverse_size;
        return (int) r;
}

class getRandomLevelTest : public TestWithParam<int> {
    protected:
        void SetUp() override {
            M = GetParam();
            int d = 16;
            int max = 10000;
            int ef_conf = 200;
            space = new hnswlib::L2Space(d);
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, max, M, ef_conf);
            min_level = getLevel(1.0, M);
            max_level = getLevel(nextafter(0.0, 1.0), M);
            std::cerr  << "getRandomLevel() with M= " << M << " => " << " in ( " << min_level << ", " << max_level << " )" << std::endl;            
        }
        void TearDown() override {
            delete alg_hnsw;
            delete space;
        }
        hnswlib::L2Space *space;
        hnswlib::HierarchicalNSW<float>* alg_hnsw;
        int M;
        int min_level;
        int max_level;
};

TEST_P(getRandomLevelTest, levelRangeTest) {
    for (int i=0; i<10000; ++i) {
        int cur_level = alg_hnsw->getRandomLevel();
        EXPECT_TRUE((cur_level >= min_level) && (cur_level < max_level));
    }
}

INSTANTIATE_TEST_SUITE_P(VariousM_Parms, getRandomLevelTest, Values(2, 6, 16, 32, 48, 64, 128, 10000));