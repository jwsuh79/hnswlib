#include <gtest/gtest.h>
#include "hnswlib/hnswlib.h"
#include <iostream>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

using idx_t = hnswlib::labeltype;

class addPointTest : public testing::Test {
    protected:
        void SetUp() override {
            dim = 16;                        
            max = 10000;            
            nq = 10;
            int M = 16;
            int ef_conf = 200;
            label_id_start = 1000;  // ensure that the returned IDs are labels, not internal IDs
            
            space = new hnswlib::L2Space(dim);
            alg_hnsw = new hnswlib::HierarchicalNSW<float>(space, max, M, ef_conf);
            fill_data();
        }
        void TearDown() override {
            delete alg_hnsw;
            delete space;
        }        

        void fill_data() {
            std::mt19937 rng;
            rng.seed(47);
            std::uniform_real_distribution<> distrib;

            data.reserve(this->max*dim);
            query.reserve(this->nq*dim);
            for (size_t i = 0; i < max * dim; ++i) {
                this->data[i] = distrib(rng);                
            }
            for (size_t i=0; i<nq*dim; ++i) {
                this->query[i] = distrib(rng);
            }
        }

        void append_points(int n) {
            static size_t i=0;  // practically same as the cur_element_count
            int last_item = i + n;
            for (; i<last_item; ++i) {
                alg_hnsw->addPoint(data.data()+ dim * i, label_id_start + i);
            }            
        }
        
        void add_points(int n) {
            for (size_t i=0; i<n; ++i) {
                alg_hnsw->addPoint(data.data()+ dim * i, label_id_start + i);
            }
        }

        hnswlib::L2Space *space;
        hnswlib::HierarchicalNSW<float>* alg_hnsw;
        int M;
        int max;
        int dim;
        int nq;

        std::vector<float> data;
        std::vector<float> query;
        int label_id_start;
};

TEST_F(addPointTest, getMaxElementsTest) {
    EXPECT_EQ(alg_hnsw->getMaxElements(), max);
    add_points(100);
    EXPECT_EQ(alg_hnsw->getMaxElements(), max);
}

TEST_F(addPointTest, getCurrentElementCount) {
    EXPECT_EQ(alg_hnsw->getCurrentElementCount(), 0);
    append_points(100);
    EXPECT_EQ(alg_hnsw->getCurrentElementCount(), 100);
    append_points(200);
    EXPECT_EQ(alg_hnsw->getCurrentElementCount(), 300);
    
    add_points(300);  // add points overwrite/update the data if the labels are same
    EXPECT_EQ(alg_hnsw->getCurrentElementCount(), 300);
}

TEST_F(addPointTest, getDeletedCount_markdeleted) {
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 0);
    add_points(100);
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 0);

    for (size_t i=0; i<10; ++i) {
        alg_hnsw->markDelete(label_id_start+i);
    }

    EXPECT_EQ(alg_hnsw->getDeletedCount(), 10);
    add_points(10);  // overwrite the deleted labels
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 0);
}

TEST_F(addPointTest, getDeletedCount_unmarkdeleted) {
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 0);
    add_points(100);
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 0);

    for (size_t i=0; i<10; ++i) {
        alg_hnsw->markDelete(label_id_start+i);
    }

    EXPECT_EQ(alg_hnsw->getDeletedCount(), 10);
    
    for (size_t i=0; i<5; ++i) {
        alg_hnsw->unmarkDelete(label_id_start+i);
    }
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 5);
    
    for (size_t i=5; i<10; ++i) {
        alg_hnsw->unmarkDelete(label_id_start+i);
    }
    EXPECT_EQ(alg_hnsw->getDeletedCount(), 0);
}

//INSTANTIATE_TEST_SUITE_P(VariousM_Parms, getRandomLevelTest, Values(2, 6, 16, 32, 48, 64, 128, 10000));