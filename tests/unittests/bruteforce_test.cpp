#include <gtest/gtest.h>
#include "hnswlib/hnswlib.h"
#include <iostream>

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

using idx_t = hnswlib::labeltype;
using hnswlib::labeltype;


int getLevel(double real_dist_value, int M) {
        double reverse_size = 1 / log(1.0 * M);
        double r = -log(real_dist_value) * reverse_size;
        return (int) r;
}

class BruteForceTest: public testing::Test {
    protected:
        void SetUp() override {
            dim = 16;                        
            //int max = 10000;
            max = 100;
            nq = 10;
            int M = 16;
            int ef_conf = 200;
            label_id_start = 1000;  // ensure that the returned IDs are labels, not internal IDs
            
            space = new hnswlib::L2Space(dim);
            alg_brute = new hnswlib::BruteforceSearch<float>(space, max * 2);
            fill_data();
        }
        void TearDown() override {
            delete alg_brute;
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
        hnswlib::L2Space *space;
        hnswlib::BruteforceSearch<float>* alg_brute;
        int M;
        int max;
        int dim;
        int nq;

        std::vector<float> data;
        std::vector<float> query;
        int label_id_start;
};

TEST_F(BruteForceTest, test_searchKnn) {
    // Add Points
    for (size_t i=0; i<max; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }

    int correct = 0;
    for (int i=0; i < max; ++i) {        
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_brute->searchKnn(data.data() + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == label_id_start+i) correct++;        
    }
    EXPECT_EQ(correct, max);    
}

TEST_F(BruteForceTest, test_searchKnnCloserFirst) {
    // Add Points
    for (size_t i=0; i<max; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }

    int correct = 0;
    for (int i=0; i < max; ++i) {        
        auto result = alg_brute->searchKnnCloserFirst(data.data() + i * dim, 10);
        hnswlib::labeltype label = result[0].second;
        if (label == label_id_start+i) correct++;        
    }
    EXPECT_EQ(correct, max);    
}

TEST_F(BruteForceTest, test_addPoint) {
    // Add Points
    for (size_t i=0; i<3; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }
    EXPECT_EQ(alg_brute->cur_element_count, 3);

    // Adding the same labels overwrite the previous data with the labels.
    for (size_t i=0; i<3; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }
    EXPECT_EQ(alg_brute->cur_element_count, 3);
}


TEST_F(BruteForceTest, test_removePoint) {
    // Add Points Data Layout: 1000, 1001, 1002
    for (size_t i=0; i<3; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }
    
    alg_brute->removePoint(label_id_start);  // remove 1st data, 100
    EXPECT_EQ(alg_brute->cur_element_count, 2);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 2);
    EXPECT_EQ(alg_brute->get_label(1), label_id_start + 1);
    
    alg_brute->removePoint(label_id_start);  // remove 1st data again, nothing removed
    EXPECT_EQ(alg_brute->cur_element_count, 2);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 2);
    EXPECT_EQ(alg_brute->get_label(1), label_id_start + 1);
    
    alg_brute->removePoint(label_id_start + 2);  // remove 3rd data, 1002
    EXPECT_EQ(alg_brute->cur_element_count, 1);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 1);
    
    alg_brute->removePoint(label_id_start + 2);  // remove 3rd data again, nothing removed
    EXPECT_EQ(alg_brute->cur_element_count, 1);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 1);

    alg_brute->removePoint(label_id_start + 1);  // remove 2nd data
    EXPECT_EQ(alg_brute->cur_element_count, 0);    
}


TEST_F(BruteForceTest, test_removePoint_reverse) {
    // Add Points Data Layout: 1000, 1001, 1002
    for (size_t i=0; i<3; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }
    
    alg_brute->removePoint(label_id_start + 2);  // remove 3rd data    
    EXPECT_EQ(alg_brute->cur_element_count, 2);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start);
    EXPECT_EQ(alg_brute->get_label(1), label_id_start + 1);

    alg_brute->removePoint(label_id_start + 2);  // remove 3st data again, nothing removed
    EXPECT_EQ(alg_brute->cur_element_count, 2);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start);
    EXPECT_EQ(alg_brute->get_label(1), label_id_start + 1);

    alg_brute->removePoint(label_id_start + 1);  // remove 2nd data
    EXPECT_EQ(alg_brute->cur_element_count, 1);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start);
   
    alg_brute->removePoint(label_id_start);  // remove 2nd data again
    EXPECT_EQ(alg_brute->cur_element_count, 0);
}

TEST_F(BruteForceTest, test_removePoint_inorder) {
    // Add Points Data Layout: 1000, 1001, 1002
    for (size_t i=0; i<3; ++i) {
        alg_brute->addPoint(data.data()+ dim * i, label_id_start + i);
    }

     alg_brute->removePoint(label_id_start);  // remove 1st data, 1000
    EXPECT_EQ(alg_brute->cur_element_count, 2);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 2);
    EXPECT_EQ(alg_brute->get_label(1), label_id_start + 1);
    
    alg_brute->removePoint(label_id_start);  // remove 1st data again, nothing removed
    EXPECT_EQ(alg_brute->cur_element_count, 2);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 2);
    EXPECT_EQ(alg_brute->get_label(1), label_id_start + 1);

    alg_brute->removePoint(label_id_start + 1);  // remove 2nd data
    EXPECT_EQ(alg_brute->cur_element_count, 1);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 2);    

    alg_brute->removePoint(label_id_start + 1);  // remove 2nd data again
    EXPECT_EQ(alg_brute->cur_element_count, 1);
    EXPECT_EQ(alg_brute->get_label(0), label_id_start + 2);    

    alg_brute->removePoint(label_id_start + 2);  // remove 3rd data again
    EXPECT_EQ(alg_brute->cur_element_count, 0);
}