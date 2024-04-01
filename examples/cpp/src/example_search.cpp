#include <iostream>
#include "hnswlib/hnswlib.h"
#include "argh.h"


int main(int argc, char* argv[]) {
    int dim;                // Dimension of the elements
    int max_elements;       // Maximum number of elements, should be known beforehand
    int M;                  // Tightly connected with internal dimensionality of the data
                            // strongly affects the memory consumption
    int ef_construction;    // Controls index search speed/build speed tradeoff
    bool check_recall;

    auto cmdl = argh::parser(argc, argv, argh::parser::SINGLE_DASH_IS_MULTIFLAG);
    if (cmdl[{"-h", "--help"}]) {
        std::cout << "Usage: " << std::endl;
        std::cout << "--dim : the dimension of the elements, default 16" << std::endl;
        std::cout << "--max : the maximum number of elements, default 10000" << std::endl;
        std::cout << "--M : the internal dimensionality of the data, default 16" << std::endl;
        std::cout << "--ef : the size of the dynamic list for the nearest neighbors, default 200" << std::endl;
        std::cout << "--check_recall : perform the serialization/deserialization for the recall check, default disabled" << std::endl;
        return 0;
    }
    cmdl("dim", 16) >> dim;
    cmdl("max", 10000) >> max_elements;
    cmdl("M", 16) >> M;
    cmdl("ef", 200) >> ef_construction;
    check_recall = cmdl["check_recall"] ? true : false;  // flag argument


    std::cout << "Dim: " << dim << ", Max: " << max_elements << ", M: " << M << ", ef: " << ef_construction << ", check_recall: " << check_recall << std::endl;
    // Initing index
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        alg_hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    if (check_recall) {
        // Serialize index
        std::string hnsw_path = "hnsw.bin";
        alg_hnsw->saveIndex(hnsw_path);
        delete alg_hnsw;

        // Deserialize index and check recall
        alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, hnsw_path);
        correct = 0;
        for (int i = 0; i < max_elements; i++) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i) correct++;
        }
        recall = (float)correct / max_elements;
        std::cout << "Recall of deserialized index: " << recall << "\n";
    }

    delete[] data;
    delete alg_hnsw;
    return 0;
}
