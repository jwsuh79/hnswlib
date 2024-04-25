#pragma once

#include "hnswlib.h"
#include <unordered_set>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;


class LinkLists {
  
    //std::vector<LinkData<M>> link_lists;

    char** link_lists {nullptr};
    size_t size_links_per_element {0};
    size_t size_links_per_element0 {0};  // for level0 links data size which is doulbed.
    std::atomic<size_t> cur_element_cnt {0};
    
    public:
    LinkLists(size_t max_elements, size_t M) {
        // data structure: link size : (link, link, ..., link)
        size_links_per_element = M * sizeof(tableint) + sizeof(linklistsizeint);
        size_links_per_element0 = M * 2 * sizeof(tableint) + sizeof(linklistsizeint);
        link_lists = (char**) malloc(sizeof(void*) * max_elements);

        if (link_lists == nullptr) {
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        }
    }

    ~LinkLists() {
        for (tableint i = 0; i < cur_element_cnt; ++i) {
            free(link_lists[i]);
        }
        free(link_lists);
        link_lists = nullptr;
        cur_element_cnt = 0;
    }

    linklistsizeint* get_link_list(tableint internal_id, int level) {
        if (level==0) {
            return (linklistsizeint*) (link_lists[internal_id]);
        } else {
            // level0 links (x2) : level1 links : level2 links : ... : levelN links
            return (linklistsizeint*) (link_lists[internal_id] + size_links_per_element0 + (level-1) * size_links_per_element);
        }
    }

    void reallocate(size_t max_elements) {
        char **tmp = (char**) realloc(link_lists, sizeof(void*) * max_elements);
        if (tmp == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        link_lists = tmp;
    }

    void reserve_link(tableint cur_c, int curlevel) {
        size_t links_size = 0;
        if (curlevel==0)
            links_size = size_links_per_element0;
        else
            links_size = size_links_per_element0 + (curlevel-1) * size_links_per_element + 1;

        link_lists[cur_c] = (char*) malloc(links_size);
        if (link_lists[cur_c] == nullptr)
            throw std::runtime_error("Not enough memory: addPoint_ failed to allocate linklist");
        memset(link_lists[cur_c], 0, links_size);
    }

    void set_links_size(linklistsizeint *link_ptr, unsigned short int size) const {
        *((unsigned short int*)(link_ptr)) = size;
    }

    unsigned short int get_links_size(linklistsizeint *link_ptr) const {
        return *((unsigned short int*) link_ptr);
    }


    void connect_links(tableint internal_id, int level, std::vector<tableint>& neighbors, std::vector<int>& element_levels, bool is_update) {
        linklistsizeint *ll_cur = get_link_list(internal_id, level);
        *((unsigned short int*)(ll_cur)) = neighbors.size();
        tableint *link_ptr = (tableint*) (ll_cur+1);  // skip the link size
        for (size_t idx = 0; idx < neighbors.size(); ++idx) {
            if(link_ptr[idx] && !is_update)
                throw std::runtime_error("Possbile memory corruption");
            if (level > element_levels[neighbors[idx]])
                throw std::runtime_error("Trying to make a link on a non-existent level");
            link_ptr[idx] = neighbors[idx];
        }
    }

};

}  // namespace hnswlib
