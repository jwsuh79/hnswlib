#pragma once

#include "hnswlib.h"
#include <unordered_set>

namespace hnswlib {
typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

class LabelLookup {
    mutable std::mutex label_lookup_lock;  // lock for label_lookup_ is mutable to pass the multiThreadLoad_test.cpp compilation
    std::unordered_map<labeltype, tableint> label_lookup_;

    public:
    void replace_label(tableint label_replaced, tableint new_label, tableint internal_id_replaced) {
        std::lock_guard<std::mutex> lock_table(label_lookup_lock);
        label_lookup_.erase(label_replaced);
        label_lookup_[new_label] = internal_id_replaced;    
    }

    bool find_label(tableint label) {
        std::lock_guard<std::mutex> lock_table(label_lookup_lock);
        auto found = label_lookup_.find(label);
        if (found != label_lookup_.end())
            return true;
        else
            return false;
    }

    void add_label(tableint label, tableint id) {
        std::lock_guard<std::mutex> lock_table(label_lookup_lock);
        label_lookup_[label] = id;
    }

    void add_label_without_lock(tableint label, tableint id) {
        label_lookup_[label] = id;
    }

    tableint get_id(tableint label) {
        std::lock_guard<std::mutex> lock_table(label_lookup_lock);  // TODO: Use readlock instead
        return label_lookup_[label];
    }

    tableint find_label_get_id(tableint label) const {
        // Assume the label exists in the label_lookup_
        std::lock_guard<std::mutex> lock_table(label_lookup_lock);
        auto found = label_lookup_.find(label);
        if (found != label_lookup_.end()) {
            return found->second;
        }
        
        throw std::runtime_error("Label not Found!");
    }
};
class DeletedElement {
    std::mutex deleted_elements_lock;
    std::unordered_set<tableint> deleted_elements_;  // contains internal ids of deleted elements

    public:
    bool empty() {
        std::lock_guard<std::mutex> lock(deleted_elements_lock);
        return deleted_elements_.empty();
    }

    tableint extract_replaced_id() {
        std::lock_guard<std::mutex> lock(deleted_elements_lock);
        tableint internal_id_replaced = *deleted_elements_.begin();
        deleted_elements_.erase(internal_id_replaced);
        return internal_id_replaced;
    }

    void add_deleted_id(tableint id) {
        std::lock_guard<std::mutex> lock(deleted_elements_lock);
        deleted_elements_.insert(id);
    }

    void add_deleted_id_without_lock(tableint id) {    
        deleted_elements_.insert(id);
    }

    void remove_deleted_id(tableint id) {
        std::lock_guard<std::mutex> lock(deleted_elements_lock);
        deleted_elements_.erase(id);
    }

};

}  // namespace hnswlib
