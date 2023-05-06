#ifndef _OOBLECK_PLANNING_EXECUTION_RESULT_H_
#define _OOBLECK_PLANNING_EXECUTION_RESULT_H_

#include <oneapi/tbb/concurrent_hash_map.h>
#include <map>
#include <memory>
#include <tuple>
#include <vector>

namespace oobleck {

class PipelineTemplate;

/**
 * Execution result of a layer.
 */
class LayerExecutionResult {
  friend class StageExecutionResult;
  friend class PipelineTemplate;

 public:
  LayerExecutionResult(int layer_index,
                       double forward,
                       double backward,
                       const std::map<int, double>& allreduce_in_node,
                       const std::map<int, double>& allreduce_cross_nodes,
                       const std::tuple<int, int>& mem_required)
      : layer_index_(layer_index),
        forward_(forward),
        backward_(backward),
        allreduce_in_node_(allreduce_in_node),
        allreduce_cross_nodes_(allreduce_cross_nodes),
        mem_required_(mem_required) {}

  int layer_index_;
  double forward_;
  double backward_;
  std::map<int, double> allreduce_in_node_;
  std::map<int, double> allreduce_cross_nodes_;
  std::tuple<int, int> mem_required_;
};

/**
 * Execution result of a stage.
 * Stage consists of multiple layers;
 * StageExecutionResult is the aggregation of LayerExecutionResults.
 */
class StageExecutionResult {
  friend class DCExecutionResult;

 public:
  StageExecutionResult(
      const std::shared_ptr<std::vector<LayerExecutionResult>> layer_results,
      const std::tuple<int, int>& layer_indices,
      int device_num)
      : device_num_(device_num) {
    int layer_start_index = std::get<0>(layer_indices);
    int layer_end_index = std::get<1>(layer_indices);
    assert(layer_end_index <= layer_results->size());

    for (int i = layer_start_index; i < layer_end_index; ++i) {
      layer_indices_.push_back((*layer_results)[i].layer_index_);
      forward_ += (*layer_results)[i].forward_;
      backward_ += (*layer_results)[i].backward_;

      if (device_num_ > 1) {
        forward_ += (*layer_results)[i].allreduce_in_node_.at(device_num_ - 1);
        backward_ += (*layer_results)[i].allreduce_in_node_.at(device_num_ - 1);
      }

      for (const auto& it : (*layer_results)[i].allreduce_cross_nodes_) {
        allreduce_cross_nodes_[it.first] += it.second;
      }
      mem_required_ += std::get<0>((*layer_results)[i].mem_required_) * 6;
      mem_required_ += std::get<1>((*layer_results)[i].mem_required_);
    }
  }

  int device_num() const { return device_num_; }
  int memory_consumption() const { return mem_required_ / device_num_; }
  int num_layers() const { return layer_indices_.size(); }
  std::string to_string() const {
    int first_layer_index = layer_indices_.front();
    int last_layer_index = layer_indices_.back();
    return "StageExecutionResult[" + std::to_string(first_layer_index) + ":" +
           std::to_string(last_layer_index) + "] with " +
           std::to_string(device_num_) + " devices";
  }

 private:
  int device_num_;
  std::vector<int> layer_indices_;
  double forward_;
  double backward_;
  std::map<int, double> allreduce_cross_nodes_;
  int mem_required_;
};

class DCExecutionResult {
 public:
  // # stage, start layer index, end layer index, num nodes, num GPUs per node
  using key = std::tuple<int, int, int, int, int>;

  class KeyCompare {
   public:
    std::size_t hash(const oobleck::DCExecutionResult::key& key) const {
      std::string string_key = std::to_string(std::get<0>(key)) + "." +
                               std::to_string(std::get<1>(key)) + "." +
                               std::to_string(std::get<2>(key)) + "." +
                               std::to_string(std::get<3>(key)) + "." +
                               std::to_string(std::get<4>(key));
      return std::hash<std::string>()(string_key);
    }
    bool equal(const oobleck::DCExecutionResult::key& key1,
               const oobleck::DCExecutionResult::key& key2) const {
      return std::get<0>(key1) == std::get<0>(key2) &&
             std::get<1>(key1) == std::get<1>(key2) &&
             std::get<2>(key1) == std::get<2>(key2) &&
             std::get<3>(key1) == std::get<3>(key2) &&
             std::get<4>(key1) == std::get<4>(key2);
    }
  };

  // Basic constructor
  DCExecutionResult(const StageExecutionResult& stage,
                    int num_nodes,
                    int num_gpus_per_node)
      : t1_(stage.forward_ + stage.backward_),
        t2_(2 * (stage.forward_ + stage.backward_)),
        t3_(stage.forward_ + stage.backward_),
        num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node),
        kstar_(0),
        stages_(std::make_shared<std::vector<StageExecutionResult>>()) {
    stages_->emplace_back(stage);
  }

  // Combine constructor
  DCExecutionResult(const std::shared_ptr<DCExecutionResult> left,
                    const std::shared_ptr<DCExecutionResult> right,
                    int num_nodes,
                    int num_gpus_per_node)
      : num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node),
        stages_(left->stages_) {
    assert(left != nullptr && right != nullptr && left->stages_ != nullptr &&
           right->stages_ != nullptr);

    kstar_ = left->get_kstar_latency() > right->get_kstar_latency()
                 ? left->kstar_
                 : right->kstar_ + left->stages_->size();
    t1_ = left->t1_ + right->t1_;
    int num_kstar_stage_microbatch =
        2 * (left->stages_->size() + right->stages_->size()) + kstar_ + 1;
    double latency = 0;
    if (kstar_ == left->kstar_) {
      t2_ = num_kstar_stage_microbatch * left->get_kstar_latency();
      for (int i = left->kstar_; i < left->stages_->size(); i++) {
        latency += (*left->stages_)[i].forward_ + (*left->stages_)[i].backward_;
      }
      for (int i = 0; i < right->stages_->size(); i++) {
        latency +=
            (*right->stages_)[i].forward_ + (*right->stages_)[i].backward_;
      }
    } else {
      t2_ = num_kstar_stage_microbatch * right->get_kstar_latency();
      for (int i = right->kstar_; i < right->stages_->size(); i++) {
        latency +=
            (*right->stages_)[i].forward_ + (*right->stages_)[i].backward_;
      }
    }
    t3_ = latency;

    std::cout << "size: " << left->stages_->size() << " "
              << right->stages_->size() << std::endl;
    stages_->insert(stages_->end(), right->stages_->begin(),
                    right->stages_->end());
  }

  double get_t() const { return t1_ + t2_ + t3_; }
  double get_kstar_latency() const {
    return (*stages_)[kstar_].forward_ + (*stages_)[kstar_].backward_;
  }
  std::shared_ptr<std::vector<StageExecutionResult>> get_stages() const {
    return stages_;
  }

 private:
  int kstar_;
  double t1_, t2_, t3_;
  int num_nodes_;
  int num_gpus_per_node_;
  const std::shared_ptr<std::vector<StageExecutionResult>> stages_;
};

}  // namespace oobleck

using CacheMap = oneapi::tbb::concurrent_hash_map<
    oobleck::DCExecutionResult::key,
    std::shared_ptr<oobleck::DCExecutionResult>,
    oobleck::DCExecutionResult::KeyCompare>;

#endif