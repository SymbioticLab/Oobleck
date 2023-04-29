#ifndef _OOBLECK_PLANNING_EXECUTION_RESULT_H_
#define _OOBLECK_PLANNING_EXECUTION_RESULT_H_

#include <map>
#include <utility>
#include <vector>

namespace oobleck {

/**
 * Execution result of a layer.
 */
class LayerExecutionResult {
  friend class StageExecutionResult;

 public:
  LayerExecutionResult(int layer_index,
                       double forward,
                       double backward,
                       const std::map<int, double>& allreduce_in_node,
                       const std::map<int, double>& allreduce_cross_nodes,
                       const std::pair<int, int>& mem_required)
      : layer_index_(layer_index),
        forward_(forward),
        backward_(backward),
        allreduce_in_node_(allreduce_in_node),
        allreduce_cross_nodes_(allreduce_cross_nodes),
        mem_required_(mem_required) {}

 private:
  int layer_index_;
  double forward_;
  double backward_;
  std::map<int, double> allreduce_in_node_;
  std::map<int, double> allreduce_cross_nodes_;
  std::pair<int, int> mem_required_;
};

/**
 * Execution result of a stage.
 * Stage consists of multiple layers;
 * StageExecutionResult is the aggregation of LayerExecutionResults.
 */
class StageExecutionResult {
 public:
  StageExecutionResult(const std::vector<LayerExecutionResult>& layer_results,
                       int device_num)
      : device_num_(device_num), layer_indices_(layer_results.size()) {
    for (int i = 0; i < layer_results.size(); ++i) {
      layer_indices_[i] = layer_results[i].layer_index_;
      forward_ += layer_results[i].forward_;
      backward_ += layer_results[i].backward_;

      if (device_num_ > 1) {
        forward_ += layer_results[i].allreduce_in_node_.at(device_num_ - 1);
        backward_ += layer_results[i].allreduce_in_node_.at(device_num_ - 1);
      }

      for (const auto& it : layer_results[i].allreduce_cross_nodes_) {
        allreduce_cross_nodes_[it.first] += it.second;
      }
      mem_required_ += layer_results[i].mem_required_.first * 6;
      mem_required_ += layer_results[i].mem_required_.second;
    }
  }

  int device_num() const { return device_num_; }
  int memory_consumption() const { return mem_required_ / device_num_; }

 private:
  int device_num_;
  std::vector<int> layer_indices_;
  double forward_;
  double backward_;
  std::map<int, double> allreduce_cross_nodes_;
  int mem_required_;
};

}  // namespace oobleck

#endif