#ifndef _OOBLECK_PLANNING_EXECUTION_RESULT_H_
#define _OOBLECK_PLANNING_EXECUTION_RESULT_H_

#include <map>
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
      mem_required_ += std::get<0>(layer_results[i].mem_required_) * 6;
      mem_required_ += std::get<1>(layer_results[i].mem_required_);
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

  // Non-valid constructor
  DCExecutionResult()
      : t1_(0),
        t2_(0),
        t3_(0),
        kstar_(0),
        num_nodes_(0),
        num_gpus_per_node_(0) {}
  // Basic constructor
  DCExecutionResult(StageExecutionResult& stage,
                    int num_nodes,
                    int num_gpus_per_node)
      : t1_(stage.forward_ + stage.backward_),
        t2_(2 * (stage.forward_ + stage.backward_)),
        t3_(stage.forward_ + stage.backward_),
        num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node),
        kstar_(0),
        stages_({stage}) {}
  // Combine constructor
  DCExecutionResult(const DCExecutionResult& left,
                    const DCExecutionResult& right,
                    int num_nodes,
                    int num_gpus_per_node)
      : kstar_([&]() -> int {
          if (left.get_kstar_latency() > right.get_kstar_latency()) {
            return left.kstar_;
          } else {
            return right.kstar_ + left.stages_.size();
          }
        }()),
        t1_(left.t1_ + right.t1_),
        t2_([&]() {
          // 4 * num_stages - |T1| (1 * num_stages) - |T3| (2 * num_stages -
          // kstar - 1)
          int num_kstar_stage_microbatch =
              2 * (left.stages_.size() + right.stages_.size()) + kstar_ + 1;
          return num_kstar_stage_microbatch * (kstar_ == left.kstar_
                                                   ? left.get_kstar_latency()
                                                   : right.get_kstar_latency());
        }()),
        t3_([&]() {
          // all stages latency after kstar and kstar
          double latency = 0;
          if (kstar_ == left.kstar_) {
            for (int i = left.kstar_; i < left.stages_.size(); i++) {
              latency += left.stages_[i].forward_ + left.stages_[i].backward_;
            }
            for (int i = 0; i < right.stages_.size(); i++) {
              latency += right.stages_[i].forward_ + right.stages_[i].backward_;
            }
          } else {
            for (int i = right.kstar_; i < right.stages_.size(); i++) {
              latency += right.stages_[i].forward_ + right.stages_[i].backward_;
            }
          }
          return latency;
        }()),
        num_nodes_(num_nodes),
        num_gpus_per_node_(num_gpus_per_node),
        stages_(left.stages_) {
    stages_.insert(stages_.end(), right.stages_.begin(), right.stages_.end());
  }

  double get_t() const { return t1_ + t2_ + t3_; }
  double get_kstar_latency() const {
    if (stages_.size() == 0) {
      return 0;
    }
    return stages_[kstar_].forward_ + stages_[kstar_].backward_;
  }
  key get_key() const {
    if (stages_.size() == 0) {
      return std::make_tuple(0, 0, 0, 0, 0);
    }
    auto& last_stage = stages_[stages_.size() - 1];
    return std::make_tuple(
        stages_.size(), stages_[0].layer_indices_[0],
        last_stage.layer_indices_[last_stage.layer_indices_.size() - 1],
        num_nodes_, num_gpus_per_node_);
  }
  const std::vector<StageExecutionResult>& get_stages() const {
    return stages_;
  }

 private:
  int kstar_;
  double t1_, t2_, t3_;
  int num_nodes_;
  int num_gpus_per_node_;
  std::vector<StageExecutionResult> stages_;
};

}  // namespace oobleck

#endif