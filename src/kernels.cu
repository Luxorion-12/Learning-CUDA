#include <vector>
#include <cuda_fp16.h>
#include <math_constants.h>

#include "../tester/utils.h"

namespace {

constexpr int kThreadsPerBlock = 256;

__device__ __forceinline__ float toFloat(float value) {
  return value;
}

__device__ __forceinline__ float toFloat(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float value);

template <>
__device__ __forceinline__ float fromFloat<float>(float value) {
  return value;
}

template <>
__device__ __forceinline__ half fromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__global__ void rmsNormKernel(const T* input, const T* weight, T* output,
                              size_t hidden_dim, float eps) {
  const size_t row = blockIdx.x;
  const size_t row_offset = row * hidden_dim;

  float square_sum = 0.0f;
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float value = toFloat(input[row_offset + column]);
    square_sum += value * value;
  }

  __shared__ float reduction[kThreadsPerBlock];
  reduction[threadIdx.x] = square_sum;
  __syncthreads();

  for (unsigned int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      reduction[threadIdx.x] += reduction[threadIdx.x + offset];
    }
    __syncthreads();
  }

  const float inverse_rms =
      rsqrtf(reduction[0] / static_cast<float>(hidden_dim) + eps);
  for (size_t column = threadIdx.x; column < hidden_dim;
       column += blockDim.x) {
    const float normalized = toFloat(input[row_offset + column]) * inverse_rms;
    output[row_offset + column] =
        fromFloat<T>(normalized * toFloat(weight[column]));
  }
}

template <typename T>
__global__ void flashAttentionKernel(
    const T* query, const T* key, const T* value, float* accumulator,
    T* output, size_t attention_rows, int target_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  const size_t attention_row =
      static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (attention_row >= attention_rows) {
    return;
  }

  const int query_head = attention_row % query_heads;
  const size_t token_index = attention_row / query_heads;
  const int target_index = token_index % target_seq_len;
  const int batch_index = token_index / target_seq_len;
  const int group_size = query_heads / kv_heads;
  const int kv_head = query_head / group_size;

  const size_t output_base =
      ((static_cast<size_t>(batch_index) * target_seq_len + target_index) *
           query_heads +
       query_head) *
      head_dim;
  const size_t query_base = output_base;

  // 常见注意力头维度不超过 256，此时将 Q 和 O 保存在局部存储中；
  // 更大的头维度回退到全局显存，保证功能完整。
  constexpr int kLocalHeadLimit = 256;
  float local_query[kLocalHeadLimit];
  float local_output[kLocalHeadLimit];
  const bool use_local_storage = head_dim <= kLocalHeadLimit;
  for (int dimension = 0; dimension < head_dim; ++dimension) {
    if (use_local_storage) {
      local_query[dimension] = toFloat(query[query_base + dimension]);
      local_output[dimension] = 0.0f;
    } else {
      accumulator[output_base + dimension] = 0.0f;
    }
  }

  const int valid_sources =
      is_causal ? min(src_seq_len, target_index + 1) : src_seq_len;
  const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
  float max_score = -CUDART_INF_F;

  // 第一遍扫描：按维度顺序使用 FMA 计算点积，并求出最大分数。
  for (int source_index = 0; source_index < valid_sources; ++source_index) {
    const size_t key_base =
        ((static_cast<size_t>(batch_index) * src_seq_len + source_index) *
             kv_heads +
         kv_head) *
        head_dim;
    float dot = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      const float query_value = use_local_storage
                                    ? local_query[dimension]
                                    : toFloat(query[query_base + dimension]);
      dot = fmaf(query_value, toFloat(key[key_base + dimension]), dot);
    }
    max_score = fmaxf(max_score, dot * scale);
  }

  // 第二遍扫描：使用相同的点积顺序计算稳定 Softmax 的归一化因子。
  float normalizer = 0.0f;
  for (int source_index = 0; source_index < valid_sources; ++source_index) {
    const size_t key_base =
        ((static_cast<size_t>(batch_index) * src_seq_len + source_index) *
             kv_heads +
         kv_head) *
        head_dim;
    float dot = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      const float query_value = use_local_storage
                                    ? local_query[dimension]
                                    : toFloat(query[query_base + dimension]);
      dot = fmaf(query_value, toFloat(key[key_base + dimension]), dot);
    }
    normalizer += expf(dot * scale - max_score);
  }

  const float inverse_normalizer =
      normalizer == 0.0f ? 0.0f : 1.0f / normalizer;

  // 第三遍扫描：重新计算分数和概率，并累加 P 与 V 的乘积。
  for (int source_index = 0; source_index < valid_sources; ++source_index) {
    const size_t key_base =
        ((static_cast<size_t>(batch_index) * src_seq_len + source_index) *
             kv_heads +
         kv_head) *
        head_dim;
    float dot = 0.0f;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      const float query_value = use_local_storage
                                    ? local_query[dimension]
                                    : toFloat(query[query_base + dimension]);
      dot = fmaf(query_value, toFloat(key[key_base + dimension]), dot);
    }
    const float probability =
        expf(dot * scale - max_score) * inverse_normalizer;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      if (use_local_storage) {
        local_output[dimension] =
            fmaf(probability, toFloat(value[key_base + dimension]),
                 local_output[dimension]);
      } else {
        const size_t index = output_base + dimension;
        accumulator[index] =
            fmaf(probability, toFloat(value[key_base + dimension]),
                 accumulator[index]);
      }
    }
  }

  for (int dimension = 0; dimension < head_dim; ++dimension) {
    if (use_local_storage) {
      output[output_base + dimension] = fromFloat<T>(local_output[dimension]);
    } else {
      output[output_base + dimension] =
          fromFloat<T>(accumulator[output_base + dimension]);
    }
  }
}

}  // 匿名命名空间

/**
 * @brief 对二维张量的最后一个维度执行 RMSNorm。
 *
 * 输入为形状 [rows, hidden_dim] 的行优先矩阵，每行独立计算：
 * output[i, j] = input[i, j] * rsqrt(mean(input[i, :]^2) + eps) * weight[j]
 *
 * @tparam T 输入、权重和输出的数据类型。
 * @param[in] h_input 展平后的输入矩阵，形状为 [rows, hidden_dim]。
 * @param[in] h_weight 每一列的缩放权重，形状为 [hidden_dim]。
 * @param[out] h_output 展平后的输出矩阵，形状为 [rows, hidden_dim]。
 * @param[in] rows 输入矩阵的行数。
 * @param[in] hidden_dim 归一化维度的大小。
 * @param[in] eps 数值稳定项。
 */
template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
              std::vector<T>& h_output, size_t rows, size_t hidden_dim,
              float eps) {
  const size_t element_count = rows * hidden_dim;
  h_output.resize(element_count);
  if (element_count == 0) {
    return;
  }

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  const size_t matrix_bytes = element_count * sizeof(T);
  const size_t weight_bytes = hidden_dim * sizeof(T);

  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_input), matrix_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_weight), weight_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), matrix_bytes));
  RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), matrix_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_bytes,
                           cudaMemcpyHostToDevice));

  rmsNormKernel<T><<<rows, kThreadsPerBlock>>>(
      d_input, d_weight, d_output, hidden_dim, eps);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_output.data(), d_output, matrix_bytes,
                           cudaMemcpyDeviceToHost));

  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_weight));
  RUNTIME_CHECK(cudaFree(d_input));
}

/**
 * @brief 计算查询、键和值张量的 Flash Attention。
 *
 * @tparam T 输入和输出张量的数据类型。
 * @param[in] h_q 查询张量，形状为 [batch_size, tgt_seq_len, query_heads, head_dim]。
 * @param[in] h_k 键张量，形状为 [batch_size, src_seq_len, kv_heads, head_dim]。
 * @param[in] h_v 值张量，形状为 [batch_size, src_seq_len, kv_heads, head_dim]。
 * @param[out] h_o 输出张量，形状为 [batch_size, tgt_seq_len, query_heads, head_dim]。
 * @param[in] batch_size 批次大小。
 * @param[in] target_seq_len 目标序列长度。
 * @param[in] src_seq_len 源序列长度。
 * @param[in] query_heads 查询头数量。
 * @param[in] kv_heads 键和值的头数量，支持 GQA。
 * @param[in] head_dim 每个注意力头的维度。
 * @param[in] is_causal 是否使用因果掩码。
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {       
  const size_t query_element_count =
      static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t kv_element_count =
      static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  h_o.resize(query_element_count);
  if (query_element_count == 0) {
    return;
  }
  if (kv_heads <= 0 || query_heads % kv_heads != 0 || src_seq_len <= 0 ||
      head_dim <= 0) {
    std::cerr << "Invalid flashAttention dimensions" << std::endl;
    exit(EXIT_FAILURE);
  }

  T* d_query = nullptr;
  T* d_key = nullptr;
  T* d_value = nullptr;
  T* d_output = nullptr;
  float* d_accumulator = nullptr;
  const size_t query_bytes = query_element_count * sizeof(T);
  const size_t kv_bytes = kv_element_count * sizeof(T);
  const size_t accumulator_bytes = query_element_count * sizeof(float);

  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_query), query_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_key), kv_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_value), kv_bytes));
  RUNTIME_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output), query_bytes));
  if (head_dim > 256) {
    RUNTIME_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&d_accumulator), accumulator_bytes));
  }
  RUNTIME_CHECK(cudaMemcpy(d_query, h_q.data(), query_bytes,
                           cudaMemcpyHostToDevice));
  RUNTIME_CHECK(
      cudaMemcpy(d_key, h_k.data(), kv_bytes, cudaMemcpyHostToDevice));
  RUNTIME_CHECK(
      cudaMemcpy(d_value, h_v.data(), kv_bytes, cudaMemcpyHostToDevice));

  const size_t attention_rows = static_cast<size_t>(batch_size) *
                                target_seq_len * query_heads;
  const size_t block_count =
      (attention_rows + kThreadsPerBlock - 1) / kThreadsPerBlock;
  flashAttentionKernel<T><<<block_count, kThreadsPerBlock>>>(
      d_query, d_key, d_value, d_accumulator, d_output, attention_rows,
      target_seq_len, src_seq_len, query_heads, kv_heads, head_dim, is_causal);
  RUNTIME_CHECK(cudaGetLastError());
  RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_output, query_bytes,
                           cudaMemcpyDeviceToHost));

  if (d_accumulator != nullptr) {
    RUNTIME_CHECK(cudaFree(d_accumulator));
  }
  RUNTIME_CHECK(cudaFree(d_output));
  RUNTIME_CHECK(cudaFree(d_value));
  RUNTIME_CHECK(cudaFree(d_key));
  RUNTIME_CHECK(cudaFree(d_query));
}

// *********************************************************************
// 显式模板实例化（与 tester.o 链接所必需）
// 请勿修改此区域
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
