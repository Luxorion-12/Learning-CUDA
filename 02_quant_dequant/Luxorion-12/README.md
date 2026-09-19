# Luxorion-12：MXFP8 / NVFP4 CUDA 软件模拟 baseline


## 已实现内容

- MXFP8：32 个连续元素共享一个 E8M0 scale，元素编码为 E4M3。
- NVFP4：全局 FP32 scale、每 16 个连续元素一个 E4M3 local scale，元素编码为 E2M1，两个元素打包为一个字节。
- CPU reference：提供格式规则、量化/反量化和误差计算的独立 oracle。
- CUDA baseline：完成两种格式的量化和反量化；NVFP4 的 global amax 也在 GPU 上归约。
- 文件输入：固定 20 字节 header，支持 FP32/FP16，data 按行主序存储。
- 性能评估：uniform、normal、outliers 三类固定输入；预热 10 次、重复 100 次，报告 median 和 p95。
- 正确性：每次正式 benchmark 后逐项比较 CPU/GPU 的 scale、编码、packed data、恢复值及零符号。

分块采用方案：先把行主序张量连续展平，再按下标分块；MXFP8 使用 `i / 32`，NVFP4 使用 `i / 16`，块可以跨越行边界。

## 目录结构

```text
Luxorion-12/
├─ cpu_reference/          CPU参考实现和公共数据契约
├─ gpu_baseline/           CUDA kernel、性能接口和运行工具
├─ input/benchmark/        固定实验输入与manifest
├─ output/gpu_baseline/    正式CSV/JSON结果
├─ CMakeLists.txt          CPU和CUDA构建入口
└─ REPORT.md               实验报告
```

依据项目总要求，提交目录没有携带 `build/`、临时冒烟结果或独立测试源码。正式 benchmark 自身保留了 CPU oracle 校验，因此运行性能实验时仍会验证数值正确性。

## 构建

要求：支持 C++17 的编译器、CMake 3.24+、CUDA Toolkit。默认使用当前 GPU 的 native architecture。

Linux/Ninja 或 Makefile：

```bash
cmake -S . -B build -DLUXORION_ENABLE_CUDA=ON
cmake --build build -j
```

Windows + Visual Studio：

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
  -T "cuda=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8" `
  -DLUXORION_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release --parallel
```

只检查 CPU reference 时可使用 `-DLUXORION_ENABLE_CUDA=OFF`。

## 运行

命令需要从本目录执行。仓库已经附带正式输入，也可以重新生成：

```powershell
.\build\Release\generate_gpu_baseline_inputs.exe input\benchmark
.\build\Release\gpu_baseline_benchmark.exe
```

Linux 或单配置生成器通常对应：

```bash
./build/generate_gpu_baseline_inputs input/benchmark
./build/gpu_baseline_benchmark
```

benchmark 的可选参数为：

```text
gpu_baseline_benchmark [input_dir] [output_dir] [warmup] [repeats]
```

默认结果写入：

- `output/gpu_baseline/baseline_results.csv`：便于制表与后续画图。
- `output/gpu_baseline/baseline_results.json`：包含设备、CUDA版本、协议语义和完整结果。

## 输入文件协议

每个 tensor 文件由固定 20 字节 header 和连续 data 构成：

```text
offset 0..7    num_rows: int64 little-endian
offset 8..15   num_cols: int64 little-endian
offset 16..19  dtype: 4-byte ASCII "fp32" 或 "fp16"
offset 20..    row-major tensor values
```

读取器严格检查维度、dtype、文件总长度以及 NaN/Inf。详细分布参数见 `input/benchmark/manifest.json`。

## 计时边界

- `quantize`、`dequantize`：CUDA event 统计对应 kernel 序列。
- `device_pipeline`：输入、量化结果和恢复结果均驻留显存，只统计 quantize + dequantize。
- `end_to_end`：host `steady_clock` 统计 H2D + quantize + dequantize + D2H。
- 显存分配、文件 IO、CPU oracle 和误差计算不进入上述计时。

当前 baseline 为清晰性使用候选值扫描编码器，所以量化明显慢于反量化。这是后续快速编码、warp shuffle、向量化等优化的稳定对照组。
