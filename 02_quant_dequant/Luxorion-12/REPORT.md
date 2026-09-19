# 题目二实验报告：MXFP8 / NVFP4 软件模拟与反量化

## 1. 任务目标

在没有原生 FP8/FP4 指令的普通 CUDA GPU 上，实现 MXFP8 和 NVFP4 的软件量化与反量化，包括低精度编码、scale 计算、打包/解包、误差评估和性能测量。CPU 实现用于明确格式语义并作为 GPU 正确性 oracle，主要计算路径由 CUDA kernel 完成。

## 2. 格式与分块约定

### MXFP8

- 每 32 个连续元素组成一个量化块。
- 每块共享一个 E8M0 scale。
- 每个元素保存为一个 E4M3 编码字节。
- 量化负载为 `N + ceil(N/32)` 字节。

### NVFP4

- 整个张量共享一个 FP32 global scale。
- 每 16 个连续元素共享一个 E4M3 local scale。
- 元素保存为 E2M1，一个字节打包两个元素。
- 尾块固定占 8 个 packed 字节，无效半字节补零。

两种格式都采用行主序连续展平分块，块边界不因换行而重置。这保证任意形状都能用相同的一维地址映射处理。

## 3. 实现设计

### CPU reference

CPU 部分提供 E2M1、E4M3、E8M0 编解码、FP4 打包、两种格式的量化/反量化、tensor 文件读写以及 MaxAbs、MAE、MSE、压缩率。它强调格式规则完整和边界检查，不承担性能目标。

### MXFP8 CUDA baseline

量化时，一个 32 线程 CUDA block 对应一个 MXFP8 块：线程加载一个元素，共享内存归约 `amax`，线程 0 选择 E8M0 scale，然后每个有效线程编码一个 E4M3 元素。反量化时，一个线程恢复一个元素，并通过 `i / 32` 读取所属 scale。

### NVFP4 CUDA baseline

先用两级 GPU 归约求整个输入的 `amax` 和 global scale。随后一个 16 线程 CUDA block 对应一个 NVFP4 块，归约 local `amax`、生成 E4M3 local scale，并将相邻两个 E2M1 编码打包。反量化时，一个线程恢复一个有效元素，通过 `i / 16` 定位 local scale、通过 `i / 2` 定位 packed 字节。

## 4. 输入数据

固定输入均为 `1024 × 1024`、FP32、seed 1234：

| 名称 | 生成方式 | 目的 |
|---|---|---|
| uniform | `U(-6000, 6000)` | 覆盖较宽动态范围 |
| normal | `N(0, 20)` | 模拟主体集中在零附近的数据 |
| outliers | normal 基础上每 1000 个元素交替写入 `±6000` | 检查离群值对共享 scale 的影响 |

文件使用固定 20 字节 header，正式参数及文件名记录在 `input/benchmark/manifest.json`。

## 5. 正确性验证

每组 GPU benchmark 完成后均重新运行 CPU reference，并比较：

1. MXFP8 的所有 E8M0 scale 与 E4M3 编码；
2. NVFP4 的 global scale、所有 local scale 与 packed data；
3. 两种格式的全部恢复 FP32 值及零的符号。

三种输入、两种格式共六组结果全部与 CPU oracle 一致。正式整理前，原工程的 CPU/GPU 回归测试为 7/7 通过。

## 6. 性能协议

- 每个计时项目先预热 10 次，再独立测量 100 次。
- 报告 median 和 nearest-rank p95；median 表示典型性能，p95 反映较慢尾部。
- kernel 和 device pipeline 使用默认 stream 上的 CUDA event。
- end-to-end 使用 host `steady_clock`，包含同步 H2D、kernel、同步 D2H。
- 所有计时均排除显存分配、文件 IO、CPU 校验和误差统计。
- `device_pipeline` 中数据始终驻留显存，用于隔离 PCIe 传输影响。

## 7. 实验环境

| 项目 | 配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU |
| Compute Capability | 8.6 |
| CUDA Toolkit / nvcc | 12.8 / 12.8.93 |
| CUDA Runtime | 12.8 (`12080`) |
| CUDA Driver API | 13.0 (`13000`) |
| CMake | 3.31.6-msvc6 |
| C++/CUDA 标准 | C++17 / CUDA C++17 |
| 构建类型 | Release |

## 8. 正式结果

### 性能

单位均为毫秒；数值来自 `output/gpu_baseline/baseline_results.csv`。

| 数据 | 格式 | quant median / p95 | dequant median / p95 | device pipeline median / p95 | end-to-end median / p95 |
|---|---|---:|---:|---:|---:|
| uniform | MXFP8 | 18.685 / 19.818 | 0.274 / 0.302 | 19.208 / 19.686 | 22.266 / 22.957 |
| uniform | NVFP4 | 26.298 / 27.699 | 0.249 / 0.268 | 26.477 / 27.303 | 29.650 / 31.074 |
| normal | MXFP8 | 18.188 / 20.121 | 0.273 / 0.294 | 18.742 / 19.730 | 21.866 / 22.900 |
| normal | NVFP4 | 26.177 / 27.367 | 0.250 / 0.275 | 26.599 / 27.036 | 29.781 / 30.439 |
| outliers | MXFP8 | 18.676 / 19.707 | 0.272 / 0.291 | 19.020 / 19.641 | 21.992 / 23.301 |
| outliers | NVFP4 | 25.775 / 27.947 | 0.252 / 0.283 | 26.212 / 27.305 | 29.525 / 30.618 |

### 误差与存储

| 数据 | 格式 | MaxAbs | MAE | MSE | 量化负载 | 压缩率 |
|---|---|---:|---:|---:|---:|---:|
| uniform | MXFP8 | 256.000 | 71.089 | 9416.274 | 1,081,344 B | 3.879× |
| uniform | NVFP4 | 999.993 | 265.315 | 123673.077 | 589,828 B | 7.111× |
| normal | MXFP8 | 3.995 | 0.360 | 0.282 | 1,081,344 B | 3.879× |
| normal | NVFP4 | 12.852 | 1.430 | 3.627 | 589,828 B | 7.111× |
| outliers | MXFP8 | 144.000 | 0.504 | 21.026 | 1,081,344 B | 3.879× |
| outliers | NVFP4 | 82.841 | 1.645 | 9.557 | 589,828 B | 7.111× |

MXFP8 的有效带宽约为：量化 0.28–0.29 GB/s、反量化 19.26–19.37 GB/s；NVFP4 分别约为 0.34–0.35 GB/s、18.99–19.23 GB/s。NVFP4 的量化流量将 global amax 和局部量化的两次输入读取都计入，MXFP8 按一次输入读取与编码/scale 写出统计。

## 9. 结果分析与局限

1. 当前主要瓶颈是量化编码，而不是反量化或数据传输。量化时间约为反量化的 70–105 倍。
2. baseline 编码器为保证规则清晰和 CPU/GPU 一致，会扫描可表示候选值；这适合作为正确性基线，但不是最终高性能实现。
3. NVFP4 获得约 7.11× 的负载压缩率，高于 MXFP8 的约 3.88×，代价是更大的量化误差。
4. outliers 会抬高共享 scale，并增加主体数据的误差；该现象在 NVFP4 中更明显。
