# PyVideoProc_RK3588

<p>
  <img alt="System" src="https://img.shields.io/badge/System-Ubuntu_22.x-E95420?logo=ubuntu&logoColor=white&style=flat-square">
  <img alt="Kernel" src="https://img.shields.io/badge/Linux_Kernel-6.x-00add8?style=flat-square&logo=linux&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white&style=flat-square">
  <img alt="Toolkit" src="https://img.shields.io/badge/rknn__toolkit2-2.3.2-7c4f9e?style=flat-square">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-2ea94f?logo=mitlicense&logoColor=white&style=flat-square">
</p>

用 Python 在 RK3588 上高效实现多路、多模型的视频流解码推理跟踪分析，并显著降低开发复杂度

https://github.com/user-attachments/assets/38396ca8-7ee7-4e8e-a025-94d1653ef343

核心特性：

- MPP 硬解码加速
- RGA 硬件转色与缩放
- dma_buf 全链路零拷贝、缓冲区池化复用
- RKNRKNN 运行时直调，无需 rknn-toolkit
- 目标跟踪批量向量化加速
- 工程实现极简，二次开发开箱即用

配套视频：[BiliBili](https://www.bilibili.com/video/BV1bRnJ6fEYJ)

🔥🔥🔥 CUDA 版本实现：[PyVideoProc](https://github.com/lmk123568/PyVideoProc)

## Usage

> 💡 在另外 x86 linux 环境导出 rknn 模型
>
> 将训练的 yolo26 onnx 模型导出 rknn 模型，输入尺寸设置 352x640，以接近视频帧的高宽比 9:16
> 
> ```bash
> pip install rknn-toolkit2
> python assets/convert_rknn.py yolo26n_352x640.onnx
> ```
>
> 本项目已包含官方权重转换的 rknn 模型

RK3588 开发板所需系统为 ubuntu 22.x，内核版本为 6.x，推荐镜像包下载：[百度网盘](https://pan.baidu.com/s/1wQJADWox-dDGDBWesU4PBA?pwd=gvbg)

搭建环境

```bash
# 安装 mpp rga 驱动
sudo add-apt-repository ppa:liujianfeng1994/rockchip-multimedia
sudo apt update
sudo apt install git build-essential ffmpeg pkg-config librockchip-mpp-dev librga-dev librga2 -y
# 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env 
```

克隆本仓库

```bash
git clone https://github.com/lmk123568/PyVideoProc_RK.git
```

开启编译

```bash
cd PyVideoProc_RK
# 安装依赖
uv sync
# 编译 c++
uv run cmake -S core -B core/build
uv run cmake --build core/build
# 开启性能模式
bash set_max_perf.sh
```

测试 yolo26 模型

```bash
uv run python yolo26_rk.py
```

测试多路视频解码推理

打开`main.py`，修改添加视频流

```bash
# 运行
uv run python main.py
```

## Benchmark

路数均分给 3 个 npu 核

- 测试模型 `assets/yolo26n_352x640_fp16.rknn`
- 测试视频流 `h264 1980x1280@25fps`
- RK3588 开最高性能

|      | CPU  | NPU                                | RAM    | YOLO Infer     |
| ---- | ---- | ---------------------------------- | ------ | -------------- |
| 3路  | 2%   | Core0: 9%, Core1: 10%, Core2: 10%  | 0.9 GB | 35 ms / frame  |
| 6路  | 8%   | Core0: 23%, Core1: 23%, Core2: 24% | 1.0 GB | 53 ms / frame  |
| 9路  | 12%  | Core0: 33%, Core1: 34%, Core2: 34% | 1.2 GB | 74 ms / frame  |
| 12路 | 16%  | Core0: 48%, Core1: 50%, Core2: 51% | 1.5 GB | 98 ms / frame  |

## License
[MIT](./LICENSE)
