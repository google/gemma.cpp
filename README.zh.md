# gemma.cpp

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

gemma.cpp 是针对 Google Gemma 系列基础模型的轻量级、独立的 C++ 推理引擎。

有关 Gemma 的更多信息，请访问 [ai.google.dev/gemma](https://ai.google.dev/gemma)。模型权重（包括 gemma.cpp 专用的构建产物）可在 [Kaggle](https://www.kaggle.com/models/google/gemma-3) 上获取。

## 这个项目适合谁？

现代大语言模型（LLM）推理引擎是非常复杂的系统，通常具备超越传统神经网络运行时的定制能力。这也为通过高层算法与底层计算的协同设计（co-design）进行前沿研究和创新带来了契机。然而，目前在面向生产部署的 C++ 推理运行时（通常并非为实验而设计）与以 Python 为中心的机器学习研究框架（往往通过编译将底层计算抽象化）之间存在断层。

gemma.cpp 提供了 Gemma-2、Gemma-3 与 PaliGemma-2 模型的极简实现，专注于简洁性与直接性，而非面面俱到的通用性。这一设计理念受到了 [ggml](https://github.com/ggerganov/ggml)、[llama.c](https://github.com/karpathy/llama2.c) 和 [llama.rs](https://github.com/srush/llama2.rs) 等垂直整合模型实现的启发。

gemma.cpp 主要面向实验和前沿研究场景。它旨在通过最小依赖轻松嵌入其他项目，并通过轻量级的核心实现（约 2K 行核心代码以及约 4K 行支持工具代码）便于开发者自由修改与定制。我们采用 [Google Highway](https://github.com/google/highway) 库以在 CPU 推理中充分利用可移植的 SIMD 加速。

对于面向生产环境的边缘端部署，我们建议通过 JAX、Keras、PyTorch 以及 Transformers 等 Python 框架的标准部署路径（[所有模型规格可见此处](https://www.kaggle.com/models/google/gemma-3)）。

## 参与贡献

热烈欢迎社区大小各类贡献。参与贡献的开发者请参阅 [DEVELOPERS.md](https://github.com/google/gemma.cpp/blob/main/DEVELOPERS.md)，也可以通过[此邀请链接加入 Discord](https://discord.gg/H5jCBAWxAe)。本项目遵循 [Google 开源社区准则](https://opensource.google.com/conduct/)。

> [!NOTE] 目前活跃的开发工作在 `dev` 分支上进行。请将 Pull Request 提交至 `dev` 分支，而非更为稳定的 `main` 分支。

## 核心特性包含什么？

-   **大语言模型（LLM）**

    -   纯 CPU 推理：支持 Gemma 2-3、PaliGemma 2。
    -   采样：支持 TopK 与温度（temperature）调节。

-   **计算优化**

    -   混合精度（fp8、bf16、fp32、fp64）GEMM 矩阵乘法：
        -   针对 BF16 指令设计，能够高效仿真。
        -   针对每个矩阵形状在运行时自动调优（autotune）7 个参数。
    -   直接整合到 GEMM 中的权重压缩：
        -   自定义 fp8 格式（具备 2..3 个尾数位，支持张量缩放）。
        -   同时支持 bf16、f32 以及非均匀 4-bit（NUQ）；易于扩展新格式。

-   **底层基础设施**

    -   SIMD：通过 Highway 实现单一代码库，并在运行时动态选择 ISA 指令集。
    -   张量并行（Tensor parallelism）：感知 CCX 架构的多插槽（multi-socket）线程池。
    -   磁盘 I/O：内存映射（mmap）或并行读取（启发式自动选择，支持用户覆盖）。
    -   自定义格式：支持前向与后向兼容的元数据序列化。
    -   从 Safetensors 格式的模型转换（暂未开源）。
    -   可移植性：支持 Linux、Windows、macOS。支持 CMake/Bazel。兼容任意主流 CPU。

-   **前端与接入层**

    -   具备流式输出能力的 C++ API，支持单请求与批量推理。
    -   基础交互式命令行应用程序。
    -   基础 Python 绑定（基于 pybind11）。

## 快速上手

### 系统要求

在开始之前，应确保系统已安装：

- [CMake](https://cmake.org/)
- [Clang C++ 编译器](https://clang.llvm.org/get_started.html)，至少支持 C++17。
- 用于解压 Kaggle 下载归档文件的 `tar` 工具。

在 Windows 上进行原生构建需要 Visual Studio 2022 Build Tools 以及可选的 Clang/LLVM C++ 前端（`clang-cl`）。可以通过 [`winget`](https://learn.microsoft.com/en-us/windows/package-manager/winget/) 在命令行中安装：

```sh
winget install --id Kitware.CMake
winget install --id Microsoft.VisualStudio.2022.BuildTools --force --override "--passive --wait --add Microsoft.VisualStudio.Workload.VCTools;installRecommended --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --add Microsoft.VisualStudio.Component.VC.Llvm.ClangToolset"
```

### 步骤 1：从 Kaggle 或 Hugging Face Hub 获取模型权重与分词器

访问 [Gemma-2 的 Kaggle 页面](https://www.kaggle.com/models/google/gemma-3/gemmaCpp) 并选择 `Model Variations |> Gemma C++`。

在该标签页中，`Variation` 下拉菜单包含以下选项。请注意，bfloat16 权重的保真度更高，而 8 位切换浮点（switched floating point, sfp）权重能提供更快的推理速度。通常建议从 `-sfp` 权重检查点开始上手。

> [!NOTE] **重要提示**：强烈建议首先使用 `gemma2-2b-it-sfp` 模型快速运行起来。

Gemma 2 模型的命名规则为：2B 对应 `gemma2-2b-it`，以及 `9b-it` 或 `27b-it`。参见 `configs.cc` 中的 `ModelPrefix` 函数。

### 步骤 2：解压文件

填写许可协议表单后即可下载 tar 归档文件 `archive.tar.gz`。解压文件（可能需要几分钟）：

```
tar -xf archive.tar.gz
```

解压后将生成模型权重文件（如 `2b-it-sfp.sbs`）以及分词器文件（`tokenizer.spm`）。建议将这些文件移动到方便的目录位置（例如本仓库中的 `build/` 目录）。

### 步骤 3：构建

构建系统采用 [CMake](https://cmake.org/)。要构建 gemma 推理运行时，请在项目根目录下创建 build 目录并使用 `cmake` 生成构建文件。注意：如果之前运行过 `cmake` 并需要使用不同设置重新运行，请务必使用 `rm -rf build/*` 删除 `build/` 目录中的所有文件。

#### 类 Unix 平台

```sh
cmake -B build
```

运行 `cmake` 之后，可以进入 `build/` 目录并运行 `make` 来编译 `./gemma` 可执行文件：

```sh
# 配置 `build` 目录
cmake --preset make

# 使用 make 构建项目
cmake --build --preset make -j [并行编译线程数]
```

将 `[并行编译线程数]` 替换为具体数字（通常系统可用的 CPU 核心数是一个合理的参考）。例如，`make -j4 gemma` 将使用 4 个线程进行构建。如果系统支持 `nproc` 命令，可以使用 `make -j$(nproc) gemma` 作为默认线程数。

如果不确定 `-j` 参数的合适数值，可以直接运行 `make gemma`，同样可以构建出 `./gemma` 可执行文件。

> [!NOTE]
> 在适用于 Linux 的 Windows 子系统（WSL）上，用户应将并行线程数设置为 1。使用更大的数值可能会导致错误。

构建成功后，`build/` 目录下将生成 `gemma` 可执行文件。

#### Windows

```sh
# 配置 `build` 目录
cmake --preset windows

# 使用 Visual Studio Build Tools 构建项目
cmake --build --preset windows -j [并行编译线程数]
```

构建成功后，`build/` 目录下将生成 `gemma.exe` 可执行文件。

#### Bazel

```sh
bazel build -c opt --cxxopt=-std=c++20 :gemma
```

构建成功后，`bazel-bin/` 目录下将生成 `gemma` 可执行文件。

#### Make

如果更喜欢使用 Makefile，@jart 在此提供了一个版本：

https://github.com/jart/gemma3/blob/main/Makefile

### 步骤 4：运行

现在可以在 `build/` 目录下运行 `gemma`。

`gemma` 具有以下必选参数：

参数 | 说明 | 示例值
------------- | ---------------------------- | ---------------
`--weights`   | 压缩权重文件路径。 | `2b-it-sfp.sbs`
`--tokenizer` | 分词器文件路径。 | `tokenizer.spm`

以下配置的运行示例：

-   权重文件 `gemma2-2b-it-sfp.sbs`（Gemma2 2B 指令微调模型，8 位切换浮点）。
-   分词器文件 `tokenizer.spm`（对于 2025-05-06 之后创建的单文件格式权重，或由 migrate_weights.cc 输出的权重，可省略此参数）。

```sh
./gemma \
--tokenizer tokenizer.spm --weights gemma2-2b-it-sfp.sbs
```

### PaliGemma 视觉-语言模型（VLM）

本仓库包含 PaliGemma 2 VLM 的实现版本（[论文](https://arxiv.org/abs/2412.03555)）。我们在此提供了 PaliGemma 2 模型的 C++ 实现。

要使用本仓库中的 PaliGemma，请按照上文步骤 3 构建 gemma 二进制文件。从 [Kaggle](https://www.kaggle.com/models/google/paligemma-2/gemmaCpp/paligemma2-3b-mix-224) 下载压缩权重和分词器，然后运行可执行文件：

```sh
./gemma \
--tokenizer paligemma_tokenizer.model \
--weights paligemma2-3b-mix-224-sfp.sbs \
--image_file paligemma/testdata/image.ppm
```

请注意，图像读取代码目前非常基础，以避免引入图像处理库依赖。目前仅支持读取二进制 PPM 格式（P6）。因此可以使用 `convert` 等工具先将图像转换为该格式，例如：

`convert image.jpeg -resize 224x224^ image.ppm`

（由于图像在处理过程中无论如何都会被缩放，在此阶段提前调整大小可略微加快加载速度。）

使用 mix-224 检查点与图像交互的示例如下：

```
> Describe the image briefly
A large building with two towers in the middle of a city.
> What type of building is it?
church
> What color is the church?
gray
> caption image
A large building with two towers stands tall on the water's edge. The building
has a brown roof and a window on the side. A tree stands in front of the
building, and a flag waves proudly from its top. The water is calm and blue,
reflecting the sky above. A bridge crosses the water, and a red and white boat
rests on its surface. The building has a window on the side, and a flag on top.
A tall tree stands in front of the building, and a window on the building is
visible from the water. The water is green, and the sky is blue.
```

### T5Gemma 编码器-解码器模型

本仓库包含对 T5Gemma S/S 编码器-解码器模型的实验性支持。可使用以下命令将本地 Hugging Face safetensors 检查点转换为 SBS 格式：

```sh
python3 python/convert_from_safetensors.py \
--model_specifier=t5gemma-s-s \
--load_path /path/to/t5gemma/model.safetensors \
--tokenizer_file /path/to/t5gemma/tokenizer.model \
--sbs_file t5gemma-s-s-it.sbs \
--metadata_file t5gemma-s-s-it.csv
```

默认的 T5Gemma 转换将写入全量 BF16 SBS 文件，这有助于进行 Hugging Face 一致性校验，也是当前推荐的途径。如需生成更小的混合 BF16/SFP 文件，可添加 `--t5gemma_weight_type=sfp` 参数。

然后运行：

```sh
./gemma \
--tokenizer /path/to/t5gemma/tokenizer.model \
--weights t5gemma-s-s-it.sbs \
--model t5gemma-s-s \
--prompt "Hello"
```

首个支持的运行时路径是全新的 seq2seq 生成。针对新编码器输入的解码器 KV 缓存多轮复用暂不支持。指令微调版 T5Gemma 检查点默认使用指令包装。对于基础/预训练检查点或原始 Hugging Face Token 对齐校验，可传入 `--wrapping=0` 使用预训练包装。

### 迁移至单文件格式

权重文件现在支持全新的单文件格式，可以直接包含分词器（以及模型类型）。本仓库提供了将多文件格式迁移至单文件格式的工具：

```sh
io/migrate_weights \
  --tokenizer .../tokenizer.spm --weights .../gemma2-2b-it-sfp.sbs \
  --output_weights .../gemma2-2b-it-sfp-single.sbs
```

迁移后即可省略分词器参数：

```sh
./gemma --weights .../gemma2-2b-it-sfp-single.sbs
```

### 故障排查与常见问题解答（FAQ）

**在 Windows / Visual Studio 中编译时遇到问题**

目前如果使用 Windows 系统，建议在 WSL（适用于 Linux 的 Windows 子系统）中进行编译。我们正在探索支持其他构建配置的方法，详情可参阅 Issue 中的活跃讨论。

**模型不听从指令并产生异常输出**

常见原因是使用了预训练基础模型，该模型未经过指令微调，因此不会听从指令交互。请确保使用的是指令微调模型（`gemma2-2b-it-sfp`），而非预训练模型（带有 `-pt` 后缀的模型）。

**支持多长的序列长度？**

请参见 `configs.cc` 中的 `max_seq_len` 和 `InferenceArgs.seq_len`。对于大于 1B 的 Gemma 3 模型，通常为 32K，但在内存足够的情况下 128K 也可以工作。请注意，由于注意力的二次方开销，超长序列的处理速度会变慢。

**如何将微调后的权重转换为 `.sbs` 压缩模型文件？**

对于 PaliGemma 2 检查点，可以使用 python/convert_from_safetensors.py 从 safetensors 格式转换（已通过 bazel 构建验证）。对于 Adapter 适配器模型，在转换前可能需要调用 `merge_and_unload()` 将适配器模型合并为单文件格式。

假设本地（虚拟环境）已安装 torch、numpy、safetensors、absl-py 等依赖，以下是使用 bazel 构建压缩库并运行转换的示例：

```sh
bazel build //compression/python:compression
BAZEL_OUTPUT_DIR="${PWD}/bazel-bin/compression"
python3 -c "import site; print(site.getsitepackages())"
# 在此处软链接到你的 site-packages 目录：
ln -s $BAZEL_OUTPUT_DIR [...]/site-packages/compression
python3 python/convert_from_safetensors.py --load_path [...].safetensors.index.json
```

**有哪些简单的方法可以提高模型运行速度？**

1.  确保使用的是 8 位切换浮点 `-sfp` 模型。它们的大小只有 bf16 的一半，因此占用的内存带宽和缓存空间更小。
2.  得益于自动调优（auto-tuning），第二条尤其是第三条请求的运行速度会显著提升。
3.  如果是笔记本电脑，请确保电源模式设置为高性能，且省电模式处于**关闭**状态。大多数笔记本电脑在未接通电源时会自动开启省电模式。
4.  关闭其他未使用的 CPU 密集型应用程序。
5.  在 Mac 电脑上，经验观察表明随着性能核心的介入，会有一个速度“预热”提升的过程。

我们也在持续探索算法和系统优化手段以提升推理速度，敬请关注。

## 使用方式

`gemma` 支持不同的使用模式，由详细程度（verbosity）参数控制。

目前所有使用模式均为交互式，输入换行即可触发文本生成。

| 详细程度 | 使用模式 | 说明 |
| --------------- | ---------- | --------------------------------------------- |
| `--verbosity 0` | 极简模式 | 仅打印生成结果。适合作为命令行管道工具使用。 |
| `--verbosity 1` | 默认模式 | 面向用户的标准终端界面。 |
| `--verbosity 2` | 详细模式 | 显示额外的开发者和调试信息。 |

### 交互式终端应用

默认情况下，详细程度设置为 1，调用 `gemma` 时会调出基于终端的交互界面：

```sh
$ ./gemma [...]
  __ _  ___ _ __ ___  _ __ ___   __ _   ___ _ __  _ __
 / _` |/ _ \ '_ ` _ \| '_ ` _ \ / _` | / __| '_ \| '_ \
| (_| |  __/ | | | | | | | | | | (_| || (__| |_) | |_) |
 \__, |\___|_| |_| |_|_| |_| |_|\__,_(_)___| .__/| .__/
  __/ |                                    | |   | |
 |___/                                     |_|   |_|

...

*Usage*
  Enter an instruction and press enter (%C reset conversation, %Q quits).

*Examples*
  - Write an email to grandma thanking her for the cookies.
  - What are some historical attractions to visit around Massachusetts?
  - Compute the nth fibonacci number in javascript.
  - Write a standup comedy bit about WebGPU programming.

> What are some outdoorsy places to visit around Boston?

[ Reading prompt ] .....................


**Boston Harbor and Islands:**

* **Boston Harbor Islands National and State Park:** Explore pristine beaches, wildlife, and maritime history.
* **Charles River Esplanade:** Enjoy scenic views of the harbor and city skyline.
* **Boston Harbor Cruise Company:** Take a relaxing harbor cruise and admire the city from a different perspective.
* **Seaport Village:** Visit a charming waterfront area with shops, restaurants, and a seaport museum.

**Forest and Nature:**

* **Forest Park:** Hike through a scenic forest with diverse wildlife.
* **Quabbin Reservoir:** Enjoy boating, fishing, and hiking in a scenic setting.
* **Mount Forest:** Explore a mountain with breathtaking views of the city and surrounding landscape.

...
```

### 作为命令行工具使用

若要将 `gemma` 可执行文件作为命令行工具使用，可以创建一个完整指定参数的别名（alias）：

```sh
alias gemma2b="~/gemma.cpp/build/gemma -- --tokenizer ~/gemma.cpp/build/tokenizer.spm --weights ~/gemma.cpp/build/gemma2-2b-it-sfp.sbs --verbosity 0"
```

请将上述路径替换为您实际下载的模型和分词器路径。

以下是通过管道截断输入文件向 `gemma` 发起提示的示例（使用上述定义的 `gemma2b` 别名）：

```sh
cat configs.h | tail -n 35 | tr '\n' ' ' | xargs -0 echo "What does this C++ code do: " | gemma2b
```

> [!NOTE]
> gemma.cpp 的 CLI 命令行用法属于实验性功能，应注意上下文长度限制。

上述命令的输出效果如下：

```sh
[ Reading prompt ] [...]
This C++ code snippet defines a set of **constants** used in a large language model (LLM) implementation, likely related to the **attention mechanism**.

Let's break down the code:
[...]
```

### 在你的项目中以库的形式集成 gemma.cpp

在项目中集成 gemma.cpp 最简单的方法是通过 `FetchContent` 引入 gemma.cpp 及其依赖项。可以在 `CMakeLists.txt` 中添加以下内容：

```cmake
include(FetchContent)

FetchContent_Declare(sentencepiece GIT_REPOSITORY https://github.com/google/sentencepiece GIT_TAG 53de76561cfc149d3c01037f0595669ad32a5e7c)
FetchContent_MakeAvailable(sentencepiece)

FetchContent_Declare(gemma GIT_REPOSITORY https://github.com/google/gemma.cpp GIT_TAG origin/main)
FetchContent_MakeAvailable(gemma)

FetchContent_Declare(highway GIT_REPOSITORY https://github.com/google/highway.git GIT_TAG 97a5dd1af1a43a8b2ccbd31e556b4139bffbdafd)
FetchContent_MakeAvailable(highway)
```

注意：对于 gemma.cpp 的 `GIT_TAG`，如果希望锁定库版本，可以将 `origin/main` 替换为特定的提交哈希值。

在定义目标可执行文件后（将下方的 `[Executable Name]` 替换为您的可执行文件名称）：

```cmake
target_link_libraries([Executable Name] libgemma hwy hwy_contrib sentencepiece)
FetchContent_GetProperties(gemma)
FetchContent_GetProperties(sentencepiece)
target_include_directories([Executable Name] PRIVATE ${gemma_SOURCE_DIR})
target_include_directories([Executable Name] PRIVATE ${sentencepiece_SOURCE_DIR})
```

### 将 gemma.cpp 构建为独立库

gemma.cpp 也可以作为项目中的独立库依赖使用。可以通过修改 make 调用以构建 `libgemma` 目标来生成共享/静态库产物：

> [!NOTE]
> 如果您按照上一节中的 `FetchContent` 步骤在项目中集成 gemma.cpp，构建库将由 `cmake` 自动处理，可以跳过本节。

首先运行 `cmake`：

```sh
cmake -B build
```

然后运行 `make` 构建 `libgemma` 目标：

```sh
cd build
make -j [并行编译线程数] libgemma
```

构建成功后，`build/` 目录下将生成 `libgemma` 库文件。在 Unix 平台上，文件名为 `libgemma.a`。

## 使用 gemma.cpp 的独立社区项目

部分使用 gemma.cpp 的独立社区开源项目：

- [gemma-cpp-python - Python 绑定](https://github.com/namtranase/gemma-cpp-python)
- [lua-cgemma - Lua 绑定](https://github.com/ufownl/lua-cgemma)
- [Godot 引擎演示项目](https://github.com/Rliop913/Gemma-godot-demo-project)

如果您希望在此列出自己的项目，欢迎联系维护者或提交修改 `README.md` 的 PR。

## 致谢与联系方式

gemma.cpp 于 2023 年秋季由 [Austin Huang](mailto:austinvhuang@google.com) 和 [Jan Wassenberg](mailto:janwas@google.com) 发起，并在 Phil Culliton、Paul Chang 和 Dan Zheng 的贡献下于 2024 年 2 月正式发布。

Griffin 支持于 2024 年 4 月在 Andrey Mikhaylov、Eugene Kliuchnikov、Jan Wassenberg、Jyrki Alakuijala、Lode Vandevenne、Luca Versari、Martin Bruse、Phil Culliton、Sami Boukortt、Thomas Fischbacher 和 Zoltan Szabadka 的贡献下实现，并于 2025 年 9 月移除。

Gemma 2 支持于 2024 年 6/7 月在包括 Daniel Keysers 和 Phil Culliton 在内的多位成员帮助下实现。

PaliGemma 支持于 2024 年 9 月在 Daniel Keysers 的贡献下实现。

Gemma 3 支持于 2025 年 1 月至 3 月在 Daniel Keysers 和 Phil Culliton 的贡献下实现。

[Jan Wassenberg](mailto:janwas@google.com) 自初始发布以来持续贡献了大量改进，包括显著的效率提升。

[Phil Culliton](mailto:philculliton@google.com) 自初始发布以来主要负责模型发布、评估与验证、GTM 以及模型量化相关工作。

本项目不是 Google 官方正式支持的产品。

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月12日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*
