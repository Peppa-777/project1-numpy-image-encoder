# NumPy 手写 OpenCLIP ViT-B/32 图像编码器

课程 Project 1：**不依赖 PyTorch / OpenCLIP 包**，只用 NumPy 复现 OpenCLIP ViT-B-32 视觉塔的完整前向，
把 `[B,3,224,224]` 的预处理图像编码成 `[B,512]` 的单位长度图像特征（output token）。
参数由课程给出，本仓库只做前向计算，不涉及训练。

| 项目 | 结果 |
| --- | --- |
| 输入 / 输出 | `[B,3,224,224]` float32 → `[B,512]` float32，逐行 L2 归一化 |
| 内部 token 序列 | `[B,50,768]`（1 个 CLS + 49 个 patch） |
| 8 个校验阶段 | assets / preprocess / patches / tokens / block0 / last_tokens / output / batch **全部 PASS** |
| 最终精度 | output 最大绝对误差 **5.44e-07**，平均 8.31e-08（容差 atol=2e-05, rtol=2e-04） |
| 运行开销 | 纯 CPU，10 张图约 **7–11 s**；`check.py --stage all` 约 **10–16 s**（无 GPU，视机器而定） |

## 1. 目录结构

```text
.
├── image_encoder.py          # 核心实现：TODO-01 ~ TODO-06（本项目的唯一作业文件）
├── helpers.py                # 课程提供：LayerNorm、exact GELU、softmax 系数、单位化
├── preprocess_image.py       # 课程提供：短边 bicubic 缩放 + 中心裁剪 + CLIP 均值方差归一化
├── runtime.py                # 课程提供：按脚本所在目录加载权重与图片
├── run.py                    # 命令行入口：图片 → [B,512] → npz
├── check.py                  # 分阶段数值校验（对照独立 FP32 参考输出）
├── config.json               # 结构超参：224/32/768/12/3072/50/512，norm eps=1e-5
├── requirements.txt          # 依赖清单（numpy、Pillow）
├── assets/
│   ├── images/               # 10 张本地测试图 image_01.jpg ~ image_10.jpg
│   ├── images.json           # 每张图的来源 URL 与 SHA-256（可追溯）
│   └── manifest.json         # 参考输出文件指纹，供 check.py 校验
├── weights/
│   └── manifest.json         # 152 个视觉张量的名字 / 形状 / dtype / SHA-256
└── docs/
    └── verification-log.txt  # 本机实际运行 check.py --stage all 的完整输出
```

模块职责是单向的：`run.py` / `check.py` → `runtime.py`（加载）→ `preprocess_image.py`（预处理）→
`image_encoder.py`（模型）→ `helpers.py`（数值算子）。除 `image_encoder.py` 外均为课程提供且未修改。

## 2. 环境与依赖

- Python **3.9+**；无需 GPU、PyTorch、OpenCLIP 或联网。
- 依赖见 `requirements.txt`：`numpy>=1.24,<3`、`Pillow>=9.2,<13`。
- 本仓库验证环境：Windows 11 + Python 3.12 + NumPy 2.3.5 + Pillow 12.3.0，float32。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 3. 数据与权重（重要）

仓库**不包含**两个课程大文件，它们被 `.gitignore` 排除（`weights/image_encoder.npz` 约 326 MB，
超过 GitHub 单文件 100 MB 限制；课程也要求不要重复提交参数与参考数据）。运行前请从课程附件
`project1-学生版.zip` 的 `project1-numpy-student/` 目录复制到本仓库同名路径：

```text
project1-numpy-student/weights/image_encoder.npz  ->  weights/image_encoder.npz   (326 MB, 152 个张量)
project1-numpy-student/assets/reference_outputs.npz ->  assets/reference_outputs.npz (7.6 MB, 仅校验用)
```

- 权重完整性：`python check.py --stage assets` 会核对文件 SHA-256、张量名、形状、dtype 与逐张量哈希。
- 10 张测试图已随仓库提供（`assets/images/`），来源、原始 URL 与 SHA-256 记录在 `assets/images.json`，
  均为课程附件中的公开新闻图片，仅用于复现数值。
- `assets/reference_outputs.npz` 只被 `check.py` 读取，**绝不会**作为编码器输入；实现中没有任何缓存或查表。

## 4. 运行与复现

```bash
# 1) 资源校验（152 个张量 + 10 组图片/参考输出）
python check.py --stage assets

# 2) 按实现顺序逐段定位第一个不一致的阶段
python check.py --stage patches
python check.py --stage tokens
python check.py --stage block0
python check.py --stage last_tokens
python check.py --stage output
python check.py --stage batch

# 3) 一次性跑完全部阶段
python check.py --stage all

# 4) 生成输出 token
python run.py --output output_tokens.npz
python run.py --images path/to/a.jpg path/to/b.png --output my_tokens.npz
```

`run.py` 保存的 `ids` 与输入图片顺序一一对应，输出形状 `[B,512]`、dtype float32、逐行单位长度。

## 5. 实现说明

全部改动集中在 `image_encoder.py`，六个 TODO 与关键 shape 变化如下：

| TODO | 位置 | shape 变化 | 做法 |
| --- | --- | --- | --- |
| 01 | `ImageEncoder.patchify` | `[B,3,224,224]` → `[B,49,3072]` | `reshape(B,3,7,32,7,32)` 拆出网格轴，`transpose(0,2,4,1,3,5)` 变成 (行,列,C,ph,pw)，再 flatten；patch 从左到右、从上到下，patch 内按 C,H,W |
| 02 | `MultiHeadAttention.split_heads` | `[B,N,768]` → `[B,12,N,64]` | `reshape(B,N,12,64)` 后 `transpose(0,2,1,3)`，把 head 轴提到 token 轴之前 |
| 03a | `MultiHeadAttention.__call__` | q,k `[B,12,N,64]` → scores `[B,12,N,N]` | `q @ swapaxes(k,-1,-2) / sqrt(64)`；k 是 4D，只交换最后两轴，batch 与 head 分组不变（不能用 2D 的 `k.T`） |
| 03b | 同上 | 系数 `[B,12,N,N]`、v `[B,12,N,64]` → merged `[B,N,768]` | `context = A @ v`，再 `transpose(0,2,1,3).reshape` 把 12 个头拼回特征轴 |
| 03c | 同上 | `[B,N,768]` → `[B,N,768]` | 权重按 `[out,in]` 存放，故 `merged @ out_weight.T + out_bias` |
| 04a/04b | `TransformerBlock.__call__` | 全程 `[B,N,768]`，隐层 `[B,N,3072]` | 残差 1：`residual = x + attn(norm1(x))`；MLP：`GELU(u @ W1.T + b1) @ W2.T + b2`；残差 2：`residual + mlp_update`。视觉塔没有 LayerScale |
| 05a/05b | `ImageEncoder.embed` | `[B,49,3072]` → `[B,49,768]` → `[B,50,768]` | patch 投影无 bias（`patches @ patch_weight.T`）；沿 token 轴 `concatenate([cls, patches])`，再加 `[50,768]` 位置表（广播到 B） |
| 06 | `ImageEncoder.__call__` | `[B,50,768]` → `[B,768]` → `[B,512]` | 取 `tokens[:,0]`（CLS）→ `post_norm` → `@ visual.proj`（已是 `[in,out]`，**不转置**）→ `unit_vector` |

几个容易写错的点：

1. **权重方向**：注意力/MLP 线性层存的是 `[out,in]`，必须右乘 `weight.T`；只有 `visual.proj` 是 `[in,out]`。
2. **k 的转置**：写 `k.T` 会把 batch 和 head 轴一起翻掉，必须用 `swapaxes(-1,-2)` 或 `transpose(0,1,3,2)`。
3. **token 顺序**：CLS 必须在 patch 行之前（第 0 行），`check.py` 的 tokens 阶段会立刻抓到这个错误。
4. **残差**：两个残差都加在 `[B,N,768]` 上，MLP 的 3072 维只存在于中间隐层。
5. **dtype**：全流程 float32，最后一步除单位长度时用 `max(length,1e-12)` 防止除零（`helpers.unit_vector` 已提供）。

## 6. 数值校验结果

容差取自 `check.py`：preprocess `atol=2e-6`，output/batch `atol=2e-5`，其余 `atol=2e-4`，统一 `rtol=2e-4`。
下表为本机实际输出（完整记录见 [docs/verification-log.txt](docs/verification-log.txt)）：

| 阶段 | 形状 | max_abs | mean_abs | 结论 |
| --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0.000e+00 | 0.000e+00 | PASS |
| patches | (2,49,3072) | 0.000e+00 | 0.000e+00 | PASS |
| tokens | (2,50,768) | 4.005e-05 | 1.289e-06 | PASS |
| block0 | (2,50,768) | 7.063e-06 | 3.250e-07 | PASS |
| last_tokens | (2,50,768) | 4.768e-05 | 2.696e-06 | PASS |
| output | (10,512) | 5.439e-07 | 8.313e-08 | PASS |
| batch | (3,512) | 4.619e-07 | 7.632e-08 | PASS |

`batch` 阶段用 `[1,0,1]` 的乱序 + 重复图片索引，确认实现不依赖 batch 顺序、也没有把结果写死；
`output` / `batch` 还额外断言每行 `sum(x²)=1`。误差全部低于容差 2~3 个数量级，说明差异只来自
float32 累加顺序（NumPy 与参考的 Torch FP32 内核不同），不存在公式或 shape 错误。

## 7. 已知限制

- 只实现了前向推理；没有反向传播、没有训练。
- GELU 使用课程提供的 `math.erf` Python 循环实现，逐元素较慢；换成向量化 `erf` 会改变 float32 舍入路径，
  需要重新对照容差（本项目保持与课程 helper 完全一致）。
- 单进程 CPU 实现，显存/内存开销随 batch 线性增长（10 张图约 100 MB 量级）。
- 仅支持这个固定 checkpoint（`config.json` 中的 224/32/768/12/50/512），换模型需改结构参数。

## 8. 参考与致谢

- 代码骨架、`helpers.py`、`preprocess_image.py`、权重与参考输出由课程提供。
- 模型结构对应 OpenCLIP **ViT-B-32**（patch 32、宽度 768、12 层、12 头、输出 512）。
- 测试图片来源见 `assets/images.json`（原始文章与图片 URL、SHA-256 均已记录）。
