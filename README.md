# 用 NumPy 手写 OpenCLIP ViT-B/32 图像编码器

课程 Project 1 的作业仓库。任务是不借助 PyTorch / OpenCLIP，只用 NumPy 实现 ViT-B/32 视觉塔的前向计算：
输入预处理好的图片 `[B,3,224,224]`，输出每张图的 512 维特征 `[B,512]`（长度归一化为 1）。
参数是课程给的，这里只做前向，不训练。

| 项目 | 结果 |
| --- | --- |
| 输入 → 输出 | `[B,3,224,224]` → `[B,512]`，float32，每行 L2 长度为 1 |
| 中间 token | `[B,50,768]`，第 0 行是 CLS，后面 49 行是 patch |
| 8 个校验阶段 | assets、preprocess、patches、tokens、block0、last_tokens、output、batch 全部 PASS |
| 最终误差 | output 最大绝对误差 5.44e-07（容差 2e-05） |
| 运行时间 | 纯 CPU，10 张图约 7–11 秒，`check.py --stage all` 约 10–16 秒 |

## 我做了什么

- 补完 `image_encoder.py` 里的 6 个 TODO（TODO-01 ~ TODO-06），一共改了 +35 / −21 行，
  涉及 patchify、多头注意力、TransformerBlock、patch 投影与 CLS 拼接、最后的 CLS 池化和投影。
- `helpers.py`、`preprocess_image.py`、`runtime.py`、`run.py`、`check.py`、`config.json`、
  `assets/` 和 `weights/manifest.json` 都是课程提供的框架、数据和配置，我没有改动。
- 自己跑完了 `check.py` 的 8 个阶段，全部通过，原始输出保存在 `docs/verification-log.txt`。

## 目录结构

```text
.
├── image_encoder.py          # 我写的部分：6 个 TODO
├── helpers.py                # 课程提供：LayerNorm、GELU、softmax、单位化
├── preprocess_image.py       # 课程提供：缩放、裁剪、归一化
├── runtime.py                # 课程提供：读取权重和图片
├── run.py                    # 课程提供：跑图片、保存 npz
├── check.py                  # 课程提供：分阶段数值校验
├── config.json               # 课程提供：模型超参数
├── requirements.txt          # 依赖：numpy、Pillow
├── assets/
│   ├── images/               # 10 张测试图
│   ├── images.json           # 图片的来源链接和 SHA-256
│   └── manifest.json         # 参考输出文件的指纹
├── weights/manifest.json     # 权重文件的校验清单
└── docs/verification-log.txt # 我本机跑 check.py 的输出
```

## 环境

- Python 3.10 及以上（我只在 3.12 上验证过），不需要 GPU，也不需要装 PyTorch 和 OpenCLIP。
- 依赖只有两个：`numpy` 和 `Pillow`，版本见 `requirements.txt`。
- 我本机是 Windows 11 + Python 3.12 + NumPy 2.3.5 + Pillow 12.3.0。

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS / Linux 用 source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 数据和权重怎么来

仓库里没有这两个大文件：`image_encoder.npz` 有 326 MB，超过 GitHub 的单文件上限；`reference_outputs.npz` 只是校验用的参考数据，课程也说了不用重复提交。
运行之前从课程附件 `project1-学生版.zip` 里解压出来的 `project1-numpy-student/` 目录，把这两个文件复制到本仓库的同名位置：

```text
project1-numpy-student/weights/image_encoder.npz   ->  weights/image_encoder.npz
project1-numpy-student/assets/reference_outputs.npz ->  assets/reference_outputs.npz
```

复制完可以先用 `python check.py --stage assets` 检查文件是否完整（会核对 SHA-256、张量名、形状和 dtype）。
`reference_outputs.npz` 只在 `check.py` 里用来对比，不会作为编码器的输入，代码里也没有任何查表。

## 运行

```bash
python check.py --stage assets        # 先检查权重和参考文件
python check.py --stage all           # 跑完 8 个阶段
python run.py --output output_tokens.npz
```

`check.py` 也可以单独跑某一阶段，比如 `--stage patches`、`--stage output`，哪个阶段先报错就说明问题在哪一段。

## 实现思路

六个 TODO 和它们的形状变化：

| TODO | 位置 | 形状变化 | 做法 |
| --- | --- | --- | --- |
| 01 | `patchify` | `[B,3,224,224]` → `[B,49,3072]` | `reshape(B,3,7,32,7,32)` 拆成 7×7 个小块，`transpose` 把块的行列提到前面，再 flatten |
| 02 | `split_heads` | `[B,N,768]` → `[B,12,N,64]` | `reshape(B,N,12,64)` 再 `transpose(0,2,1,3)`，把 12 个头分出来 |
| 03 | `MultiHeadAttention` | `[B,12,N,64]` → `[B,12,N,N]` → `[B,N,768]` | 打分用 `q @ kᵀ / √64`，加权求和得到 `context`，再把 12 个头拼回去，最后过一次输出投影 |
| 04 | `TransformerBlock` | 一直是 `[B,N,768]` | 先 `x + attention(norm1(x))`，再 `residual + MLP(...)`，两个残差 |
| 05 | `embed` | `[B,49,3072]` → `[B,50,768]` | patch 投影没有 bias，投影完把 CLS 拼在第 0 行，再加位置编码 |
| 06 | `__call__` | `[B,50,768]` → `[B,512]` | 取第 0 行 CLS，过 `post_norm`，再乘 `visual.proj`，最后单位化 |

写的时候几个容易出错的地方：

1. 线性层的权重是 `[out,in]` 存的，要右乘 `weight.T`；但 `visual.proj` 是 `[in,out]`，不用转置。
2. 注意力的 `k` 是 4 维的，直接 `k.T` 会把 batch 和 head 轴也翻过去，要用 `swapaxes(-1,-2)`。
3. CLS 必须在 patch 前面，也就是第 0 行；顺序反了 `tokens` 阶段会立刻报错。
4. 两个残差都加在 768 维上，3072 只是 MLP 中间的隐层。
5. 全程用 float32，最后单位化时长度可能为 0，`helpers.unit_vector` 里已经用 `max(length, 1e-12)` 处理了。

## 校验结果

下面是我本机跑 `python check.py --stage all` 的实际结果（完整记录见 [docs/verification-log.txt](docs/verification-log.txt)）：

| 阶段 | 形状 | 最大误差 | 平均误差 | 结果 |
| --- | --- | --- | --- | --- |
| preprocess | (10,3,224,224) | 0.000e+00 | 0.000e+00 | PASS |
| patches | (2,49,3072) | 0.000e+00 | 0.000e+00 | PASS |
| tokens | (2,50,768) | 4.005e-05 | 1.289e-06 | PASS |
| block0 | (2,50,768) | 7.063e-06 | 3.250e-07 | PASS |
| last_tokens | (2,50,768) | 4.768e-05 | 2.696e-06 | PASS |
| output | (10,512) | 5.439e-07 | 8.313e-08 | PASS |
| batch | (3,512) | 4.619e-07 | 7.632e-08 | PASS |

`batch` 阶段故意把图片顺序打乱成 `[1,0,1]`，还重复了一张，用来确认结果和 batch 顺序无关，也不是写死的。
误差比容差小两三个数量级，我理解是 float32 累加顺序不同造成的（NumPy 和参考实现的底层并不完全一样），
公式和形状应该没有问题，否则误差会大得多。

## 没做的地方

- 只做了前向，没有反向传播，也没有训练。
- GELU 用的是课程 `helpers.py` 里的 Python 循环，速度比较慢；换成向量化写法会改变舍入方式，所以没动。
- 只支持 `config.json` 里这一套参数（224/32/768/12/50/512），换模型要改结构。

## 参考

- 代码框架、权重和参考输出由课程提供。
- 模型结构对应 OpenCLIP 的 ViT-B-32。
- 测试图的来源链接记录在 `assets/images.json`。
