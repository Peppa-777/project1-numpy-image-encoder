# 用 NumPy 写一个 CLIP 图像编码器（Project 1）

这是课程 Project 1 的作业。要求是不用 PyTorch，只用 NumPy 把 OpenCLIP 的 ViT-B/32 图像编码器的前向计算写一遍：
输入一张预处理好的图片，输出一个 512 维的特征向量。

## 整个流程一句话

224×224 的图 → 切成 7×7 共 49 个小块 → 加一个"总代表"（CLS）→ 一共 50 个 token →
过 12 层 Transformer → 只取"总代表" → 压成 512 维 → 长度归一化成 1。

## 我改了什么

课程给了完整的代码，只在 `image_encoder.py` 里留了 6 个 TODO。我补的就是这 6 处：

1. **patchify**：把图切成小块。224 正好是 7 个 32，所以切成 7×7，每块 32×32，摊平是 3×32×32 = 3072 维。
2. **split_heads**：把 768 维平均分给 12 个头，每个头 64 维，让每个头单独算注意力。
3. **注意力打分**：每个 token 用自己的 q 去乘别人的 k，得到"关注程度"；除以 √64 防止数值太大；
   softmax 变成权重后，对 v 加权求和。
4. **TransformerBlock**：先算注意力再加回原来的值（残差），然后过 MLP（768 → 3072 → 768）再加回原值。
5. **embed**：小块投影成 768 维，把 CLS 拼到最前面，再加上位置编码。
6. **最后的输出**：只取第 0 个 token（CLS），过 LayerNorm，乘投影矩阵变成 512 维，最后归一化。

## 容易搞错的两个地方

- 权重方向：大部分线性层的权重存的是 `[输出, 输入]`，要乘 `W.T`；只有最后的投影矩阵是 `[输入, 输出]`，直接乘。
- CLS 必须放在最前面（第 0 行），因为最后取的是第 0 个 token，放后面就取错了。

## 仓库里有什么

```text
image_encoder.py           我补的 6 个 TODO
helpers.py                 课程给的：LayerNorm、GELU、softmax、归一化
preprocess_image.py        课程给的：缩放、裁剪、归一化
run.py                     课程给的：跑 10 张测试图，输出 npz
check.py                   课程给的：分 8 个阶段和参考结果对比
config.json                课程给的：模型参数
assets/images/             10 张测试图
docs/verification-log.txt  我跑 check.py 的完整输出
```

## 怎么运行

环境：Python 3.10 以上，只需要 `numpy` 和 `Pillow`，不需要 GPU 和 PyTorch。

```bash
pip install -r requirements.txt
python check.py --stage assets     # 先检查权重文件在不在
python check.py --stage all        # 跑完 8 个阶段
python run.py --output output_tokens.npz
```

## 两个大文件怎么来

`weights/image_encoder.npz`（326 MB）和 `assets/reference_outputs.npz`（7.6 MB）太大，没有放进仓库。
从课程附件 `project1-学生版.zip` 解压出来的 `project1-numpy-student/` 目录里，把这两个文件复制到仓库的同名位置：

```text
project1-numpy-student/weights/image_encoder.npz    ->  weights/image_encoder.npz
project1-numpy-student/assets/reference_outputs.npz ->  assets/reference_outputs.npz
```

## 结果

`check.py` 的 8 个阶段全部 PASS。最终输出和参考结果相比，最大误差是 5.44e-07，容差是 2e-05。
10 张图在 CPU 上跑完大约 7 到 11 秒。

## 说明

- 只做了前向计算，没有训练，也没有反向传播。
- GELU 用的是课程 `helpers.py` 里的 Python 循环版本，比较慢，但没有改，因为改了舍入会变。
- 只支持 `config.json` 里这一套参数（224、patch 32、768 维、12 层）。
