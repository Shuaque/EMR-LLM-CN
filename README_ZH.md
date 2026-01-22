# EMR-LLM-CN

EMR-LLM-CN 是一个基于 **Qwen2.5-3B-Instruct** 的大语言模型（LLM），能够处理**多模态对话输入**，并将其总结为**结构化、基于标签的电子病历（EMR）**，同时支持**视听语音识别**和**电子病历标签生成**。

代码受版权保护，使用时请**注明来源**，**禁止任何侵权行为**。

![Overview](data/examples/overview-fig.png)

## 环境配置

```bash

conda create -n emr python=3.9 -y
conda activate emr
git clone https://github.com/Shuaque/EMR-LLM-CN.git
cd EMR-LLM-CN

# PyTorch and related packages
pip install -U "pip<24.1" "setuptools<72" #(If your pip version > 24.1, please run this)
pip install "PyYAML>=5.1" "omegaconf==2.0.6" "hydra-core==1.0.7"


cd fairseq
pip install --editable ./

pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5 scipy opencv-python
pip install editdistance python_speech_features einops soundfile sentencepiece tqdm tensorboard unidecode librosa
pip install omegaconf==2.0.6 hydra-core==1.0.7 #(If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install transformers==4.47.1 peft==0.14.0
pip install bitsandbytes --prefer-binary
pip install tensorboardX av matplotlib scikit-image

```

* 在 `../../EMR-LLM-CN/data/preparation/detectors/retinaface` 目录下安装 `retinaface` 检测器，你也可以在 `../detectors` 中放置其他检测器：

```bash
cd ../data/preparation/detectors/retinaface/
# Install [ibug.face_detection](https://github.com/hhj1897/face_detection)


git clone https://github.com/hhj1897/face_detection.git

# if its exisct in dir, than:
cd face_detection
pip install -e .
cd ..
```

* 建议：由于 `ibug/face_detection/retina_face/weights/Resnet50_Final.pth` 经常下载出错，推荐手动下载并放置到指定目录。

```bash
Install [*`ibug.face_alignment`*](https://github.com/hhj1897/face_alignment)

git clone https://github.com/hhj1897/face_alignment.git

# if its exisct in dir, than:
cd face_alignment
pip install -e .
cd ..
```

* 从以下链接下载 `Reference mean face`，并同样放置到 `../detectors` 目录中：[Link](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/20words_mean_face.npy)

## 下载

1. 下载下方由 AI 生成的多模态（视听）EMR 数据集，并放置到 `../../EMR-LLM-CN/data` 中
2. 下载 `AVSR` 与 `EMR` 任务对应的模型 checkpoint，并放置到 `../../EMR-LLM-CN/pretrianed` 中
3. 下载 `avhubert` checkpoint，并放置到 `../../EMR-LLM-CN/pretrained/Avhubert/base_vox_iter5.pt`
4. 从 HuggingFace 或 ModelScope 平台下载 `whisper`，并放置到 `../../EMR-LLM-CN/pretrained/Whisper/whisper-large`
5. 从 HuggingFace 或 ModelScope 平台下载 `Qwen2.5-3B-Instruct`，并放置到 `../../EMR-LLM-CN/pretrained/LLM/Qwen2.5-3B-Instruct`

| File Name               | Source URL                                                                                                                                                                      | File Size |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| CMDD-MIE-EMR-AV         | [GoogleDrive](https://drive.google.com/drive/folders/1XjJ0T5kQ-ntyWc_2M1EHZJxpba1c9FId?usp=sharing) or [BaiduDrive](https://pan.baidu.com/s/1FI7JZw8aSFEoO13X405SBg)(key: cize) | 18GB      |
| emr_checkpoint_best.pt  | [GoogleDrive]() or [BaiduDrive]()(key: )                                                                                                                                        | 8.3GB     |
| avsr_checkpoint_best.pt | [GoogleDrive]() or [BaiduDrive]()(key: )                                                                                                                                        | 14.3GB    |

由于超参数优化，部分结果略优于论文中的结果。AV-ASR 相关代码与 checkpoint 可在发布版本中获取。

## 推理

运行脚本 `../../EMR-LLM-CN/inference.py` 以进行**单样本评估**。脚本内置了评估示例，支持在 **纯文本**、**音视频** 和 **仅音频** 等多种模态的 EMR 数据上进行评估。多模态数据示例可在 `../../EMR-LLM-CN/data/examples` 中找到，其中提供了可用于评估的音频或视频样本。

```bash
export ROOT=../../EMR-LLM-CN
export TOKENIZERS_PARALLELISM=false
export SRC_PTH="$ROOT/src"
export PYTHONPATH=$ROOT:$ROOT/fairseq:$SRC_PTH

CHECKPOINT=/workspace/shuaque/Classification_Semantic_att_LLM/exp/202512/run/28190745_A800_Optimized_Exp1_3_B_3_loss/checkpoints/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0 python $ROOT/inference.py \
    --common-user-dir /workspace/shuaque/EMR-LLM-CN/src \
    --checkpoint-path $CHECKPOINT \
    --ontology-path /workspace/shuaque/EMR-LLM-CN/data/ontology.json \
    --device cuda:0
```

示例结果可在 `../../EMR-LLM-CN/inference.log` 中查看。

```bash
| [Pipeline] Processing Dialogue Inputs...
...
```

## EMR 数据的批量评估

请确保 `test.json` 保存在 `../data` 中，并且 `$CHECKPOINT` 已下载并放置在 `../../EMR-LLM-CN/pretrained` 目录下。`--ratios` 参数可在 `[0.1–1.0]` 范围内设置，用于减少候选标签数量，从而缩短评估时长。评估结果将保存在 `../../EMR-LLM-CN/results/` 中。评估脚本位于 `../../EMR-LLM-CN/scritps/eval_emr.sh`，示例结果可在 `../../EMR-LLM-CN/results/emr_eval_sweep.log` 中查看。

```bash
export ROOT=../../EMR-LLM-CN
export TOKENIZERS_PARALLELISM=false
export SRC_PTH="$ROOT/src"
export PYTHONPATH=$ROOT:$ROOT/fairseq:$SRC_PTH

CHECKPOINT=../EMR-LLM-CN/pretrained/emr_checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0 python3 $SRC_PTH/eval.py \
    --common-user-dir $SRC_PTH \
    --checkpoint-path $CHECKPOINT \
    --split test \
    --device cuda:0 \
    --output-dir $ROOT \
    --ratios 1.0
```

## 视听数据的批量评估

请将音视频路径文件 `test.tsv` 及对应的文本标签文件 `test.ltr` 存放在 `$DATA` 中。音视频路径文件应包含 `id, role, video_path, audio_path, video_frames, audio_frames`，标签文件应包含 `id, role, text`。或者修改数据集类 `../../EMR-LLM-CN/scr_avsr/dataset.py`。更多细节请参考下方 **Train for Audio-Visual Speech Recognotion** 部分。示例结果可在 `../../EMR-LLM-CN/results/avsr_eval_log.txt` 中查看。

```bash
#! /bin/bash
...
```

## EMR 生成任务训练

请确保 **纯文本 EMR 数据集** 保存在 `../data/test.json | train.json | dev.json` 中，**训练脚本** 位于 `../../EMR-LLM-CN/scirpts/train_emr.sh`，可在 `../../EMR-LLM-CN/src/conf/train.yaml` 中修改参数配置，**训练日志** 将保存在 `../../EMR-LLM-CN/exp` 中。

```bash
# #!/usr/bin/env bash
...
```

## 视听语音识别训练

请确保 **视听数据集** 保存在 `../data/test.tsv | train.tsv | valid.tsv` 中，每条记录格式为 `id, role, video_path, audio_path, video_frames, audio_frames`，对应的 **文本标签** 保存在 `../data/test.ltr | train.ltr | valid.ltr` 中，格式为 `id, role, text`，例如：

```bash
# train.tsv
...
```

**训练脚本** 位于 `../../EMR-LLM-CN/scirpts/train_avsr.sh`，参数配置可在 `../../EMR-LLM-CN/src_avsr/conf/train_avsr.yaml` 中修改，**训练日志** 将保存在 `../../EMR-LLM-CN/exp` 中。

```bash
#!/usr/bin/env bash
...
```
