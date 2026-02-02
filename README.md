# 3D-SCoT: Teaching 3D-LLMs to Think Spatially with Million-scale CoT Annotations

This is the official PyTorch implementation of SCoT. 

## Abstract

<p align="justify">
Recent advances in 3D Large Language Models (3D-LLMs) show strong potential in understanding and interacting with 3D environments, yet their training data typically lack explicit reasoning processes, limiting complex spatial reasoning and task planning. To address this, we annotate SCoT, a million-scale Chainof-Thought dataset spanning three levels: a) Spatial Perception (what is there), recognizing object properties, relations, and scene attributes; b) Spatial Analysis (what does it mean), inferring rationality, functionalities, and physical implications; c) Spatial Planning (what should I do), integrating perception and reasoning for actionable strategies. Unlike prior datasets supervising only answers, SCoT annotates intermediate reasoning grounded in scene cues, specifically for analysis and planning tasks. Results show that CoT supervision greatly benefits complex analysis and planning but induces hallucinations and accuracy drops in simple perception. These findings highlight both the necessity and the nuanced challenges of scene-grounded reasoning for advancing 3D intelligence.
</p>

 ![overview](Fig/TeaserFig.png)

## 💾 SCoT Dataset

### Text Annotations

We have released SCoT dataset (including Spatial Perception, Spatial Analysis, and Spatial Planning Data) in [Google Drive](https://drive.google.com/drive/folders/1Nc1KnpOZYh6WWq57Uusx2Zcwp_YYJxuN?usp=drive_link) and [Hugging Face](https://huggingface.co/datasets/Orange-zZZ/SCoT/tree/main).

### Source Data and Preprocessed Features

We have released all source data and preprocessed features in [Google Drive](https://drive.google.com/drive/folders/1_4L8uDdSG4ykFwcZjgEbo-uvCprJ5hWP?usp=drive_link).

If you intend to train and test on your own 3D dataset, or if you want to explore generating the features yourself, it is recommended to refer to the way similar to [Chat Scene](https://github.com/ZzZZCHS/Chat-Scene/tree/dev/preprocess).

## 💻 Requirements
The code has been tested on:
- Ubuntu 20.04
- CUDA 12.2
- Python 3.10
- Pytorch 2.2.1
- NVIDIA A100 GPU (40G).

## 🔧 Installation
  
- Create and activate the conda environment
  ```
  conda create -n SCoT python=3.10
  conda activate SCoT
  ```

- Install the necessary packages
  ```
  conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
  pip install -r requirements.txt
  ```

## 🚅 Train

In the first stage, SCoT-Reasoner is trained to establish a basic understanding for 3D scenes. You can train the SCoT-Reasoner using [Vicuna-7B v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5) as backbone, note that you should change the `llama_model_path` in [run.sh](./scripts/run.sh) to the path of `Vicuna-7B v1.5`.

- Please download the spatial perception data and modify `scripts/config_stage_1.py`. We have organized it into a trainable format in `./SCoT_Dataset/annotations`.
- Please modify `run.sh` with the following configuration. Then, run the code: `bash scripts/run.sh`.
 ```python
    # run.sh (stage 1)
    train_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref#nr3d_caption#obj_align"
    val_tag="scanrefer#scan2cap#scanqa#sqa3d#multi3dref"
    evaluate=False

    python tasks/train.py \
       "$(dirname $0)/${config}config_stage_1.py" \
       ...
 ```

In the second stage, SCoT-Reasoner is fine-tuned to generate reasoning chains. Please change `pretrained_path` in [run.sh](./scripts/run.sh) to the path of checkpoint from the first stage, or you can use [checkpoint](https://huggingface.co/datasets/ZzZZCHS/Chat-Scene/tree/main/pretrained) from Chat Scene as pre-trained model.

 - Please download the spatial analysis and planning data and modify `scripts/config_stage_2.py`. We have organized it into a trainable format in `./SCoT_Dataset/SCoT_Training_Stage_2`.
 - Please modify `run.sh` with the following configuration. Then, run the code: `bash scripts/run.sh`.
 ```python
    # run.sh (stage 2)
    train_tag="scanrefer#obj_align#scan2cap#sqa3d"
    val_tag="scanrefer"
    evaluate=False

    python tasks/train.py \
       "$(dirname $0)/${config}config_stage_2.py" \
       ...
 ```

## ✏️ Evaluation

You can evaluate the model performances on SCoT dataset. Please change the `pretrained_path` in [run.sh](./scripts/run.sh) to the path of `checkpoints` and eval with the code `bash scripts/run.sh`:

```python
    # run.sh (Test)
    val_tag="scanrefer#sqa3d"
    evaluate=True
    pretrained_path="/Path_to_Pretrained_Model.pth"
 ```

We have provided the pretrained checkpoint of SCoT-Reasoner in [Google Drive](https://drive.google.com/drive/folders/1dT8Z2hATGWDzTkksbVYgGkVG-HU7fpd-?usp=drive_link) and [Hugging Face](https://huggingface.co/Orange-zZZ/SCoT-Reasoner-PTH/tree/main).

- Text-based Metrics
 ```python
    python utils/Eval_SCoT.py
 ```
- LLM-based Assessments
 ```python
    python utils/Eval_SCoT_LLM_Score.py
 ```

## 😊 Acknowledgement

Thanks to these extremely wonderful open-source projects:

3D Dataset: [ScanNet](https://github.com/ScanNet/ScanNet), [ARKitScenes](https://github.com/apple/ARKitScenes).

3D-Language Dataset: [ScanRefer](https://github.com/daveredrum/ScanRefer), [Scan2Cap](https://github.com/daveredrum/Scan2Cap), [Sqa3D](https://github.com/SilongYong/SQA3D), [MSR3D](https://github.com/MSR3D/MSR3D).

Representations: [Uni3D](https://github.com/baaivision/Uni3D), [DINO v2](https://github.com/facebookresearch/dinov2).

3D-LLMs: [3D LLM](https://github.com/UMass-Embodied-AGI/3D-LLM), [Video-3D-LLM](https://github.com/LaVi-Lab/Video-3D-LLM), [Chat Scene](https://github.com/ZzZZCHS/Chat-Scene).

## Contact us
If you find this repo helpful, please give us a star. For any questions, please contact us via lijp57@whu.edu.cn.

