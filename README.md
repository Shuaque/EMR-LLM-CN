# EMR-LLM-CN
An audio-visual LLM



### Environment Setup
```bash

conda create -n emr python=3.9 -y
conda activate emr
git clone https://github.com/Shuaque/EMR-LLM-CN.git
cd EMR-LLM-CN


# PyTorch and related packages
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.23.5 scipy opencv-python
pip install editdistance python_speech_features einops soundfile sentencepiece tqdm tensorboard unidecode librosa
pip install omegaconf==2.0.6 hydra-core==1.0.7 (If your pip version > 24.1, please run "python3 -m pip install --upgrade pip==24.0")
pip install transformers==4.47.1 peft==0.14.0 bitsandbytes==0.45.0
cd fairseq
pip install --editable ./

```