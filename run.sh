#!/bin/bash

# # speech seperator
# conda activate py310  # 如果是 conda: conda activate env1
# python /home/xinying/Speaker2/seperate_speech/run.py  # 将 project A 的输出保存到文件

# conda deactivate

# speaker recognition
source ~/.bashrc
conda activate py36
tempfile=$(mktemp) # Which speaker
python /home/xinying/Speaker2/SpeakerRecognition_tutorial-master/featureExtractor.py 
python /home/xinying/Speaker2/SpeakerRecognition_tutorial-master/verification.py >"$tempfile"
conda deactivate

# STT
tempfile2=$(mktemp) # text without punctuation
conda activate py310
python /home/xinying/Speaker2/AIpendant/stt.py  > "$tempfile2"
conda deactivate

# Punctuation
conda activate puncrestore
python /home/xinying/Speaker2/AIpendant/punctuation.py "$tempfile" "$tempfile2"
conda deactivate

# Fuzzy Search
conda activate QW
python /home/xinying/Speaker2/Qwen-AIpendant/timeinfo.py


