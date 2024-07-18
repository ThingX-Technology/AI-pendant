#py310
# b.py
import sys
import wenet


model = wenet.load_model("chinese")
result = model.transcribe(
    '/home/xinying/Speaker2/Qwen-AIpendant/SPEAKER_00.wav'
    )
print(result['text'])





