# from funasr import AutoModel

# model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
#                   vad_model="fsmn-vad", vad_model_revision="v2.0.4",
#                   punc_model="ct-punc-c", punc_model_revision="v2.0.4",
#                   # spk_model="cam++", spk_model_revision="v2.0.2",
#                   )
# res0 = model.generate(input="/home/xinying/Speaker2/AIpendant/SPEAKER_00.wav", batch_size_s=300)
# print(res0)

from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad",  punc_model="ct-punc", 
                  # spk_model="cam++", 
                  )
res = model.generate(input=f"/home/xinying/Speaker2/AIpendant/SPEAKER_00.wav", 
                     batch_size_s=300, 
                     hotword='魔搭')
print(res[0]['text']) #with timestamp