from speechbrain.inference.interfaces import Pretrained
import torch

def classify_accent(audio_path):
    out_prob, score, index, label = classifier.classify_file(audio_path)
    return label[0], round(float(score[0]), 3)
