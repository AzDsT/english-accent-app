import os
import soundfile as sf
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class
os.environ["SPEECHBRAIN_CACHE_STRATEGY"] = "copy"
# Load the model from Hugging Face with the custom interface
# Register the custom class
classifier = foreign_class(
    source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
    pymodule_file="custom_interface.py",
    classname="CustomEncoderWav2vec2Classifier"
)

def classify_accent(audio_path):
    out_prob, score, index, label = classifier.classify_file(audio_path)
    return label[0], round(float(score[0]), 3)