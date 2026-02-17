# 🧠 Multimodal Facial and Vocal Emotion Recognition  
### Transformer-Based Audio–Visual Emotion Fusion System  

## 📌 Overview  

This project presents a multimodal deep learning system for emotion recognition using both facial expressions and speech signals. The system integrates Vision Transformers (ViT) for facial feature extraction, Wav2Vec2 for speech representation learning, and a cross-modal Transformer for audio–visual fusion.

Unlike traditional unimodal approaches, this framework jointly models visual and acoustic cues to improve robustness in real-world conditions such as poor lighting, facial occlusion, and background noise. The implementation supports real emotion datasets including MELD, IEMOCAP, and CMU-MOSEI.

This project is developed as part of an academic research review and experimental framework for multimodal affective computing.

---

## 🚀 Key Features  

- Multimodal emotion recognition (face + voice)  
- Vision Transformer (ViT) for facial embeddings  
- Wav2Vec2 for speech emotion embeddings  
- Cross-modal Transformer fusion  
- Real dataset integration (MELD / IEMOCAP / CMU-MOSEI)  
- Real-time prediction from image + audio files  
- Training and evaluation pipeline  
- Edge-aware architecture support  
- Academic review + experimental framework  

---

## 📊 Emotion Classes  

0 – Neutral
1 – Happy
2 – Sad
3 – Angry
4 – Fear
5 – Surprise
6 – Disgust


---

## 🏗 System Architecture  

Image Frame → ViT → Visual Embeddings ┐
→ Cross-Modal Transformer → Emotion Classifier
Audio Signal → Wav2Vec2 → Audio Embeddings ┘


---

## 📁 Project Structure  

.
├── transformer_fusion_emotion.py
├── datasets/
│ └── MELD / IEMOCAP / CMU-MOSEI
├── models/
├── checkpoints/
├── README.md
└── requirements.txt


---

## ⚙ Installation  

### Create virtual environment  

```bash
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
Install dependencies
pip install torch torchvision torchaudio transformers timm librosa opencv-python datasets scikit-learn
📥 Dataset Setup
MELD (automatic loading)
from datasets import load_dataset
dataset = load_dataset("declare-lab/MELD")
IEMOCAP / CMU-MOSEI
Download manually:

IEMOCAP: https://sail.usc.edu/iemocap

CMU-MOSEI: https://github.com/A2Zadeh/CMU-MultimodalSDK

Place datasets inside:

datasets/
▶ Running the Model
Real Emotion Prediction
predict_from_files("face.jpg", "audio.wav")
Outputs:

Predicted emotion

Confidence score

Probability distribution

Training
python transformer_fusion_emotion.py
Default configuration:

Epochs: 20

Optimizer: AdamW

Loss: Label-smoothed Cross Entropy

First 5 epochs: frozen ViT + Wav2Vec2

📈 Example Output
Predicted Emotion: Happy
Confidence: 0.87

Class Probabilities:
Neutral: 0.02
Happy: 0.87
Sad: 0.04
Angry: 0.03
Fear: 0.01
Surprise: 0.02
Disgust: 0.01
🧪 Training Strategy
Pretrained ViT + Wav2Vec2 frozen initially

Cross-modal Transformer trained first

Gradual unfreezing for fine-tuning

Feature-level fusion

Attention-based modality weighting

🔬 Research Motivation
Human emotion perception is inherently multimodal. Facial expressions alone or speech alone are insufficient in real-world environments. This project demonstrates how joint audio–visual learning significantly improves emotion recognition robustness.

⚠ Ethical Considerations
Emotion recognition involves sensitive biometric data

On-device inference is encouraged

No identity storage

Intended strictly for academic research

👨‍🎓 Authors
Saaransh Malik
Arnav Juneja

Manipal University Jaipur
Department of Computer Science Engineering
