import torch
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nlp.dataset import preprocess_text, OCCASIONS, WEATHER_CLASSES
from nlp.model import WardrobeNLPModel
from nlp.dataset import Vocabulary

class NLPInference:
    def __init__(self, model_dir='nlp/saved_models'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load vocabulary
        with open(os.path.join(model_dir, 'vocab.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
            
        # Load model
        self.model = WardrobeNLPModel(
            vocab_size=self.vocab.vocab_size,
            embed_dim=64,
            hidden_dim=128,
            num_occasions=len(OCCASIONS),
            num_weather=len(WEATHER_CLASSES)
        )
        self.model.load_state_dict(torch.load(os.path.join(model_dir, 'nlp_model.pth'), map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.idx2occasion = {idx: occ for idx, occ in enumerate(OCCASIONS)}
        self.idx2weather = {idx: wea for idx, wea in enumerate(WEATHER_CLASSES)}
        
        # Simple mapping from our weather classes to API's expected temperature
        self.weather_to_temp = {
            "hot": 30,
            "mild": 20,
            "cold": 10
        }

    def predict(self, text: str):
        # Preprocess and encode text
        tokens = preprocess_text(text)
        encoded = [self.vocab.word2idx.get(token, self.vocab.word2idx["<UNK>"]) for token in tokens]
        
        # Convert to tensor and add batch dimension
        input_ids = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            occ_logits, wea_logits = self.model(input_ids)
            
            predicted_occ_idx = torch.argmax(occ_logits, dim=1).item()
            predicted_wea_idx = torch.argmax(wea_logits, dim=1).item()
            
        occasion = self.idx2occasion[predicted_occ_idx]
        weather_class = self.idx2weather[predicted_wea_idx]
        temperature = self.weather_to_temp[weather_class]
        
        return {
            "occasion": occasion,
            "weather_class": weather_class,
            "weather": {"temperature": temperature}
        }

if __name__ == "__main__":
    # Test script locally
    infer = NLPInference()
    texts = [
        "I need a formal outfit for tonight, it's very cold.",
        "What should I wear to the gym? It's pretty hot outside."
    ]
    for t in texts:
        print(f"Input: {t}")
        print(f"Output: {infer.predict(t)}\n")
