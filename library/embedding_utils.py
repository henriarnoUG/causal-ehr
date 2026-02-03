import re
import nltk
nltk.download('punkt_tab')
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

##########################################################################################################

def save_tensors(tensor_list, path):
    embeddings = torch.stack(tensor_list, dim=0)
    torch.save(embeddings, path)

##########################################################################################################

class ModernBERT(torch.nn.Module):
    def __init__(self):
        """Basic encoder class that returns ModernBERT embeddings."""
        super(ModernBERT, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('nomic-ai/modernbert-embed-base')
        self.model = AutoModel.from_pretrained('nomic-ai/modernbert-embed-base')
        self.embedding_dim = self.model.config.hidden_size
    
    # from HuggingFace model card
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
    def forward(self, text):     
        device = self.model.device
        text = ['classification: ' + text]

        # tokenize
        encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {key: value.to(device) for key, value in encoded_input.items()}

        # forward pass
        model_output = self.model(**encoded_input)
        representation = self.mean_pooling(model_output, encoded_input['attention_mask'])
        representation = F.normalize(representation.squeeze(0), p=2, dim=0)

        return representation