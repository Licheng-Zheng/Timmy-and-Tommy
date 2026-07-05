import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class TextEmbedder:
    def __init__(self, model_name="pritamdeka/S-PubMedBert-MS-MARCO"):
        """
        The embedder needs to be the same model as the one used for embedding into the vector db
        """
        print(f"{model_name} loading")
        self.model = SentenceTransformer(model_name)
        
    def get_embedding(self, text: str) -> torch.Tensor:
        # returns vector of size 768 (the embedding size of the model)
        embedding = self.model.encode(text)
        return torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)

class CognitiveStateTracker(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, num_layers=1):
        """
        This is used to track the user persona 
        
        - input_dim: 768 (We first embed the text with S-PubMedBert, it has an output of 768 dimensions)
        - hidden_dim: 64 (Compress into 64 dimensional Latent Space)
        - num_layers: 1 (It gets from the input to the output in one layer, makes it faster, and it has worked pretty well so far)
        """
        super(CognitiveStateTracker, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # A GRU, or Gated Recurrent Unit. Tells it the input, output and layers, then informs it of the order the data arrives in
        self.gru = nn.GRU(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True
        )

    def forward(self, x, h_prev=None):
        # h_prev is has content is there is previous chat history, if there isn't the latent space is set to all zeroes
        if h_prev is None:
            h_prev = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
            
        # The gru returns the output (the new values), and the new state (which is passed to the next stage of evaluation, so the "personality" persists)
        out, h_new = self.gru(x, h_prev)
        
        return out, h_new
