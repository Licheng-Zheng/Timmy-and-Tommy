##### cluster_centroids.json 
These are the different embeddings for each user persona that has been embedded. After user interaction, its embedding is calculated and the nearest persona is found, which changes how the model interacts. 

The below files should be put into a google drive if they were large, but they are both less than a megabyte, so I'm going to put it into the repo normally. 
##### cognitive_rnn_weights.pth 
This is the RNN used to evaluate the user's persona. Here is the class for it: 
```python 
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
```


##### timmy_personality_weights.pth 
The RNN weights used to change Timmy's cognitive state. Here is the class: 

```python
class TimmyPersonalityRNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, num_traits=4):
        super(TimmyPersonalityRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_traits)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h_prev=None):
        if h_prev is None:
            h_prev = torch.zeros(1, x.size(0), self.hidden_dim).to(x.device)
        out, h_new = self.gru(x, h_prev)
        raw_traits = self.fc(out)
        return self.sigmoid(raw_traits), h_new
```