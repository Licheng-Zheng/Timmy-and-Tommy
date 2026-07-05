import os
import json
import torch
import torch.nn as nn
from cognitive_state import TextEmbedder, CognitiveStateTracker

# this is the neural network that calculates timmy's emotional traits based on the input
class TimmyPersonalityRNN(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, num_traits=5):
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

class CognitiveRouter:
    def __init__(self):
        """
        handles the math for routing tommy and timmy so I don't need any database connections in this file
        """
        print("Loading Cognitive Router on retrieval_router.py")
        self.embedder = TextEmbedder()
        self.rnn = CognitiveStateTracker(input_dim=768, hidden_dim=64)
        
        # loads in the rnn weights
        weights_path = "cognitive_rnn_weights.pth"
        if os.path.exists(weights_path):
            # Only the weights are loaded in because I have the class for the model in cognitive_state.py                 
            self.rnn.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu'), weights_only=True))
            self.rnn.eval()
        
        # this is where the embeddings for the user personas are held
        centroids_path = "cluster_centroids.json"
        self.centroids = {}
        if os.path.exists(centroids_path):
            with open(centroids_path, 'r') as f:
                self.centroids = json.load(f)

        # this section wakes up timmy's emotional tracking model
        print("Timmys personality RNN in retrieval_router.py")
        
        # 5 traits total
        self.timmy_rnn = TimmyPersonalityRNN(input_dim=768, hidden_dim=64, num_traits=5)
        timmy_weights_path = "timmy_personality_weights.pth"
        if os.path.exists(timmy_weights_path):
            self.timmy_rnn.load_state_dict(torch.load(timmy_weights_path, map_location=torch.device('cpu'), weights_only=True))
            self.timmy_rnn.eval()
            print("personality loaded in retrieval_router.py")

    def get_user_persona(self, user_input: str, h_prev: torch.Tensor):
        x = self.embedder.get_embedding(user_input)
        if x.dim() == 2: x = x.unsqueeze(0)

        with torch.no_grad():
            out, h_new = self.rnn(x, h_prev=h_prev)

        current_coord = h_new.squeeze()
        best_persona = "Unknown"
        min_distance = float('inf')

        # these are the artificial bias controls for the personas
        # numbers over 1.0 push the centroid away so it's harder to trigger
        # numbers under 1.0 pull it closer so it triggers easier
        distance_penalties = {
            "Layman_Curious": 1.4, # this one was getting triggered too easily 
            "Medical_Expert": 0.9 # triggered too infrequently 
        }

        for persona_name, centroid_coords in self.centroids.items():
            centroid_tensor = torch.tensor(centroid_coords, dtype=torch.float32)
            distance = torch.norm(current_coord - centroid_tensor).item()

            # the distance penalty outlined above
            if persona_name in distance_penalties:
                distance *= distance_penalties[persona_name]

            if distance < min_distance:
                min_distance = distance
                best_persona = persona_name

        return best_persona, h_new

    def get_timmy_state(self, user_input: str, h_prev: torch.Tensor):
        x = self.embedder.get_embedding(user_input)
        if x.dim() == 2: x = x.unsqueeze(0)
        
        with torch.no_grad():
            traits, h_new = self.timmy_rnn(x, h_prev=h_prev)
            
        # pulls the 5 floats out of the tensor so they can be mapped to the traits
        traits_list = traits.squeeze().tolist()

        return {
            "Sarcasm": traits_list[0],
            "Brevity": traits_list[1],
            "Pedantry": traits_list[2],
            "Humor": traits_list[3],
            "Annoyance": traits_list[4] 
        }, h_new