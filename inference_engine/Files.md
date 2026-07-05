##### cognitive_state.py
The cognitive tracker for the user persona detector. Includes the `TextEmbedder` (which uses `S-PubMedBert-MS-MARCO` to translate text into 768-dimensional biological vectors) and the base `CognitiveStateTracker` GRU classes used to track conversational states.

##### consumer.py
Handles the downstream consumption of vector data and inference requests. 

##### fastapi_backend.py
This REST API built in FastAPI connects everything together.
- Qdrant semantic retrieval system
- Groq inference
- User RNNs
- Frontend (basically everything)

##### retrieval_router.py
Loads the trained `.pth` model weights and processes incoming prompts through the RNNs. Calculates the distance between the user's current cognitive state and the predefined persona centroids (the cluster_centroids.json). Also applies distance penalities for better results