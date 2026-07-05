import os
import time
import uvicorn
import torch
import numpy as np
from fastapi import FastAPI, Request, HTTPException, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import FileResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
from dotenv import load_dotenv
from sklearn.decomposition import PCA
from groq import Groq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from retrieval_router import CognitiveRouter

# this is the number that the cosine similarity must cross for it to be considered a match to the user query
CONFIDENCE_THRESHOLD = 45.0
MODEL = "llama-3.3-70b-versatile"

# uses dotenv for loading env variables
load_dotenv()

# Determine if the app is running in production or development to set appropriate security rules
ENV = os.getenv("ENVIRONMENT", "development")

app = FastAPI(title="Timmy and Tommy", version="1.0.0")

# I use this for if its running on digital ocean or locally so I don't need to switch any code when I push it and deploy
if ENV == "production":
    print("Running on Digital Ocean")
    cors_origins = ["http://your-server-ip", "https://your-custom-domain.com"] # will set this up when I get a domain
    RATE_LIMIT = "15/minute"
else:
    print("Running locally")
    cors_origins = ["*"]
    RATE_LIMIT = "1000/minute"

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup security headers and request rate limiting to protect the API from abuse
API_KEY = "Timmy!"
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def get_api_key(api_key_header: str = Security(api_key_header)):
    if api_key_header != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return api_key_header

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize database connections, embedding models, and cognitive engine components once at startup
print("Connecting to Qdrant Database")
qdrant_client = QdrantClient(
    url="http://127.0.0.1:6333", 
    api_key=os.getenv("QDRANT_API_KEY") or os.getenv("Qdrant_API_Key")
)

print("Loading Text Embedding Model")
# This model creates 384-dimensional vector embeddings, which matches the configuration of the Qdrant database
embed_model = SentenceTransformer('all-MiniLM-L6-v2') 

print("Loading Cognitive Engine")
router = CognitiveRouter()

print("Calculating 3D Latent Space")
pca = PCA(n_components=3)
centroid_matrix = np.array(list(router.centroids.values()))

if len(centroid_matrix) >= 3:
    pca.fit(centroid_matrix)
    c_3d = pca.transform(centroid_matrix)
else:
    # this file is how I provide the user persona clusters
    print("cluster_centroids.json missing")
    c_3d = np.zeros((4, 3))
    router.centroids = {"Fallback_1": [], "Fallback_2": [], "Fallback_3": [], "Fallback_4": []}

client = Groq()
sessions = {}

class ChatRequest(BaseModel):
    # This is the structure of the request to the api endpoint 
    session_id: str
    message: str
    mode: str = "tommy" 
    search_depth: int = 50 
    persona_override: str = "Auto"

# Serve the main frontend HTML interface
@app.get("/")
async def serve_frontend():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"error": "index.html not found. Ensure it is in the same directory as this script."}

# Endpoint to check system health
@app.get("/api/system-health")
async def system_health():
    try:
        # data on the vector db 
        info = qdrant_client.get_collection(collection_name="nexus_rag_nodes")
        return {
            "status": info.status.name,
            "total_vectors": info.points_count,
            "indexed_vectors": info.indexed_vectors_count
        }
    except Exception as e:
        return {"status": "ERROR", "total_vectors": 0, "indexed_vectors": 0}

# Tells the backend where to route the user requests  
@app.post("/api/chat")
@limiter.limit(RATE_LIMIT) 
async def chat_endpoint(request: Request, payload: ChatRequest, api_key: str = Depends(get_api_key)):
    """
    - Evaluates user persona and cognitive state
    - Retrieves relevant facts from the vector DB
    - Generates an LLM response constrained strictly by retrieved data
    """
    session_id = payload.session_id
    user_input = payload.message
    active_mode = payload.mode
    search_depth = payload.search_depth
    persona_override = payload.persona_override

    if session_id not in sessions:
        sessions[session_id] = {
            "messages": [],
            "h_state_tommy": None,
            "h_state_timmy": None,
            "trajectory_3d": [] 
        }
    
    session = sessions[session_id]
    session["messages"].append({"role": "user", "content": user_input})
    
    
    detected_persona, h_new_tommy = router.get_user_persona(user_input, h_prev=session["h_state_tommy"])
    session["h_state_tommy"] = h_new_tommy
    
    current_coord = h_new_tommy.squeeze().detach().numpy()
    current_3d = pca.transform([current_coord])[0] if len(centroid_matrix) >= 3 else [0.0, 0.0, 0.0]
    session["trajectory_3d"].append(current_3d.tolist())

    # if it is on Tommy mode, it goes through this, otherwise it goes through Tommy which has different things that the text passes through
    if active_mode == "tommy":
        timmy_traits = {"Sarcasm": 0, "Brevity": 0, "Pedantry": 0, "Humor": 0, "Annoyance": 0}
    else:
        timmy_traits, h_new_timmy = router.get_timmy_state(user_input, h_prev=session["h_state_timmy"])
        session["h_state_timmy"] = h_new_timmy

    if persona_override != "Auto":
        detected_persona = persona_override

    t1 = time.time()

    # embeds the user input into a vector embedding so vector search can be performed with it 
    query_vector = embed_model.encode(user_input).tolist()
    search_results = qdrant_client.query_points(
        collection_name="nexus_rag_nodes",
        query=query_vector,
        using="dense",
        limit=search_depth,
        with_payload=True,
        with_vectors=True  
    )
    db_latency = time.time() - t1
    
    # stores the information for displaying on the frontend
    usable_facts = []
    rich_sources = [] 
    simulated_hnsw_logs = []
    avg_confidence = 0.0

    if search_results.points:
        points = search_results.points
        worst_score = points[-1].score if len(points) > 0 else 0
        mid_score = points[len(points)//2].score if len(points) > 1 else worst_score
        best_score = points