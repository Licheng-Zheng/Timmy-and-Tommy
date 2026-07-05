import os
import json
import time
import redis
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# loads my environment variables so I don't accidentally leak secrets
load_dotenv()

# loads in the passwords from my environment file
redis_pw = os.getenv("REDIS_PASSWORD") or os.getenv("Redis_Password")
qdrant_key = os.getenv("QDRANT_API_KEY") or os.getenv("Qdrant_API_Key") or os.getenv("QDrant_API_Key")
redis_host = os.getenv("REDIS_HOST", "127.0.0.1") 

# Redis buffer set up here 
try:
    redis_client = redis.Redis(
        host=redis_host,
        port=6379,
        password=redis_pw,
        decode_responses=True
    )
    redis_client.ping()
    print(f"Redis connection on {redis_host}.")
except Exception as e:
    print(f"Redis connection failed: {e}")
    exit(1)

# Qdrant database connection to store vectors
try:
    qdrant_client = QdrantClient(
        url="http://143.198.183.98:6333", 
        api_key=qdrant_key
    )
    print("Qdrant connected")
except Exception as e:
    print(f"Qdrant connection failed: {e}")
    exit(1)

# this is the local name of my qdrant database
COLLECTION_NAME = "nexus_rag_nodes"

# must have this many new vectors before they are put into the database
BATCH_SIZE = 100  

def start_consuming():
    # Daemon is the background process, it is watching the Redis buffer to see if there are any vectors, if there are it adds them into the db
    print(f"Starting Consumer Daemon. Listening to 'rag_node_queue'")
    batch = []
    
    while True:
        try:
            # I have this block for up to 3 seconds waiting for new data from the Modal fleet before continuing the loop
            item = redis_client.brpop("rag_node_queue", timeout=3)
            
            if item:
                # the item comes back as a tuple, so I need to unpack it and parse the json string
                _, payload_str = item
                payload = json.loads(payload_str)
                
                # I pop the ID and vector out specifically, which leaves the rest of the dictionary perfectly intact to use as metadata
                point_id = payload.pop("chunk_id")
                vector = payload.pop("dense_vector")
                
                # packing it all into structure that Qdrant expects
                point = models.PointStruct(
                    id=point_id,
                    vector={"dense": vector},
                    payload=payload
                )
                batch.append(point)
            
            # this flushes the batch to the database if it hits the size limit, or if the queue happens to be empty but I still have some leftover vectors sitting here
            if len(batch) >= BATCH_SIZE or (not item and len(batch) > 0):
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=batch
                )
                print(f"{len(batch)} new claims added to Qdrant.")
                
                # clearing out the list so I can start a fresh batch
                batch = [] 
                
        except Exception as e:
            print(f"Error processing queue item: {e}")
            time.sleep(5) 

if __name__ == "__main__":
    start_consuming()