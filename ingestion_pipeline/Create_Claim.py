import modal
import json
import uuid
import os
import time

# ---------------------------------------------------------------------------
# 1. Environment & Dependencies
# ---------------------------------------------------------------------------
def download_models():
    from transformers import AutoTokenizer, AutoModel
    from sentence_transformers import SentenceTransformer
    import nltk
    
    # 1. NER Model Checkpoint
    ner_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    AutoTokenizer.from_pretrained(ner_checkpoint)
    AutoModel.from_pretrained(ner_checkpoint)
    
    # 2. Embedding Model Checkpoint
    SentenceTransformer('all-MiniLM-L6-v2') 
    
    # 3. Download NLTK tokenizers
    nltk.download('punkt')
    nltk.download('punkt_tab')

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.0+cpu",
        "transformers==4.41.0",
        "pytorch-crf",
        "sentence-transformers", 
        "redis",                 
        "accelerate",
        "requests",
        "beautifulsoup4",
        "nltk",
        extra_index_url="https://download.pytorch.org/whl/cpu"
    )
    .run_function(download_models)
    .add_local_python_source("Scraping_and_Chunking") 
)

app = modal.App("nexus-cloud-pipeline", image=image)

# Volumes for custom trained weights
model_volume = modal.Volume.from_name("bert-details")
MODEL_DIR = "/vol/bert-details"
MODEL_FILE = f"{MODEL_DIR}/best_pubmedbert_crf_model.pt"

LABEL2ID = {
    "O": 0,
    "B-CELL_FUNCTION": 1, "I-CELL_FUNCTION": 2, "L-CELL_FUNCTION": 3, "U-CELL_FUNCTION": 4,
    "B-CELL_TYPE": 5, "I-CELL_TYPE": 6, "L-CELL_TYPE": 7, "U-CELL_TYPE": 8,
    "B-CONDITION": 9, "I-CONDITION": 10, "L-CONDITION": 11, "U-CONDITION": 12,
    "B-LOCATION": 13, "I-LOCATION": 14, "L-LOCATION": 15, "U-LOCATION": 16,
    "B-MOLECULE": 17, "I-MOLECULE": 18, "L-MOLECULE": 19, "U-MOLECULE": 20,
    "B-PATHOGEN": 21, "I-PATHOGEN": 22, "L-PATHOGEN": 23, "U-PATHOGEN": 24,
    "B-PROCESS": 25, "I-PROCESS": 26, "L-PROCESS": 27, "U-PROCESS": 28,
    "B-RECEPTOR": 29, "I-RECEPTOR": 30, "L-RECEPTOR": 31, "U-RECEPTOR": 32,
    "B-TREATMENT": 33, "I-TREATMENT": 34, "L-TREATMENT": 35, "U-TREATMENT": 36,
    "B-VISUAL_PROPERTY": 37, "I-VISUAL_PROPERTY": 38, "L-VISUAL_PROPERTY": 39, "U-VISUAL_PROPERTY": 40
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# ---------------------------------------------------------------------------
# 2. The Unified Extraction Engine
# ---------------------------------------------------------------------------
@app.cls(
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("redis-credentials")],
    timeout=600,
    max_containers=100,
)
class ExtractionEngine:

    @modal.enter()
    def load_models_into_memory(self):
        """Runs once per container boot. Loads all models straight into RAM."""
        import torch
        import torch.nn as nn
        from transformers import AutoTokenizer, AutoModel
        from sentence_transformers import SentenceTransformer
        from torchcrf import CRF

        self.device = "cpu"
        
        # --- Init Redis Connection ---
        import redis
        redis_pw = os.environ.get("REDIS_PASSWORD") or os.environ.get("Redis_Password")
        
        self.redis_client = redis.Redis(
            host="143.198.183.98",
            port=6379,
            password=redis_pw,
            decode_responses=True
        )
        print("[MODAL] Redis connection established.")
        
        # --- Load NER Model ---
        class PubMedBertCRF(nn.Module):
            def __init__(self, model_checkpoint, num_labels):
                super().__init__()
                self.bert = AutoModel.from_pretrained(model_checkpoint)
                self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
                self.crf = CRF(num_tags=num_labels, batch_first=True)

            def forward(self, input_ids, attention_mask, **kwargs):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                return self.classifier(outputs.last_hidden_state)

        ner_checkpoint = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        self.tokenizer = AutoTokenizer.from_pretrained(ner_checkpoint)
        self.ner_model = PubMedBertCRF(ner_checkpoint, num_labels=len(LABEL2ID))
        
        if os.path.exists(MODEL_FILE):
            self.ner_model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device, weights_only=True))
        
        self.ner_model.to(self.device)
        self.ner_model.eval()

        # --- Load Embedding Model ---
        self.embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("[MODAL] All models loaded successfully.")

    @modal.method()
    def process_and_push(self, document_url: str, document_id: str):
        import torch
        import uuid
        import time
        import json
        from Scraping_and_Chunking import scrape, chunk_text, split_chunks_into_sentences
        
        print(f"processing : {document_url}")

        text = scrape(document_url)
        if not text:
            return {"status": "FAILED_SCRAPE", "url": document_url}
            
        chunks = chunk_text(text, source_url=document_url, max_tokens=500)
        sentences = split_chunks_into_sentences(chunks)
        
        if not sentences:
            return {"status": "NO_DATA", "url": document_url}

        extracted_entities = []
        texts_to_embed = []
        chunk_entities_tracker = []

        BOILERPLATE_KEYWORDS = {
            "publisher site", "cite this", "download pdf", "similar articles", 
            "add to collection", "an error occurred", "please try again", "ncbi link", "nbib"
        }
        
        for sentence_obj in sentences:
            sentence_text = sentence_obj["sentence_text"]
            if any(keyword in sentence_text.lower() for keyword in BOILERPLATE_KEYWORDS):
                continue
                
            texts_to_embed.append(sentence_text)
            inputs = self.tokenizer(sentence_text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                emissions = self.ner_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                mask = inputs["attention_mask"].bool()
                mask[:, 0] = True
                predictions = self.ner_model.crf.decode(emissions, mask=mask)[0]

            tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            raw_entities = []
            current_word = ""
            current_label = None

            for token, pred_id in zip(tokens, predictions):
                if token in ["[CLS]", "[SEP]", "[PAD]"]: 
                    continue
                    
                label = ID2LABEL[pred_id]
                clean_token = token.replace("##", "")

                if label.startswith("B-") or label.startswith("U-"):
                    if current_word: 
                        raw_entities.append(f"{current_label}: {current_word}")
                    current_word = clean_token
                    current_label = label.split("-")[1]
                elif label.startswith("I-") or label.startswith("L-"):
                    if token.startswith("##"):
                        current_word += clean_token  
                    else:
                        current_word += " " + clean_token 
                else: 
                    if current_word:
                        raw_entities.append(f"{current_label}: {current_word}")
                        current_word = ""
                        current_label = None

            if current_word:
                raw_entities.append(f"{current_label}: {current_word}")
            extracted_entities.extend(list(set(raw_entities)))
            
            chunk_entities_tracker.append(list(set(raw_entities)))

        if not texts_to_embed:
            return {"status": "ALL_BOILERPLATE", "url": document_url}
            
        extracted_entities = list(set(extracted_entities))
            
        print(f"Computing embeddings for {len(texts_to_embed)} chunks")
        dense_vectors = self.embed_model.encode(texts_to_embed).tolist()

        chunk_ids = [str(uuid.uuid4()) for _ in texts_to_embed]

        for i, (sentence_text, vector, chunk_entities) in enumerate(zip(texts_to_embed, dense_vectors, chunk_entities_tracker)):
            current_id = chunk_ids[i]
            prev_id = chunk_ids[i-1] if i > 0 else None
            next_id = chunk_ids[i+1] if i < len(chunk_ids) - 1 else None
            
            node_data = {
                "chunk_id": current_id,
                "document_id": document_id,
                "content": sentence_text,
                "dense_vector": vector,
                
                "previous_chunk_id": prev_id,
                "next_chunk_id": next_id,
                
                "created_at": int(time.time()),
                "abstraction_level": 0,
                "agreement_count": 1,
                "confidence_score": 0.0,
                "sources": [document_id],
                "is_medoid": True,
                "parent_id": None,
                "child_ids": [],
                "document_summary": "Europe PMC Document",
                "graph_entities": chunk_entities, 
                "graph_edges": []
            }
            
            self.redis_client.lpush("rag_node_queue", json.dumps(node_data))

        print(f"Successfully pushed {len(texts_to_embed)} chunks to buffer.")
        return {"status": "SUCCESS", "url": document_url, "chunks_pushed": len(texts_to_embed)}