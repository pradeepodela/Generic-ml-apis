# pii_service.py - Run this separately
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import asyncio
import time
from typing import Dict, List, Optional
from contextlib import asynccontextmanager
from utils.clustering_service import TextClusteringService
# Import your existing utilities
from utils.indic import *
from utils.GemmaPraocessing import *
from utils.SentimentAnalysisProcesser import *
# Import the optimized few-shot classifier
from utils.fast_few_shot_classifier import FastFewShotClassifier
import ast
# Global classifier instance
classifier = None
svc = TextClusteringService()


#Sentiment Analysis Request Model
class SentimentAnalysisRequest(BaseModel):
    text: str

#clustering 
class ClusteringRequest(BaseModel):
    texts: List[str]
    embedding_model_name: str="all-MiniLM-L6-v2",
    cluster_algo: str = "agglomerative", # "kmeans"|"agglomerative"|"dbscan"
    n_clusters: Optional[int] = None,  # Only used for kmeans/agglomerative
    dim_reduction: Optional[str] = None  # "pca"|"umap"

# Pydantic models for request validation
class FewShotRequest(BaseModel):
    text: str
    few_shot_examples: Dict[str, List[str]]

class BatchFewShotRequest(BaseModel):
    texts: List[str]
    few_shot_examples: Dict[str, List[str]]

class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class GemmaRequest(BaseModel):
    text: str

class PIIRequest(BaseModel):
    text: str

# Lifespan manager for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global classifier
    print("Starting up FastAPI service...")
    
    # Initialize classifier and start background loading
    classifier = FastFewShotClassifier(
        model_name="jinaai/jina-embeddings-v2-base-en",
        cache_dir="./model_cache"
    )
    
    # Start loading model in background
    print("Starting model loading in background...")
    classifier.preload_model_background()
    
    # Optionally prepare common few-shot examples
    # This would make the first request even faster
    common_examples = {
        "positive": [
            "I love this product!",
            "This is the best service I've ever used.",
            "Absolutely fantastic experience.",
            "Highly recommend to everyone!",
            "I'm very satisfied with my purchase."
        ],
        "negative": [
            "I hate this product.",
            "This service was terrible.",
            "I'm very disappointed with my experience.",
            "Would not recommend to anyone.",
            "This was a waste of money."
        ]
    }
    
    # Prepare common examples in background
    async def prepare_common_examples():
        try:
            await classifier.prepare_few_shot_examples(common_examples)
            print("Common examples prepared!")
        except Exception as e:
            print(f"Error preparing common examples: {e}")
    
    # Start preparation in background
    asyncio.create_task(prepare_common_examples())
    
    print("FastAPI service started!")
    yield
    
    # Shutdown
    print("Shutting down...")

# Create FastAPI app with lifespan
app = FastAPI(
    title="PII Service with Few-Shot Classification",
    description="FastAPI service with PII extraction, translation, and few-shot classification",
    version="1.0.0",
    lifespan=lifespan
)

@app.post("/sentiment-analysis")
async def sentiment_analysis_endpoint(request: SentimentAnalysisRequest):
    """
    Endpoint for sentiment analysis.
    
    Example request:
    {
        "text": "I love this product!"
    }
    """
    try:
        response = analyze(request.text)
        res = ast.literal_eval(response)  # Convert string to dict
        return res
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis error: {str(e)}")

@app.post("/few-shot-classify")
async def few_shot_classify_endpoint(request: FewShotRequest):
    """
    Single text classification endpoint using few-shot learning.
    
    Example request:
    {
        "text": "I love this product!",
        "few_shot_examples": {
            "positive": ["Great product!", "I love it!", "Amazing quality!"],
            "negative": ["Terrible product!", "I hate it!", "Poor quality!"]
        }
    }
    """
    try:
        start_time = time.time()
        result = await classifier.classify(request.text, request.few_shot_examples)
        end_time = time.time()
        
        result["processing_time"] = round(end_time - start_time, 4)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

@app.post("/few-shot-classify-batch")
async def few_shot_classify_batch_endpoint(request: BatchFewShotRequest):
    """
    Batch text classification endpoint for multiple texts.
    
    Example request:
    {
        "texts": ["I love this!", "This is terrible!", "Great product!"],
        "few_shot_examples": {
            "positive": ["Great product!", "I love it!"],
            "negative": ["Terrible!", "I hate it!"]
        }
    }
    """
    try:
        start_time = time.time()
        results = await classifier.classify_batch(request.texts, request.few_shot_examples)
        end_time = time.time()
        
        return {
            "results": results,
            "total_processing_time": round(end_time - start_time, 4),
            "average_time_per_text": round((end_time - start_time) / len(request.texts), 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch classification error: {str(e)}")

@app.post("/extract-pii")
async def extract_pii_endpoint(request: PIIRequest):
    """
    PII extraction endpoint.
    
    Example request:
    {
        "text": "My name is John Doe and my email is john.doe@example.com"
    }
    """
    try:
        from utils.pii import extract_pii
        result = extract_pii(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PII extraction error: {str(e)}")

@app.post("/health")
async def health_check():
    """
    Health check endpoint that shows service and model status.
    """
    global classifier
    
    if classifier:
        model_status = classifier.get_status()
    else:
        model_status = {"error": "Classifier not initialized"}
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "classifier_status": model_status
    }

@app.post("/indic-translation")
async def indic_translation_endpoint(request: TranslationRequest):
    """
    Indic translation endpoint.
    
    Example request:
    {
        "text": "Hello world",
        "src_lang": "en",
        "tgt_lang": "hi"
    }
    """
    try:
        print(f"Translation request: {request.text}, {request.src_lang} -> {request.tgt_lang}")
        result = TransulationWorkerIndictoEnglish([request.text], request.src_lang, request.tgt_lang)
        return {"translation": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation error: {str(e)}")

@app.post("/gemma-processing")
async def gemma_processing(request: GemmaRequest):
    """
    Gemma processing endpoint.
    
    Example request:
    {
        "text": "Generate a summary of machine learning"
    }
    """
    try:
        response = generate_text(request.text)  # Fixed the dict access bug from original
        print(f"Response from generate_text: {response}")
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemma processing error: {str(e)}")

@app.post("/clustering")
def clustering_endpoint(request: ClusteringRequest):
    """
    Clustering endpoint for text data.
    # "kmeans"|"agglomerative"|"dbscan"
    #"pca"|"umap"
    # make sure to make dim_reduction None if the input size is less than 16
    Example request:
    {
        "texts":  ["I love programming in Python!","Python is great for data science.","I enjoy hiking and outdoor activities.","Hiking in the mountains is refreshing.","Data science involves statistics and machine learning.","Machine learning is a subset of AI."," love to travel to varanasi"],
        "embedding_model_name": "all-MiniLM-L6-v2",
        "cluster_algo": "kmeans", 
        "n_clusters": 2,
        "dim_reduction": null
    }
    """
    try:
        result = svc.cluster(
            texts=request.texts,
            embedding_model_name=request.embedding_model_name,
            cluster_algo=request.cluster_algo,
            n_clusters=request.n_clusters,
            dim_reduction=request.dim_reduction
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "PII Service with Few-Shot Classification API",
        "version": "1.0.0",
        "endpoints": {
            "few_shot_classify": "/few-shot-classify",
            "few_shot_classify_batch": "/few-shot-classify-batch",
            "extract_pii": "/extract-pii",
            "indic_translation": "/indic-translation",
            "gemma_processing": "/gemma-processing",
            "health": "/health"
        },
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8001,
        reload=False,  # Disable reload for production
        workers=1  # Single worker for model consistency
    )