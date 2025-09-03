# fast_few_shot_classifier.py
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
import time
import asyncio
import threading
from typing import Dict, List, Optional
import os
from pathlib import Path

class FastFewShotClassifier:
    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        """Ultra-fast few-shot classifier with aggressive optimizations."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model will be loaded lazily or in background
        self.model = None
        self.is_loading = False
        self.is_ready = False
        
        # Cache for embeddings
        self._embedding_cache = {}
        self._examples_cache = {}
        
        # Set environment for faster loading
        if cache_dir:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_path)
            os.environ['TRANSFORMERS_CACHE'] = str(cache_path)
        
        # Optimize torch settings
        torch.set_num_threads(min(4, os.cpu_count()))
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def _load_model(self):
        """Load model with optimizations."""
        if self.model is not None:
            return
        
        self.is_loading = True
        print(f"Loading model {self.model_name}...")
        start_time = time.time()
        
        try:
            # Load with optimizations
            self.model = SentenceTransformer(
                self.model_name,
                device=self.device,
                trust_remote_code=True,
                cache_folder=self.cache_dir
            )
            self.model.eval()
            
            # Use half precision for speed on GPU
            if self.device.type == 'cuda':
                self.model.half()
            
            # Warm up the model with a dummy encoding
            with torch.no_grad():
                self.model.encode(["warmup"], convert_to_tensor=True, show_progress_bar=False)
            
            self.is_ready = True
            end_time = time.time()
            print(f"Model loaded in {end_time - start_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        finally:
            self.is_loading = False
    
    async def ensure_model_loaded(self):
        """Ensure model is loaded, wait if currently loading."""
        if self.model is None and not self.is_loading:
            # Load in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model)
        
        # Wait for loading to complete
        while self.is_loading:
            await asyncio.sleep(0.1)
        
        if not self.is_ready:
            raise Exception("Model not ready")
    
    def preload_model_background(self):
        """Start loading model in background thread."""
        if self.model is None and not self.is_loading:
            thread = threading.Thread(target=self._load_model, daemon=True)
            thread.start()
    
    def load_model_sync(self):
        """Load model synchronously."""
        self._load_model()
    
    def _generate_cache_key(self, few_shot_examples: Dict[str, List[str]]) -> str:
        """Generate cache key for few-shot examples."""
        return str(hash(str(sorted([(k, tuple(v)) for k, v in few_shot_examples.items()]))))
    
    async def prepare_few_shot_examples(self, few_shot_examples: Dict[str, List[str]]):
        """Pre-compute embeddings for few-shot examples."""
        await self.ensure_model_loaded()
        
        cache_key = self._generate_cache_key(few_shot_examples)
        
        if cache_key in self._embedding_cache:
            return
        
        print("Preparing few-shot embeddings...")
        start_time = time.time()
        embeddings_dict = {}
        
        with torch.no_grad():
            for label, examples in few_shot_examples.items():
                embeddings = self.model.encode(
                    examples,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                    batch_size=32,
                    normalize_embeddings=True
                )
                embeddings_dict[label] = embeddings
        
        self._embedding_cache[cache_key] = embeddings_dict
        self._examples_cache[cache_key] = few_shot_examples
        
        end_time = time.time()
        print(f"Few-shot embeddings prepared in {end_time - start_time:.4f} seconds")
    
    def prepare_few_shot_examples_sync(self, few_shot_examples: Dict[str, List[str]]):
        """Synchronous version of prepare_few_shot_examples."""
        if self.model is None:
            self.load_model_sync()
        
        cache_key = self._generate_cache_key(few_shot_examples)
        
        if cache_key in self._embedding_cache:
            return
        
        print("Preparing few-shot embeddings...")
        start_time = time.time()
        embeddings_dict = {}
        
        with torch.no_grad():
            for label, examples in few_shot_examples.items():
                embeddings = self.model.encode(
                    examples,
                    convert_to_tensor=True,
                    device=self.device,
                    show_progress_bar=False,
                    batch_size=32,
                    normalize_embeddings=True
                )
                embeddings_dict[label] = embeddings
        
        self._embedding_cache[cache_key] = embeddings_dict
        self._examples_cache[cache_key] = few_shot_examples
        
        end_time = time.time()
        print(f"Few-shot embeddings prepared in {end_time - start_time:.4f} seconds")
    
    async def classify(self, text: str, few_shot_examples: Dict[str, List[str]]) -> Dict:
        """Fast classification with caching."""
        await self.ensure_model_loaded()
        
        # Prepare examples if needed
        cache_key = self._generate_cache_key(few_shot_examples)
        if cache_key not in self._embedding_cache:
            await self.prepare_few_shot_examples(few_shot_examples)
        
        cached_embeddings = self._embedding_cache[cache_key]
        
        # Encode input text
        with torch.no_grad():
            text_emb = self.model.encode(
                text,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        # Find best match
        best_score = -1
        best_label = None
        scores_dict = {}
        
        for label, example_embs in cached_embeddings.items():
            scores = torch.mm(text_emb.unsqueeze(0), example_embs.t())
            max_score = scores.max().item()
            scores_dict[label] = round(max_score, 4)
            
            if max_score > best_score:
                best_score = max_score
                best_label = label
        
        return {
            "prediction": best_label,
            "confidence": round(best_score, 4),
            "scores": scores_dict
        }
    
    def classify_sync(self, text: str, few_shot_examples: Dict[str, List[str]]) -> Dict:
        """Synchronous classification."""
        if self.model is None:
            self.load_model_sync()
        
        # Prepare examples if needed
        cache_key = self._generate_cache_key(few_shot_examples)
        if cache_key not in self._embedding_cache:
            self.prepare_few_shot_examples_sync(few_shot_examples)
        
        cached_embeddings = self._embedding_cache[cache_key]
        
        # Encode input text
        with torch.no_grad():
            text_emb = self.model.encode(
                text,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
                normalize_embeddings=True
            )
        
        # Find best match
        best_score = -1
        best_label = None
        scores_dict = {}
        
        for label, example_embs in cached_embeddings.items():
            scores = torch.mm(text_emb.unsqueeze(0), example_embs.t())
            max_score = scores.max().item()
            scores_dict[label] = round(max_score, 4)
            
            if max_score > best_score:
                best_score = max_score
                best_label = label
        
        return {
            "prediction": best_label,
            "confidence": round(best_score, 4),
            "scores": scores_dict
        }
    
    async def classify_batch(self, texts: List[str], few_shot_examples: Dict[str, List[str]]) -> List[Dict]:
        """Batch classification for multiple texts."""
        await self.ensure_model_loaded()
        
        cache_key = self._generate_cache_key(few_shot_examples)
        if cache_key not in self._embedding_cache:
            await self.prepare_few_shot_examples(few_shot_examples)
        
        cached_embeddings = self._embedding_cache[cache_key]
        
        # Encode all texts at once
        with torch.no_grad():
            text_embs = self.model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
        
        results = []
        for i, text_emb in enumerate(text_embs):
            best_score = -1
            best_label = None
            scores_dict = {}
            
            for label, example_embs in cached_embeddings.items():
                scores = torch.mm(text_emb.unsqueeze(0), example_embs.t())
                max_score = scores.max().item()
                scores_dict[label] = round(max_score, 4)
                
                if max_score > best_score:
                    best_score = max_score
                    best_label = label
            
            results.append({
                "text": texts[i],
                "prediction": best_label,
                "confidence": round(best_score, 4),
                "scores": scores_dict
            })
        
        return results
    
    def classify_batch_sync(self, texts: List[str], few_shot_examples: Dict[str, List[str]]) -> List[Dict]:
        """Synchronous batch classification."""
        if self.model is None:
            self.load_model_sync()
        
        cache_key = self._generate_cache_key(few_shot_examples)
        if cache_key not in self._embedding_cache:
            self.prepare_few_shot_examples_sync(few_shot_examples)
        
        cached_embeddings = self._embedding_cache[cache_key]
        
        # Encode all texts at once
        with torch.no_grad():
            text_embs = self.model.encode(
                texts,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False,
                batch_size=32,
                normalize_embeddings=True
            )
        
        results = []
        for i, text_emb in enumerate(text_embs):
            best_score = -1
            best_label = None
            scores_dict = {}
            
            for label, example_embs in cached_embeddings.items():
                scores = torch.mm(text_emb.unsqueeze(0), example_embs.t())
                max_score = scores.max().item()
                scores_dict[label] = round(max_score, 4)
                
                if max_score > best_score:
                    best_score = max_score
                    best_label = label
            
            results.append({
                "text": texts[i],
                "prediction": best_label,
                "confidence": round(best_score, 4),
                "scores": scores_dict
            })
        
        return results
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._embedding_cache.clear()
        self._examples_cache.clear()
        print("Cache cleared")
    
    def get_status(self) -> Dict:
        """Get classifier status."""
        return {
            "model_loaded": self.model is not None,
            "is_loading": self.is_loading,
            "is_ready": self.is_ready,
            "device": str(self.device),
            "cached_examples": len(self._embedding_cache)
        }

# Example usage
if __name__ == "__main__":
    # Test the classifier
    classifier = FastFewShotClassifier(
        model_name="jinaai/jina-embeddings-v2-base-en",
        cache_dir="./model_cache"
    )
    
    # Example few-shot examples
    few_shot_examples = {
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
    
    # Synchronous usage
    print("Testing synchronous classification...")
    
    # Load model and prepare examples
    classifier.load_model_sync()
    classifier.prepare_few_shot_examples_sync(few_shot_examples)
    
    # Single classification
    text = "I am extremely happy with the quality of this product!"
    start_time = time.time()
    result = classifier.classify_sync(text, few_shot_examples)
    end_time = time.time()
    
    print(f"Classification: {result}")
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    
    # Batch classification
    test_texts = [
        "This product is amazing!",
        "I'm not happy with this purchase.",
        "Great quality and fast delivery.",
        "Terrible customer service."
    ]
    
    start_time = time.time()
    batch_results = classifier.classify_batch_sync(test_texts, few_shot_examples)
    end_time = time.time()
    
    print(f"\nBatch results: {batch_results}")
    print(f"Batch time: {end_time - start_time:.4f} seconds")
    print(f"Average per text: {(end_time - start_time) / len(test_texts):.4f} seconds")