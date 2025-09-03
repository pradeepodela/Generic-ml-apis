from typing import Dict, List, Union
import torch
from gliner import GLiNER
import gc
import time


import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# Global model cache to avoid reloading
_model_cache = None

def get_model():
    """Get cached model or load new one with optimizations"""
    global _model_cache
    
    if _model_cache is None:
        print("Loading GLiNER model for the first time...")
        try:
            # Load model with CPU optimization for worker environment
            _model_cache = GLiNER.from_pretrained(
                "urchade/gliner_multi_pii-v1",
                torch_dtype="auto",  # Use float32 for stability
                device_map="cpu"  # Force CPU to avoid GPU conflicts
            )
            print("Model loaded successfully and cached")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    return _model_cache

def extract_pii(
    text: str, 
    labels: str = None, 
    threshold: float = 0.5, 
    nested_ner: bool = False
) -> Dict[str, Union[str, List[Dict]]]:
    """
    Extract personally identifiable information (PII) from text using GLiNER model.
    """
    print(f"Starting PII extraction for text: {text[:50]}...")
    
    # Input validation
    if not text or not text.strip():
        return {"text": text, "entities": []}
    
    try:
        # Get cached model
        print("Getting model...")
        model = get_model()
        print("Model retrieved successfully")
        
        # Default PII labels if none provided
        if labels is None:
            labels = [
                "person", "organization", "address", "email", "phone number", 
                "social security number", "credit card number", "passport number",
                "driver license", "bank account number", "date of birth",
                "medical record number", "insurance policy number", 
                "employee ID number", "tax ID number" , "order value" , "product name",
            ]
        else:
            labels = [label.strip() for label in labels.split(",")]
         
        print(f"Using {len(labels)} labels with threshold {threshold}")
        
        # Optimize text length to avoid memory issues
        max_length = 5000  # Limit text length
        if len(text) > max_length:
            text = text[:max_length]
            print(f"Text truncated to {max_length} characters")
        
        print("Starting entity prediction...")
        
        # Extract entities with timeout handling
        start_time = time.time()
        entities = model.predict_entities(
            text, 
            labels, 
            flat_ner=not nested_ner, 
            threshold=threshold
        )
        print(f"Entity prediction completed in {time.time() - start_time:.2f} seconds")
        print(f"Entity prediction completed. Found {len(entities)} raw entities")
        
        # Format results
        formatted_entities = []
        for entity in entities:
            try:
                formatted_entity = {
                    "entity": entity.get("label", "unknown"),
                    "word": entity.get("text", ""),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "score": float(entity.get("score", 0.0)),
                }
                formatted_entities.append(formatted_entity)
                print(f"  Found: '{formatted_entity['word']}' -> {formatted_entity['entity']} "
                      f"(score: {formatted_entity['score']:.3f})")
            except Exception as e:
                print(f"Error formatting entity {entity}: {e}")
                continue
        
        results = {
            "text": text,
            "entities": formatted_entities,
        }
        
        print(f"PII extraction completed successfully. Final count: {len(formatted_entities)} entities")
        
        # Clean up memory
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"Error in extract_pii: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        
        return {
            "text": text,
            "entities": [],
            "error": str(e)
        }


# Example usage
if __name__ == "__main__":
    sample_text = "John Smith, from London, teaches mathematics at Royal Academy located at 25 King's Road. His employee ID is UK-987654-321 and he has been working there since 2015."
    
    # Example 1: Using default labels
    results = extract_pii(sample_text)
    print("Example 1: Using default labels")
    print(f"Text: {results['text']}")
    print("Detected entities:")
    for entity in results["entities"]:
        print(f"  {entity['word']} => {entity['entity']} (positions {entity['start']}-{entity['end']})")
    
    # Example 2: Using custom labels with different threshold
    custom_labels = "person, profession, organization, address, employee ID number"
    results = extract_pii(sample_text, labels=custom_labels, threshold=0.3)
    print("\nExample 2: Using custom labels with lower threshold")
    print(f"Text: {results['text']}")
    print("Detected entities:")
    for entity in results["entities"]:
        print(f"  {entity['word']} => {entity['entity']} (positions {entity['start']}-{entity['end']})")