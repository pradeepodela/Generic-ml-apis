from conductor.client.worker.worker_task import worker_task
from pathlib import Path
from utils.pii import *
from utils.groqApplications import *
from utils.indic import *
from utils.ollamaprocesser import ollamaParserClient
from utils.mistralocrr import *
import requests
import json
import ast
import traceback

# ─────────────────────────────────────────────────────────────
# BASE CONFIG
# ─────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8001"
TIMEOUT = 30

def _post(endpoint: str, payload: dict):
    try:
        print(f"Calling endpoint: {endpoint} with payload: {payload}")
        response = requests.post(f"{BASE_URL}{endpoint}", json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        print(f"Response from {endpoint}: {response}")
        if response.text:
            return ast.literal_eval(response.text)
        else:
            return ast.literal_eval(response)
    except Exception as e:
        print(f"Error calling {endpoint}: {e}")
        print(traceback.format_exc())
        return {"error": str(e), "endpoint": endpoint}


# ─────────────────────────────────────────────────────────────
# ORIGINAL CUSTOM WORKERS (UNCHANGED)
# ─────────────────────────────────────────────────────────────

@worker_task(task_definition_name='myTaskdf')
def worker(name: str) -> str:
    print(f'Worker called with name: {name}')
    return f'hello, {name}'


@worker_task(task_definition_name='OCRTask')
def ocr_worker(URL: str , TYPE: str) -> str:
    print(f'OCR Worker called with URL: {URL} and TYPE: {TYPE}')
    try:
        return ast.literal_eval(ocr_docu(URL, TYPE))
    except Exception as e:
        print(f"Error in OCR processing: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"


@worker_task(task_definition_name='StructuredOCRTask')
def structured_ocr_worker(URL: str) -> str:
    try:
        return ast.literal_eval(structured_ocr(URL))
    except Exception as e:
        print(f"Error in Structured OCR processing: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"


@worker_task(task_definition_name='transcribeTask')
def transcribe_worker(url: str, model: str = "whisper-large-v3-turbo", prompt: str = None, response_format: str = "verbose_json", timestamp_granularities: list = None, language: str = None, temperature: float = 0.0):
    try:
        result = transcribe_audio_from_url_groq(
            url=url,
            model=model,
            response_format=response_format,
            timestamp_granularities=["word", "segment"],
            language=language,
            temperature=temperature
        )
        print(result)
        return ast.literal_eval(result)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"


@worker_task(task_definition_name='queryTask')
def query_worker(query: str) -> str:
    response = LLMChat(query)
    print(f'Query Worker called with query: {query}')
    print(f'Chat completion response: {response}')
    return response


@worker_task(task_definition_name='InidcToEnglish')
def inidc_worker(text:str, src:str, dst:str) -> str:
    try:
        response = requests.post(f"{BASE_URL}/indic-translation", json={"text": text, "src_lang": src, "tgt_lang": dst})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error in Indic Translation processing: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"


@worker_task(task_definition_name='StructurdTexttoJson')
def structured_text_to_json_worker(text: str, template: str) -> str:
    try:
        response = ollamaParserClient(text, template , model='nuextract')
        response = response.replace("<|end-output|>", "")  # Clean model output
        return ast.literal_eval(response)
    except Exception as e:
        print(f"Error in Structured Text to JSON processing: {e}")
        print(traceback.format_exc())
        return f"Error: {str(e)}"


# ─────────────────────────────────────────────────────────────
# NEW WORKERS FOR FASTAPI ENDPOINTS
# ─────────────────────────────────────────────────────────────

@worker_task(task_definition_name='PIIExtraction')
def pii_worker(text: str) -> str:
    return _post("/extract-pii", {"text": text})


@worker_task(task_definition_name='SentimentAnalysis')
def sentiment_analysis_worker(text: str) -> str:
    return _post("/sentiment-analysis", {"text": text})


@worker_task(task_definition_name='FewShotClassify')
def few_shot_classify_worker(text: str, few_shot_examples: dict) -> str:
    return _post("/few-shot-classify", {"text": text, "few_shot_examples": few_shot_examples})


@worker_task(task_definition_name='FewShotClassifyBatch')
def few_shot_classify_batch_worker(texts: list, few_shot_examples: dict) -> str:
    return _post("/few-shot-classify-batch", {"texts": texts, "few_shot_examples": few_shot_examples})


@worker_task(task_definition_name='GemmaProcessing')
def gemma_processing_worker(text: str) -> str:
    return _post("/gemma-processing", {"text": text})


@worker_task(task_definition_name='TextClustering')
def clustering_worker(
    texts: list,
    embedding_model_name: str = "all-MiniLM-L6-v2",
    cluster_algo: str = "agglomerative",
    n_clusters: int = None,
    dim_reduction: str = None,
) -> str:
    return _post("/clustering", {
        "texts": texts,
        "embedding_model_name": embedding_model_name,
        "cluster_algo": cluster_algo,
        "n_clusters": n_clusters,
        "dim_reduction": dim_reduction
    })


@worker_task(task_definition_name='HealthCheck')
def health_check_worker() -> str:
    return _post("/health", {})
