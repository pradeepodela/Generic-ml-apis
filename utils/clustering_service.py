# clustering_service.py  (v2 – UMAP & auto‑k)
from __future__ import annotations
import threading, json
from typing import List, Dict, Any, Optional
import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

w
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, homogeneity_score,
    completeness_score, v_measure_score, fowlkes_mallows_score,
)
from sklearn.preprocessing import LabelEncoder

# Optional: UMAP for powerful non‑linear reduction

import umap.umap_ as umap       # pip install umap-learn
_HAS_UMAP = True


# --------------------------------------------------------------------------- #
# GLOBAL CACHES                                                              #
# --------------------------------------------------------------------------- #
_model_lock = threading.Lock()
_EMBEDDING_MODELS: dict[str, SentenceTransformer] = {}

def _get_embedding_model(name: str) -> SentenceTransformer:
    with _model_lock:
        if name not in _EMBEDDING_MODELS:
            _EMBEDDING_MODELS[name] = SentenceTransformer(name)
        return _EMBEDDING_MODELS[name]

# --------------------------------------------------------------------------- #
# MAIN SERVICE                                                               #
# --------------------------------------------------------------------------- #
class TextClusteringService:
    """
    Text clustering with strong embeddings, dimensionality reduction, and auto‑k.
    """

    # ----------------------------- public API --------------------------------
    def cluster(
        self,
        texts: List[str],
        embedding_model_name: str = "all-mpnet-base-v2",
        cluster_algo: str = "kmeans",           # "kmeans"|"agglomerative"|"dbscan"
        n_clusters: Optional[int] = None,       # if None → auto‑k search
        dim_reduction: Optional[str] = "umap",  # "umap"|"pca"|None
        n_components: int = 20,                 # target dims after reduction
        true_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        self.starttime = time.time()
        self._validate_inputs(texts, true_labels, n_clusters, cluster_algo)

        # 1. Embeddings
        emb_model = _get_embedding_model(embedding_model_name)
        embeddings = emb_model.encode(texts, show_progress_bar=False)

        # 2. Dimensionality reduction (optional)
        embeddings = self._reduce_dim(
            embeddings, method=dim_reduction, n_components=n_components
        )

        # 3. Clustering
        cluster_labels, chosen_k, cluster_params = self._run_clustering(
            embeddings, cluster_algo, n_clusters
        )

        # 4. Metrics & summaries
        metrics = self._calculate_metrics(embeddings, cluster_labels, true_labels)
        cluster_json = self._summarise_clusters(texts, cluster_labels, true_labels)
        end_time = time.time()
        return {
            "embedding_model": embedding_model_name,
            "cluster_algo": cluster_algo.lower(),
            "cluster_params": cluster_params,
            "dim_reduction": dim_reduction,
            "metrics": metrics,
            "clusters": cluster_json,
            "processing_time": round(end_time - self.starttime, 2)
        }

    # ----------------------------- helpers -----------------------------------
    @staticmethod
    def _validate_inputs(texts, true_labels, n_clusters, algo):
        if len(texts) < 4:
            raise ValueError("Need at least 4 texts")
        if true_labels and len(texts) != len(true_labels):
            raise ValueError("texts and true_labels length mismatch")
        if algo.lower() in {"kmeans", "agglomerative"} and n_clusters is not None:
            if n_clusters < 2:
                raise ValueError("n_clusters must be >=2")

    @staticmethod
    def _reduce_dim(emb, method="umap", n_components=20):
        if method == "umap":
            if not _HAS_UMAP:
                raise RuntimeError("UMAP not installed; `pip install umap-learn`")
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            return reducer.fit_transform(emb)
        if method == "pca":
            return PCA(n_components=n_components, random_state=42).fit_transform(emb)
        return emb  # None → no reduction

    # ------------ clustering (with optional auto‑k search) -------------------
    def _run_clustering(self, emb, algo: str, n_clusters: Optional[int]):
        algo = algo.lower()
        if algo == "dbscan":
            model = DBSCAN()
            labels = model.fit_predict(emb)
            return labels, None, {"eps": model.eps, "min_samples": model.min_samples}

        # KMeans or Agglomerative
        if n_clusters is None:
            best_k, best_score, best_labels = None, -1, None
            search_range = range(2, min(10, len(emb) // 2) + 1)
            for k in search_range:
                labels = self._fit_predict(algo, emb, k)
                if len(set(labels)) < 2:
                    continue
                score = silhouette_score(emb, labels)
                if score > best_score:
                    best_k, best_score, best_labels = k, score, labels
            if best_labels is None:
                raise RuntimeError("Auto‑k failed to find a valid clustering")
            return best_labels, best_k, {"n_clusters": best_k}
        else:
            labels = self._fit_predict(algo, emb, n_clusters)
            return labels, n_clusters, {"n_clusters": n_clusters}

    @staticmethod
    def _fit_predict(algo, emb, k):
        if algo == "kmeans":
            return KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)
        return AgglomerativeClustering(n_clusters=k).fit_predict(emb)

    # -------------------- metrics / summaries --------------------------------
    @staticmethod
    def _calculate_metrics(emb, labels, true):
        metrics = {
            "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
            "silhouette": float(
                silhouette_score(emb, labels)
            ) if len(set(labels)) > 1 else -1.0,
        }
        if true:
            le = LabelEncoder().fit(true)
            y_true = le.transform(true)
            metrics.update({
                "adjusted_rand": float(adjusted_rand_score(y_true, labels)),
                "homogeneity": float(homogeneity_score(y_true, labels)),
                "completeness": float(completeness_score(y_true, labels)),
                "v_measure": float(v_measure_score(y_true, labels)),
                "fowlkes_mallows": float(fowlkes_mallows_score(y_true, labels)),
            })
        return metrics

    @staticmethod
    def _summarise_clusters(texts, labels, true):
        df = pd.DataFrame({"text": texts, "cluster": labels})
        if true:
            df["true"] = true
        out: List[Dict[str, Any]] = []
        for cid in sorted(set(labels)):
            if cid == -1:  # DBSCAN noise
                continue
            grp = df[df.cluster == cid]
            item = {"cluster_id": int(cid), "texts": grp.text.tolist()}
            if true:
                maj = grp.true.mode()[0]
                item["label"] = maj
                item["confidence"] = round(float((grp.true == maj).mean()), 3)
            out.append(item)
        return out


if __name__ == "__main__":
    # sample_dataset.py
    test_cases = [
        # -------------------- SUPPORT (12) --------------------
        {"text": "My order #7123 hasn’t shipped yet—could you check the status?",                  "expected": ["Support"]},
        {"text": "The activation link in my email has expired. Please resend it.",                "expected": ["Support"]},
        {"text": "We’re getting a 401 error when we hit /api/v2/report. Any idea why?",           "expected": ["Support"]},
        {"text": "I accidentally deleted a project. Is there a way to restore it?",              "expected": ["Support"]},
        {"text": "Push notifications stopped working on Android after the latest update.",        "expected": ["Support"]},
        {"text": "Our SSO login keeps redirecting in a loop—need help ASAP.",                     "expected": ["Support"]},
        {"text": "The invoice PDF we downloaded is blank. Can you regenerate it?",               "expected": ["Support"]},
        {"text": "Payment failed but the card was charged twice—please refund one.",              "expected": ["Support"]},
        {"text": "I can’t assign tasks; the dropdown is stuck loading.",                          "expected": ["Support"]},
        {"text": "Latency spiked to 1 s per request at 09:00 UTC. Is there an outage?",           "expected": ["Support"]},
        {"text": "Export to CSV includes hidden columns we don’t need—how to fix?",               "expected": ["Support"]},
        {"text": "Our SLA says 99.9 % but we had 2 h downtime last month; request RCA.",          "expected": ["Support"]},

        # -------------------- ENQUIRY (12) --------------------
        {"text": "What is the maximum attachment size for the free tier?",                        "expected": ["Enquiry"]},
        {"text": "Do you support two‑factor authentication with hardware tokens?",                "expected": ["Enquiry"]},
        {"text": "How long do you keep logs in the basic plan?",                                   "expected": ["Enquiry"]},
        {"text": "Is your roadmap public? We’re curious about upcoming AI features.",             "expected": ["Enquiry"]},
        {"text": "Can I pay annually instead of monthly on the starter plan?",                    "expected": ["Enquiry"]},
        {"text": "Do you offer a sandbox environment for testing webhooks?",                      "expected": ["Enquiry"]},
        {"text": "Where are your data‑centres located? Need EU only.",                            "expected": ["Enquiry"]},
        {"text": "Is there an offline mode for the mobile app?",                                  "expected": ["Enquiry"]},
        {"text": "What compliance certifications do you have besides SOC 2?",                     "expected": ["Enquiry"]},
        {"text": "Are there keyboard shortcuts for quick navigation?",                            "expected": ["Enquiry"]},
        {"text": "Can multiple workspaces share the same license seat?",                          "expected": ["Enquiry"]},
        {"text": "How do I nominate someone else as the billing owner?",                          "expected": ["Enquiry"]},

        # -------------------- SALES (12) --------------------
        {"text": "We’re planning to add 150 users—could you quote enterprise pricing?",           "expected": ["Sales"]},
        {"text": "Does the premium tier include on‑site training, and what’s the cost?",          "expected": ["Sales"]},
        {"text": "Looking to upgrade to Pro; can we get a three‑year contract discount?",         "expected": ["Sales"]},
        {"text": "Bundle pricing for CRM + Helpdesk if we migrate both stacks?",                  "expected": ["Sales"]},
        {"text": "We saw your booth at SaaStr. Can we schedule a demo for next week?",            "expected": ["Sales"]},
        {"text": "Need a formal quote for 300 seats, paid via purchase order.",                   "expected": ["Sales"]},
        {"text": "Do you offer reseller margins for APAC partners?",                              "expected": ["Sales"]},
        {"text": "What’s the lead time to roll out an on‑prem appliance edition?",                "expected": ["Sales"]},
        {"text": "Any incentives if we migrate before Q4 ends?",                                  "expected": ["Sales"]},
        {"text": "Can we co‑brand the white‑label version? Please share pricing.",                "expected": ["Sales"]},
        {"text": "Requesting a comparison sheet versus Salesforce for our CFO.",                  "expected": ["Sales"]},
        {"text": "Our legal team needs the MSA to finalise a $200 k deal. Who can send it?",      "expected": ["Sales"]},

        # -------------------- OTHER (12) --------------------
        {"text": "Loved your keynote at TechCrunch—great vision for the future!",                 "expected": ["Other"]},
        {"text": "Congrats on the Series B funding round—well deserved!",                         "expected": ["Other"]},
        {"text": "Your latest blog post on prompt engineering was 🔥🔥.",                          "expected": ["Other"]},
        {"text": "Just ran a quick benchmark—your API is twice as fast as before. Nice work!",    "expected": ["Other"]},
        {"text": "Met your CEO at DevOps Days; awesome chat about open source.",                  "expected": ["Other"]},
        {"text": "Sharing a case study we wrote featuring your platform—feel free to repost.",    "expected": ["Other"]},
        {"text": "Spotted a typo on your pricing page (‘definately’) 😅.",                        "expected": ["Other"]},
        {"text": "Can I use a screenshot of your dashboard in my conference slides?",             "expected": ["Other"]},
        {"text": "Our community just hit 10 k members—thanks for supporting us early on!",        "expected": ["Other"]},
        {"text": "Listened to your podcast episode on remote culture—super insightful.",          "expected": ["Other"]},
        {"text": "FYI: your docs site loads slowly on Safari mobile. Not urgent.",                "expected": ["Other"]},
        {"text": "Happy holidays to the whole team—keep building great stuff!",                   "expected": ["Other"]},
    ]
    sample_texts = [
        "I love programming in Python!",
        "Python is great for data science.",
        "I enjoy hiking and outdoor activities.",
        "Hiking in the mountains is refreshing.",
        "Data science involves statistics and machine learning.",
        "Machine learning is a subset of AI."
    ]

    # Example usage
    svc = TextClusteringService()
    texts        = [c["text"]      for c in test_cases]
    true_labels  = [c["expected"][0] for c in test_cases]
    print('------ Clustering with auto-k search ---')
    print('-'*80)
    print(type(texts))
    print('Texts:', texts)

    svc = TextClusteringService()
    out = svc.cluster(
        texts=sample_texts,
        embedding_model_name="all-MiniLM-L6-v2",
        cluster_algo="kmeans",
        n_clusters=2,
        dim_reduction=None
        # true_labels=true_labels,
    )
    print(out)

    # true_labels=true_labels, # ignored for DBSCAN
                     # omit if you don’t have ground truth
    
    # print(json.dumps(out, indent=2))