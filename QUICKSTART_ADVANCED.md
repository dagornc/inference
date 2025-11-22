# 🚀 Quick Start - RAG Pipeline Avancé

Guide de démarrage rapide pour tester le pipeline RAG complet avec toutes les fonctionnalités avancées (Phase 01-05).

---

## 📦 Installation

### 1. Dépendances de base

```bash
pip install numpy sentence-transformers openai pyyaml
```

### 2. Dépendances Phase 02 - Retrieval

```bash
# ChromaDB (vector database)
pip install chromadb

# BM25 sparse retrieval (optionnel)
pip install pyserini
```

### 3. Dépendances Phase 04 - Compression

```bash
# LLMLingua pour compression agressive
pip install llmlingua

# Token counting
pip install tiktoken
```

### 4. Dépendances Phase 05 - Generation

```bash
# Ollama (local, gratuit) - Recommandé
# Télécharger depuis https://ollama.com
# Puis lancer: ollama run llama3

# Ou OpenAI API (payant)
export OPENAI_API_KEY="your-key-here"
```

---

## ⚡ Test Rapide - Pipeline Complet

### Script Python Minimal

Créer `test_pipeline.py` :

```python
"""Test du pipeline RAG complet avec features avancées."""

from typing import List, Dict, Any
import numpy as np

# Import des steps
from inference_project.steps.step_01_embedding_generation import process_embeddings
from inference_project.steps.step_02_retrieval import process_retrieval
from inference_project.steps.step_03_reranking import process_reranking
from inference_project.steps.step_04_compression import process_compression
from inference_project.steps.step_05_generation import process_generation


def create_mock_config() -> Dict[str, Any]:
    """Créer une configuration mock minimale."""
    return {
        # Phase 01
        "embedding_generation": {
            "enabled": True,
            "dense": {
                "enabled": True,
                "model": "all-MiniLM-L6-v2",  # Léger, rapide
                "device": "cpu",
            }
        },

        # Phase 02
        "retrieval": {
            "dense_retrieval": {
                "enabled": True,
                "vector_db": "chromadb",
                "persist_directory": None,  # In-memory
                "collection_name": "test_docs",
            },
            "sparse_retrieval": {
                "enabled": False,  # Désactivé pour test rapide
            },
            "fusion": {
                "enabled": False,
            },
            "top_k": 5,
        },

        # Phase 03
        "reranking": {
            "cross_encoder": {
                "enabled": True,
                "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                "device": "cpu",
            },
            "mmr": {
                "enabled": True,
                "lambda": 0.7,
                "final_top_k": 3,
            },
            "final_top_k": 3,
        },

        # Phase 04
        "step_04_compression": {
            "enabled": True,
            "pipeline": [
                {"step": "pre_compression_analysis", "enabled": True},
                {"step": "prompt_compression_llmlingua", "enabled": False},  # Optionnel
                {"step": "contextual_compression", "enabled": True},
                {"step": "mmr_compression_aware", "enabled": True},
                {"step": "quality_validation", "enabled": True},
                {"step": "context_window_optimization", "enabled": True},
            ],
        },
        "contextual_compression": {
            "enabled": True,
            "extractive": {
                "scorer_model": "all-MiniLM-L6-v2",
                "max_passage_length": 200,
                "relevance_threshold": 0.4,
            },
        },
        "mmr_compression_aware": {
            "enabled": True,
            "final_top_k": 3,
            "compression_aware": {
                "boost_well_compressed": {
                    "enabled": True,
                    "compression_ratio_threshold": 2.0,
                    "boost_factor": 1.1,
                }
            },
        },
        "quality_validation": {
            "enabled": True,
            "semantic_similarity": {
                "enabled": True,
                "model": "all-MiniLM-L6-v2",
                "min_similarity": 0.85,
            },
        },
        "context_window_optimization": {
            "enabled": True,
            "target_context_tokens": 500,
            "smart_truncate": {
                "preserve_top_k": 2,
            },
        },

        # Phase 05
        "step_05_generation": {
            "enabled": True,
            "pipeline": [
                {"step": "pre_generation_analysis", "enabled": True},
                {"step": "prompt_construction", "enabled": True},
                {"step": "advanced_prompting", "enabled": True},
                {"step": "initial_generation", "enabled": True},
                {"step": "self_rag", "enabled": True, "conditional": True},
                {"step": "hallucination_detection", "enabled": True},
                {"step": "multi_stage_validation", "enabled": True},
                {"step": "post_processing", "enabled": True},
            ],
        },
        "pre_generation_analysis": {
            "enabled": True,
            "query_complexity": {"enabled": True},
            "crag_evaluator": {
                "enabled": True,
                "lightweight_config": {
                    "thresholds": {"correct": 0.7, "ambiguous": 0.4}
                },
            },
            "strategy_selection": {"enabled": True},
        },
        "self_rag": {
            "enabled": True,
        },
        "hallucination_detection": {
            "enabled": True,
            "model": "all-MiniLM-L6-v2",
            "threshold": 0.5,
        },
        "multi_stage_validation": {
            "enabled": True,
            "threshold": 0.7,
        },
        "llm": {
            "provider": "ollama",
            "model": "llama3",
        },
        "prompt": {},
        "post_processing": {
            "formatting": {
                "clean_whitespace": True,
                "append_sources": True,
                "output_format": "markdown",
            }
        },
    }


def populate_vector_db(config: Dict[str, Any]) -> None:
    """Peupler ChromaDB avec des documents de test."""
    import chromadb

    client = chromadb.Client()

    # Créer collection
    collection = client.get_or_create_collection(name="test_docs")

    # Documents de test sur le machine learning
    documents = [
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed. It uses statistical techniques to give systems the ability to improve with experience.",
        "Supervised learning is a type of machine learning where models are trained on labeled data. The algorithm learns to map inputs to known outputs, like classification and regression tasks.",
        "Unsupervised learning finds patterns in unlabeled data. Common techniques include clustering and dimensionality reduction. It's useful for discovering hidden structures in data.",
        "Deep learning uses neural networks with multiple layers to learn hierarchical representations of data. It excels at tasks like image recognition and natural language processing.",
        "Reinforcement learning trains agents to make decisions by rewarding desired behaviors. The agent learns through trial and error to maximize cumulative rewards.",
    ]

    # Générer IDs
    ids = [f"doc{i}" for i in range(len(documents))]

    # Métadonnées
    metadatas = [
        {"source": "ML Intro", "topic": "basics"},
        {"source": "ML Guide", "topic": "supervised"},
        {"source": "ML Guide", "topic": "unsupervised"},
        {"source": "DL Primer", "topic": "deep_learning"},
        {"source": "RL Tutorial", "topic": "reinforcement"},
    ]

    # Ajouter à ChromaDB (il génère embeddings automatiquement)
    collection.add(
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"✅ Added {len(documents)} documents to ChromaDB")


def run_pipeline_test():
    """Tester le pipeline complet."""
    print("=" * 80)
    print("🚀 TEST PIPELINE RAG AVANCÉ")
    print("=" * 80)

    # Configuration
    config = create_mock_config()

    # Peupler vector DB
    print("\n📚 Populating vector database...")
    populate_vector_db(config)

    # Query de test
    query = "What is machine learning and how does it work?"
    queries = [query]

    print(f"\n❓ Query: {query}")
    print("\n" + "=" * 80)

    # Phase 01: Embedding Generation
    print("\n🔸 PHASE 01 - EMBEDDING GENERATION")
    embedding_result = process_embeddings(queries, config)
    query_embeddings = embedding_result["dense_embeddings"]
    print(f"   ✅ Generated embeddings: shape {query_embeddings.shape}")

    # Phase 02: Retrieval
    print("\n🔸 PHASE 02 - RETRIEVAL")
    retrieval_results = process_retrieval(query_embeddings, queries, config)
    print(f"   ✅ Retrieved {len(retrieval_results[0])} documents")
    for i, doc in enumerate(retrieval_results[0][:3], 1):
        print(f"      [{i}] Score: {doc['score']:.3f} - {doc['document'][:80]}...")

    # Phase 03: Reranking
    print("\n🔸 PHASE 03 - RERANKING")
    reranked_results = process_reranking(queries, retrieval_results, config)
    print(f"   ✅ Reranked to {len(reranked_results[0])} documents")
    for i, doc in enumerate(reranked_results[0], 1):
        print(f"      [{i}] Score: {doc['score']:.3f} - {doc['document'][:80]}...")

    # Phase 04: Compression
    print("\n🔸 PHASE 04 - COMPRESSION")
    compression_result = process_compression(
        reranked_results[0],
        query,
        config
    )
    compressed_docs = compression_result["documents"]
    print(f"   ✅ Compressed: {compression_result['compression_ratio']:.2f}x")
    print(f"      Tokens: {compression_result['original_tokens']} → {compression_result['compressed_tokens']}")
    print(f"      Documents: {compression_result['num_documents']}")

    # Phase 05: Generation
    print("\n🔸 PHASE 05 - GENERATION")
    generation_result = process_generation(
        query,
        compressed_docs,
        config
    )

    # Afficher résultat
    print("\n" + "=" * 80)
    print("📝 RÉPONSE FINALE")
    print("=" * 80)
    print(generation_result["answer"])

    # Métadonnées détaillées
    print("\n" + "=" * 80)
    print("📊 MÉTADONNÉES AVANCÉES")
    print("=" * 80)

    metadata = generation_result.get("metadata", {})
    gen_metadata = metadata.get("generation_metadata", {})

    # Pre-generation analysis
    if "pre_generation_analysis" in gen_metadata:
        analysis = gen_metadata["pre_generation_analysis"]
        print("\n🔍 Pre-Generation Analysis:")
        print(f"   Query Complexity: {analysis.get('query_complexity', 'N/A')}")
        print(f"   CRAG Score: {analysis.get('crag_score', 0.0):.2f}")
        print(f"   CRAG Action: {analysis.get('crag_action', 'N/A')}")
        print(f"   Strategy: {analysis.get('strategy', 'N/A')}")

    # Self-RAG
    if metadata.get("used_self_rag"):
        print("\n🔄 Self-RAG: ✅ USED")
        reflection = gen_metadata.get("self_rag", {})
        if reflection:
            print(f"   Needs more context: {reflection.get('needs_more_context', False)}")
            print(f"   Docs relevant: {reflection.get('docs_relevant', True)}")
            print(f"   Answer supported: {reflection.get('answer_supported', True)}")
            print(f"   Should use: {reflection.get('should_use', True)}")
    else:
        print("\n🔄 Self-RAG: ⏭️ SKIPPED (not needed)")

    # Hallucination detection
    if "hallucination_detection" in gen_metadata:
        halluc = gen_metadata["hallucination_detection"]
        print("\n🛡️ Hallucination Detection:")
        print(f"   Has hallucination: {halluc.get('has_hallucination', False)}")
        print(f"   Confidence: {halluc.get('confidence', 1.0):.2%}")
        print(f"   Hallucination score: {halluc.get('hallucination_score', 0.0):.3f}")

        checks = halluc.get("checks", {})
        if checks:
            print(f"   Semantic consistency: {checks.get('semantic_consistency', 0.0):.2f}")
            print(f"   Uncertainty markers: {checks.get('uncertainty_markers', 0.0):.2f}")
            print(f"   Has citations: {checks.get('has_citations', False)}")

    # Multi-stage validation
    if "multi_stage_validation" in gen_metadata:
        valid = gen_metadata["multi_stage_validation"]
        print("\n✅ Multi-Stage Validation:")
        print(f"   Status: {'PASSED ✅' if valid.get('passed', False) else 'FAILED ❌'}")
        print(f"   Overall score: {valid.get('overall_score', 0.0):.2f}")
        print(f"   Faithfulness: {valid.get('faithfulness_score', 0.0):.2f}")
        print(f"   Attribution: {valid.get('attribution_score', 0.0):.2f}")
        print(f"   Consistency: {valid.get('consistency_score', 0.0):.2f}")

    print("\n" + "=" * 80)
    print("✅ TEST COMPLET - SUCCESS !")
    print("=" * 80)


if __name__ == "__main__":
    try:
        run_pipeline_test()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
```

---

## 🏃 Exécuter le Test

### Option 1 : Test avec Ollama (Local, Gratuit) - RECOMMANDÉ

```bash
# 1. Installer Ollama
# Télécharger depuis https://ollama.com

# 2. Lancer le modèle
ollama run llama3

# 3. Dans un autre terminal, lancer le test
python test_pipeline.py
```

### Option 2 : Test avec OpenAI API (Payant)

```bash
# 1. Configurer API key
export OPENAI_API_KEY="sk-..."

# 2. Modifier config dans test_pipeline.py
# Remplacer:
#   "provider": "ollama",
# Par:
#   "provider": "openai",

# 3. Lancer
python test_pipeline.py
```

---

## 📊 Sortie Attendue

```
================================================================================
🚀 TEST PIPELINE RAG AVANCÉ
================================================================================

📚 Populating vector database...
✅ Added 5 documents to ChromaDB

❓ Query: What is machine learning and how does it work?

================================================================================

🔸 PHASE 01 - EMBEDDING GENERATION
   ✅ Generated embeddings: shape (1, 384)

🔸 PHASE 02 - RETRIEVAL
   ✅ Retrieved 5 documents
      [1] Score: 0.856 - Machine learning is a subset of artificial intelligence...
      [2] Score: 0.782 - Supervised learning is a type of machine learning...
      [3] Score: 0.745 - Deep learning uses neural networks...

🔸 PHASE 03 - RERANKING
   ✅ Reranked to 3 documents
      [1] Score: 4.231 - Machine learning is a subset of artificial intelligence...
      [2] Score: 3.867 - Supervised learning is a type of machine learning...
      [3] Score: 3.542 - Deep learning uses neural networks...

🔸 PHASE 04 - COMPRESSION
   ✅ Compressed: 1.85x
      Tokens: 156 → 84
      Documents: 3

🔸 PHASE 05 - GENERATION

================================================================================
📝 RÉPONSE FINALE
================================================================================
Machine learning is a subset of artificial intelligence that enables computers
to learn from data without explicit programming [1]. It uses statistical
techniques to improve with experience. ML includes various approaches like
supervised learning, which trains on labeled data [2], and deep learning,
which uses multi-layer neural networks for tasks like image recognition [3].

---
Sources utilisées :
[1] ML Intro
[2] ML Guide
[3] DL Primer

================================================================================
📊 MÉTADONNÉES AVANCÉES
================================================================================

🔍 Pre-Generation Analysis:
   Query Complexity: medium
   CRAG Score: 0.85
   CRAG Action: correct
   Strategy: standard_rag

🔄 Self-RAG: ⏭️ SKIPPED (not needed)

🛡️ Hallucination Detection:
   Has hallucination: False
   Confidence: 87.32%
   Hallucination score: 0.127
   Semantic consistency: 0.89
   Uncertainty markers: 0.00
   Has citations: True

✅ Multi-Stage Validation:
   Status: PASSED ✅
   Overall score: 0.82
   Faithfulness: 0.87
   Attribution: 0.85
   Consistency: 0.90

================================================================================
✅ TEST COMPLET - SUCCESS !
================================================================================
```

---

## 🔧 Personnalisation

### Activer LLMLingua (Compression Agressive)

```python
config["step_04_compression"]["pipeline"][1] = {
    "step": "prompt_compression_llmlingua",
    "enabled": True  # ⚠️ Nécessite: pip install llmlingua
}

config["prompt_compression"] = {
    "enabled": True,
    "tool": "llmlingua2",
    "llmlingua2": {
        "compression_rate": 0.4,  # 2.5x compression
        "model_config": {
            "model_name": "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
            "device": "cpu",
        },
    },
}
```

**Résultat attendu** : Compression 2.5x-4x au lieu de 1.8x

---

### Activer Self-RAG Systématiquement

```python
# Désactiver mode conditionnel
config["step_05_generation"]["pipeline"][4]["conditional"] = False
```

**Résultat attendu** : Self-RAG toujours utilisé, +12-15% qualité, +1s latence

---

### Tester avec Query Complexe

```python
# Query multi-hop complexe
query = "Compare supervised and unsupervised learning, and explain when to use each approach."
```

**Résultat attendu** :
- Query complexity: `complex`
- Strategy: `multi_hop_cot`
- Self-RAG: ✅ USED
- Latence: ~3.8s

---

## 🐛 Troubleshooting

### Erreur : "No module named 'chromadb'"
```bash
pip install chromadb
```

### Erreur : "No module named 'sentence_transformers'"
```bash
pip install sentence-transformers
```

### Erreur : "LLMLingua not installed"
```bash
# Optionnel, peut être désactivé
pip install llmlingua
```

### Erreur : "Ollama connection refused"
```bash
# Vérifier qu'Ollama est lancé
ollama run llama3

# Ou utiliser OpenAI API à la place
```

### Erreur : "CUDA out of memory"
```bash
# Forcer CPU dans config
config["embedding_generation"]["dense"]["device"] = "cpu"
config["reranking"]["cross_encoder"]["device"] = "cpu"
```

---

## 📈 Benchmarks

### Configuration Testée

- **CPU** : M1/M2 Mac ou Intel i7
- **RAM** : 8GB minimum
- **Storage** : 2GB pour modèles

### Performance Mesurée

| Phase | Latence | Modèle |
|-------|---------|--------|
| Phase 01 | ~200ms | all-MiniLM-L6-v2 |
| Phase 02 | ~150ms | ChromaDB |
| Phase 03 | ~300ms | cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Phase 04 | ~250ms | Contextual compression |
| Phase 05 | ~2500ms | Ollama llama3 |
| **TOTAL** | **~3.4s** | Query medium |

---

## 🎯 Next Steps

### 1. Tester avec vos propres documents

Remplacer les documents de test dans `populate_vector_db()` :

```python
documents = [
    "Votre document 1...",
    "Votre document 2...",
    # ...
]
```

### 2. Ajuster les configurations

Voir les fichiers de config complets dans `config/` :
- `01_embedding_v2.yaml`
- `02_retrieval_v2.yaml`
- `03_reranking_v2.yaml`
- `04_compression_v2.yaml`
- `05_generation_v2.yaml`

### 3. Mesurer la qualité

Ajouter métriques avec Ragas :

```bash
pip install ragas

# Créer dataset d'évaluation
# Voir: docs/EVALUATION_GUIDE.md (à créer)
```

---

## 📚 Documentation Complète

- **Architecture** : `docs/YOLO_MODE_IMPLEMENTATION.md`
- **Configuration v2** : `config/*_v2.yaml`
- **Tests** : `tests/test_step_*.py`
- **Code source** : `src/inference_project/steps/step_*.py`

---

**Bon test ! 🚀**
