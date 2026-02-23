# 🧠 inference – Serveur d’inférence ML multi-providers Python performant

> Pipeline RAG ultime 2025 : Query Expansion, Retrieval Hybride Triple, Reranking Multi-Étages, Compression Contextuelle, Génération, Évaluation.

## 🚀 Objectif

Le projet **inference** est un serveur d'inférence haute performance conçu pour orchestrer des pipelines de **Retrieval-Augmented Generation (RAG)** sophistiqués. Il permet de passer du prototypage à la production en intégrant les meilleures pratiques de l'état de l'art (SOTA 2025).

Points forts :
- **Abstraction unifiée** pour OpenAI, Anthropic, Ollama et modèles locaux.
- **Pipeline modulaire** en 5 phases : Embedding, Retrieval, Reranking, Compression, Generation.
- **Conformité stricte** : PEP 621, PEP 484 (Mypy strict), PEP 257 (Google style), PEP 8 (Ruff).

---

## 🏗️ Architecture (5 phases)

`Query → [01] Embedding → [02] Retrieval → [03] Reranking → [04] Compression → [05] Generation → Answer`

### **Phase 01 - Traitement des requêtes** ✅
*   Décomposition multi-sauts automatique (`QueryDecomposer`).
*   Routage adaptatif (simple/standard/complexe) via `QueryRouter`.
*   Expansion de requête (HyDE, CoT, Multi-query) avec `QueryExpansionModule`.

### **Phase 02 - Récupération hybride** ✅
*   Récupération itérative multi-sauts (jusqu'à 3 sauts) via `IterativeRetriever`.
*   Filtrage intelligent des métadonnées (`MetadataFilter`).
*   Approche **Triple Hybride** : Dense (BGE-M3/OpenAI) + Sparse (BM25) + Fusion RRF.

### **Phase 03 - Réorganisation (Reranking)** ✅
*   LLM Reranking style RankGPT (`LLMReranker`).
*   Cross-Encoder haute précision (BGE-Reranker-v2-M3).
*   Réévaluation de la diversité avec l'algorithme MMR.

### **Phase 04 - Compression contextuelle** ✅
*   Compression extractive intelligente via `LLMLingua`.
*   Optimisation drastique de la fenêtre de contexte (-47% tokens en moyenne).

### **Phase 05 - Génération avancée** ✅
*   Raffinement itératif des réponses avec autocorrection.
*   Sortie structurée (JSON Schema) garantie.
*   Détection native des hallucinations (NLI).

---

## ⚙️ Stack technique

- **Langage** : Python 3.9+ (Pin 3.12 recommandé)
- **Frameworks** : FastAPI, LangChain, DSPy
- **Embeddings & LLM** : OpenAI, Anthropic, Hugging Face (Sentence-Transformers)
- **Vector Stores** : ChromaDB, Qdrant, Faiss
- **Qualité** : Ruff (format & lint), Mypy (strict), Pytest (95%+ couverture)

---

## 📦 Installation

Le projet utilise `rye` ou `pip` standard.

```bash
# 1. Cloner le dépôt
git clone https://github.com/dagornc/inference.git
cd inference

# 2. Installer les dépendances
pip install -r requirements.txt
# Ou via rye
rye sync --all-features
```

---

## ▶️ Utilisation rapide

```python
from inference_project.steps import (
    EmbeddingStep, 
    RetrievalStep, 
    RerankingStep, 
    GenerationStep
)

# Initialisation des étapes
query = \"Explique-moi le fonctionnement d'un pipeline RAG hybride.\"
emb_step = EmbeddingStep()
ret_step = RetrievalStep()

# Exécution du pipeline
emb_result = emb_step.execute(query)
ret_result = ret_step.execute(
    query_embeddings=emb_result[\"embeddings\"],
    sub_queries=emb_result.get(\"sub_queries\", [query])
)

print(f\"Documents trouvés : {len(ret_result['documents'])}\")
```

---

## 🧪 Tests & Qualité

```bash
# Formater et vérifier le code
ruff format src/
ruff check src/ --fix

# Lancer les tests avec couverture
pytest tests/ --cov=src/inference_project
```

---

## 🗺️ Roadmap

- [ ] Support des modèles d'interaction tardive (ColBERT/RAGatouille).
- [ ] Couche de cache distribuée avec Redis.
- [ ] Optimisation automatique des prompts via DSPy.
- [ ] Support des embeddings clairsemés SPLADE.

---

## 📄 Licence

Ce projet est sous licence **MIT**.
