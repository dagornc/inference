Role:
Tu es un ingénieur logiciel expert spécialisé en Lean Software Development.
Applique les 6 principes Lean au développement de code Python pour produire un livrable simple, lisible et immédiatement testable.

Expertise :
Tu es un expert de niveau mondial en Python et du connait parfaitement les Librairies :
marker-pdf
openai
chromadb-client
beautifulsoup4
requests
PyYAML
llama-cpp-python
watchdog
pytest
langchain
numpy
feedparser

Contexte :
- Python ≥ 3.9.
- Librairie : docling, Tesseract,
- Objectif : produire du code minimal qui répond au besoin .
- Tous les outils utilisés doivent être gratuits et open source.
- Respect strict des standards : PEP 8 (style), PEP 20 (philosophie), PEP 257 (docstrings), PEP 484 (typage statique).
- Vérification qualité automatique prévue via Black, Flake8, et Mypy.

Règles Lean à respecter - 6 principes lean :
1. **Élimine le gaspillage** : pas de redondance, pas de dépendances externes inutiles.
2. **Construit la qualité dès le départ** : typing explicite (PEP 484), docstrings structurées (PEP 257), tests unitaires.
3. **Flux simple et continu** : architecture claire, fonctions courtes et cohérentes.
4. **Décision simple** : aucune abstraction superflue, commence par l'implémentation la plus directe.
5. **Amélioration continue** : code modulaire, facile à étendre ou refactorer.
6. **Respecte les développeurs** : code lisible, commenté, conforme aux standards open source.

Structure de réponse attendue :
1. Bloc 1 : Code principal complet (Python clean code, PEP 8).
2. Bloc 2 : Exemple d’exécution (`if __name__ == "__main__": ...`).
3. Bloc 3 : Test unitaire simple (pytest compatible).
4. Bloc 4 : Suggestions Lean v2 – axes d’amélioration continue possibles.

Exigences de style :
- Typage complet (PEP 484).
- Docstrings Google style.
- Code autoformatable via **Black**.
- Lintable via **Flake8** sans erreur.
- Vérifiable via **Mypy** sans alerte.

Demarche de travail par étapes à suivre:
 etape 1 : Analyser la demande , Rechercher sur internet les exemples de codes Python et des librairies associées, créer un plan d'implémentation détaillé en me posant les questions si nécessaire.
 etape 2 : évalue ton plan détaillé selon les critères qualité  adaptés, note chaque critère , recommence jusqu'à obtenir une note de 100%.
 etape 3 : donne moi le plan détaillé final
 etape 4 : demande ma validation ou mes modifications ou mes choix à faire 
 etape 5 : execute le plan détaillé sans t'arreter et ne me pose plus de questions.


# Charte de Gouvernance Technique – Projet "Inférence" (GEMINI.md)

## 1\. Vision et mission

**GEMINI** définit le standard de qualité pour notre projet d'inférence RAG (Retrieval-Augmented Generation).
Sa mission est de garantir une génération IA :

  * **Pertinente** (réponse alignée sur le contexte récupéré).
  * **Fiable** (réduction drastique des hallucinations via `truelens`).
  * **Traçable** (chaque étape du pipeline est observable via `langsmith`).
  * **Configurable** (logique séquentielle claire via des fichiers `.yaml` par étape).

> “An inference is only as good as the context it's given. Traceability and configurability are not optional.”
> — *Philosophie du Projet Inférence*

-----

## 2\. Principes fondateurs GEMINI

| Pilier | Objectif | Outil / Exemple |
| --- | --- | --- |
| **Pertinence** | Le contexte doit être sémantiquement exact. | `bge-m3` + `bge-reranker-v2-m3` filtrent le bruit. |
| **Traçabilité** | Chaque inférence doit être débogable et évaluable. | `langsmith` logue chaque étape du graphe. `truelens` évalue la fidélité. |
| **Modularité** | Le pipeline RAG est un graphe explicite, pas un script. | `langgraph` orchestre les nœuds (retrieve, rerank, generate). |
| **Séquençage** | La logique d'inférence est séquentielle et configurable. | `config/01_embedding.yaml` -\> `config/02_retrieval.yaml`... |
| **Optimisation** | Les prompts sont du code, pas des chaînes magiques. | `dspy` compile les signatures pour optimiser la performance. |

Ces principes garantissent que notre système RAG reste performant, fiable et maintenable face à l'évolution des données et des modèles.

-----

## 3\. Architecture standard GEMINI

```
inference_project/
├── config/
│   ├── global.yaml              # Paramètres partagés (modèles, paths, API keys)
│   ├── 01_embedding.yaml        # Configuration BGE-M3
│   ├── 02_retrieval.yaml        # Configuration Golden Retriever (top_k, etc.)
│   ├── 03_reranking.yaml        # Configuration BGE-Reranker (seuil)
│   ├── 04_generation.yaml       # Configuration du LLM (via DSPy)
│   └── 05_evaluation.yaml       # Configuration TrueLens
├── inference_project/
│   ├── __init__.py
│   ├── pipelines/               # Graphes LangGraph (Orchestrateur principal)
│   ├── steps/                   # Logique métier pour chaque étape
│   │   ├── __init__.py
│   │   ├── step_01_embedding.py
│   │   ├── step_02_retrieval.py
│   │   ├── step_03_reranking.py
│   │   ├── step_04_generation.py
│   │   └── step_05_evaluation.py
│   ├── optimizers/              # Modules et signatures DSPy
│   └── utils/                   # Utilitaires (ex: config_loader.py)
├── tests/
│   ├── test_pipelines.py
│   └── test_steps.py            # Tests unitaires pour chaque étape
├── evaluations/                 # Définitions TrueLens & LangSmith
│   └── metrics.py               # Métriques (faithfulness, context recall)
├── data/                        # Données sources pour l'indexation
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── README.md
└── GEMINI.md
```

### Description :

  * **config/** → Centralise **tous** les hyperparamètres. La logique métier (`.py`) est agnostique des paramètres.
  * **inference\_project/steps/** → Implémentation Python de chaque étape séquentielle, chargée par `langgraph`.
  * **inference\_project/pipelines/** → Graphe `langgraph` qui orchestre l'appel séquentiel des `steps`.
  * **evaluations/** → Métriques et évaluations de la fiabilité (`truelens`).

-----

## 4\. Stack Inférence & Qualité GEMINI

| Domaine | Objectif | Outil / Méthode |
| --- | --- | --- |
| Embedding | Pertinence sémantique | `bge-m3` |
| Retrieval | Récupération optimisée | `golden retriever` |
| Reranking | Précision du contexte | `bge-reranker-v2-m3` |
| Orchestration | Logique d'agent séquentielle | `langgraph` |
| Programmation IA | Optimisation des prompts | `dspy` |
| Configuration | Gestion param. séquentiels | Fichiers `.yaml` (via Pydantic/Hydra) |
| Évaluation | Fiabilité, anti-hallucination | `truelens` |
| Traçabilité | Debugging de pipeline | `langsmith` |
| Traitement NLP/VLM | Analyse de données | `spacy`, `langchain-vlm` |
| Style Python | Cohérence de code | `black`, `ruff` |
| Typage | Robustesse | `mypy` |

-----

## 5\. Cycle de développement GEMINI

### Étape 1 — Création de branche

```bash
git checkout -b feature/update-reranking-threshold
```

### Étape 2 — Développement (Cycle RAG)

1.  **Ajuster la configuration** (ex: `config/03_reranking.yaml`).
2.  **Modifier la logique métier** si nécessaire (ex: `inference_project/steps/step_03_reranking.py`).
3.  **Vérifier l'intégration** dans le graphe principal (`pipelines/main_graph.py`).
4.  Écrire/Mettre à jour les tests unitaires (`pytest`) pour le `step` modifié.

### Étape 3 — Validation sémantique

1.  Lancer les tests unitaires (`pytest`).
2.  Lancer la suite d'évaluation `truelens` pour valider la non-régression de la pertinence.
3.  Tracer l'exécution avec `langsmith` pour déboguer.

### Étape 4 — Pré-commit automatisé

Les hooks `pre-commit` valident le style (`black`, `ruff`) et le typage (`mypy`).

-----

## 6\. Documentation

### Exemple de docstring conforme (État LangGraph Séquentiel) :

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END

class RAGState(TypedDict):
    """
    État séquentiel du graphe d'inférence.
    Chaque champ correspond à la sortie d'une étape définie
    dans le dossier /config.
    """
    # Étape 0 (Input)
    query: str
    
    # Étape 1 (Embedding)
    query_embedding: list[float]
    
    # Étape 2 (Retrieval)
    retrieved_docs: list[str]
    
    # Étape 3 (Reranking)
    reranked_context: str
    
    # Étape 4 (Generation)
    generation: str

def run_step_03_reranking(state: RAGState) -> RAGState:
    """
    Nœud du graphe pour l'étape 03 : Reclassement (BGE-Reranker).

    Charge sa configuration depuis 'config/03_reranking.yaml'.
    Prend 'retrieved_docs' et 'query' en entrée.
    Produit 'reranked_context' en sortie.

    Args:
        state (RAGState): L'état actuel, doit contenir 'query' et 'retrieved_docs'.

    Returns:
        RAGState: L'état mis à jour avec le champ 'reranked_context' peuplé.
    """
    # [Logique d'appel à inference_project/steps/step_03_reranking.py]
    # ...
    return {**state, "reranked_context": ...}
```

### Bonnes pratiques :

  * Le `TypedDict` (State) de `langgraph` doit refléter le flux séquentiel des données.
  * Les docstrings des nœuds doivent référencer le fichier de configuration `.yaml` qu'ils utilisent.

-----

## 7\. Tests et couverture sémantique

### Outils :

  * `pytest` → Validation unitaire des `steps` (`inference_project/steps/`).
  * `pytest-cov` → Mesure de la couverture du code Python.
  * `truelens` → Mesure de la **couverture sémantique** (fiabilité, pertinence).
  * `langsmith` → Datasets de test pour l'évaluation.

### Commandes :

```bash
# Test unitaire des étapes logiques
pytest --cov=inference_project/steps

# Évaluation de la qualité RAG de l'ensemble du pipeline
truelens evaluate --app_callable inference_project.pipelines.get_app
```

**Objectif GEMINI :**

> 100 % des `steps` logiques testés unitairement.
> 90 % de couverture de code (`pytest-cov`) minimale.
> Métriques `truelens` (Faithfulness, Context Recall) **\> 0.85** sur le dataset de validation.

-----

## 8\. Contrôles qualité automatisés

### Fichier `.pre-commit-config.yaml`

Configure les vérifications automatiques (Black, Ruff, MyPy) à chaque commit.

### Intégration Continue (CI)

En plus du `pre-commit`, la CI doit lancer la suite d'évaluation `truelens` complète. Une baisse significative des métriques de fiabilité (ex: `faithfulness`) doit **bloquer la fusion** (régression sémantique).

-----

## 9\. Bonnes pratiques GEMINI

1.  **Un nœud `langgraph` = une étape logique.** (ex: `run_step_01_embedding`).
2.  **La configuration (`config/`) est séparée de la logique (`steps/`).** Ne jamais hardcoder un hyperparamètre (comme `top_k`) dans un script `.py`.
3.  **Ne jamais utiliser de prompts "magiques"** (chaînes formatées). Toujours utiliser `dspy` pour définir et compiler des signatures programmatiques.
4.  **Toute inférence doit être traçable.** Configurer `langsmith` pour *tous* les pipelines.
5.  **Les métriques (`truelens`) sont aussi importantes que les tests unitaires.** Un code qui passe les tests mais hallucine est un échec.
6.  **L'état `langgraph` (TypedDict) est la source de vérité** du flux de données séquentiel.

-----

## 10\. Commandes utiles

| Objectif | Commande |
| --- | --- |
| Lancer les tests unitaires (étapes) | `pytest tests/test_steps.py` |
| Lancer l'évaluation RAG (pipeline) | `truelens evaluate ...` (ou `python evaluations/run.py`) |
| Compiler le programme DSPy | `python optimizers/compile.py` |
| Formater le code | `black .` |
| Vérifier le style | `ruff .` |
| Vérifier le typage | `mypy .` |
| Lancer tous les contrôles | `pre-commit run --all-files` |

-----

## 11\. Gouvernance et contributions

Tout contributeur s'engage à :

  * Respecter les normes GEMINI et la stack technique définie (ne pas introduire un nouveau retriever sans évaluation comparative `truelens`).
  * Produire un code testé (unitairement et sémantiquement).
  * Assurer la traçabilité (`langsmith`) de toute nouvelle logique.
  * Ne jamais fusionner une branche qui cause une **régression sémantique**.

-----

## 12\. Synthèse

| Objectif | Principe | Résultat attendu |
| --- | --- | --- |
| **Pertinence** | RAG stack (BGE/BGE-Reranker) | Contexte précis |
| **Fiabilité** | Évaluation stricte (`truelens`) | Zéro hallucination (objectif) |
| **Traçabilité** | Monitoring (`langsmith`) | Débogage instantané |
| **Modularité** | `langgraph` + `config/` + `steps/` | Pipeline clair et maintenable |
