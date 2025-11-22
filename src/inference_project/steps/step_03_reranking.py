"""Module pour le reranking des documents (Phase 03).

Ce module implémente le reranking des documents récupérés pour améliorer
la précision. Il utilise principalement des cross-encoders qui calculent
un score de pertinence pour chaque paire (query, document).

Fonctions:
    CrossEncoderReranker: Classe pour reranking avec cross-encoder.
    MMR: Classe pour Maximal Marginal Relevance (diversité).
    process_reranking: Point d'entrée principal orchestrant le reranking.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class CrossEncoderReranker:
    """Reranker utilisant un cross-encoder.

    Un cross-encoder prend en entrée une paire (query, document) et prédit
    un score de pertinence. C'est plus précis qu'un bi-encoder (embeddings
    séparés) mais plus lent car il faut traiter chaque paire.

    Modèles recommandés :
    - BGE-reranker-v2-m3 (multilingue)
    - cross-encoder/ms-marco-MiniLM-L-6-v2
    - BAAI/bge-reranker-large

    Attributes:
        config: Configuration du reranking.
        model: Modèle cross-encoder.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le cross-encoder reranker.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("cross_encoder", {})
        self.model: Optional[Any] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialise le modèle cross-encoder."""
        if not self.config.get("enabled", True):
            return

        try:
            from sentence_transformers import CrossEncoder

            model_name = self.config.get(
                "model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            device = self.config.get("device", "cpu")
            max_length = self.config.get("max_length", 512)

            self.model = CrossEncoder(model_name, max_length=max_length, device=device)

        except ImportError:
            raise ImportError(
                "sentence-transformers not installed for CrossEncoder. "
                "Run: pip install sentence-transformers"
            )

    def rerank(
        self,
        queries: List[str],
        results: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Rerank les documents avec le cross-encoder.

        Args:
            queries: Liste des queries textuelles.
            results: Résultats du retrieval pour chaque query.
            top_k: Nombre de documents à garder après reranking (None = tous).

        Returns:
            Résultats reranked pour chaque query.
        """
        if not self.config.get("enabled", True) or self.model is None:
            return results

        reranked_results = []

        for query, query_results in zip(queries, results):
            if not query_results:
                reranked_results.append([])
                continue

            # Préparer paires (query, document)
            pairs = [[query, doc["document"]] for doc in query_results]

            # Calculer scores avec cross-encoder
            scores = self.model.predict(pairs)

            # Mettre à jour scores dans résultats
            for doc, score in zip(query_results, scores):
                doc["score"] = float(score)
                doc["source"] = "reranked"

            # Trier par score décroissant
            query_results_sorted = sorted(
                query_results, key=lambda x: x["score"], reverse=True
            )

            # Limiter au top_k si spécifié
            if top_k is not None:
                query_results_sorted = query_results_sorted[:top_k]

            reranked_results.append(query_results_sorted)

        return reranked_results


class MMR:
    """Maximal Marginal Relevance pour diversifier les résultats.

    MMR sélectionne des documents à la fois pertinents et divers.
    Il équilibre pertinence et diversité via un paramètre lambda.

    Score MMR = lambda * relevance - (1 - lambda) * similarité_max

    Attributes:
        config: Configuration MMR.
        lambda_param: Paramètre d'équilibre (0=diversité, 1=pertinence).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise MMR.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("mmr", {})
        self.lambda_param = self.config.get("lambda", 0.7)

    def apply(
        self,
        results: List[List[Dict[str, Any]]],
        doc_embeddings: Optional[List[np.ndarray]] = None,
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Applique MMR pour diversifier les résultats.

        Args:
            results: Résultats reranked pour chaque query.
            doc_embeddings: Embeddings des documents (pour calcul similarité).
                           Si None, MMR basé uniquement sur ordre de relevance.
            top_k: Nombre de documents finaux à retourner.

        Returns:
            Résultats avec diversité maximale.
        """
        if not self.config.get("enabled", True):
            return results

        final_top_k = top_k or self.config.get("final_top_k", 10)
        mmr_results = []

        for i, query_results in enumerate(results):
            if len(query_results) <= final_top_k:
                # Déjà moins que top_k, pas besoin de MMR
                mmr_results.append(query_results)
                continue

            # MMR avec embeddings
            if doc_embeddings is not None and i < len(doc_embeddings):
                selected = self._mmr_with_embeddings(
                    query_results,
                    doc_embeddings[i],
                    final_top_k,
                )
            else:
                # MMR sans embeddings (diversité basique)
                selected = query_results[:final_top_k]

            mmr_results.append(selected)

        return mmr_results

    def _mmr_with_embeddings(
        self,
        documents: List[Dict[str, Any]],
        embeddings: np.ndarray,
        k: int,
    ) -> List[Dict[str, Any]]:
        """MMR avec embeddings pour calcul de similarité précis.

        Args:
            documents: Liste des documents à diversifier.
            embeddings: Embeddings des documents (n_docs x dim).
            k: Nombre de documents à sélectionner.

        Returns:
            k documents sélectionnés avec MMR.
        """
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # Sélectionner le premier document (meilleur score)
        selected_indices.append(0)
        remaining_indices.remove(0)

        # Sélectionner k-1 documents restants
        for _ in range(k - 1):
            if not remaining_indices:
                break

            max_mmr_score = -float("inf")
            max_mmr_idx = None

            for idx in remaining_indices:
                # Relevance score
                relevance = documents[idx]["score"]

                # Similarité maximale avec documents déjà sélectionnés
                max_sim = max(
                    self._cosine_similarity(embeddings[idx], embeddings[selected_idx])
                    for selected_idx in selected_indices
                )

                # Score MMR
                mmr_score = (
                    self.lambda_param * relevance - (1 - self.lambda_param) * max_sim
                )

                if mmr_score > max_mmr_score:
                    max_mmr_score = mmr_score
                    max_mmr_idx = idx

            if max_mmr_idx is not None:
                selected_indices.append(max_mmr_idx)
                remaining_indices.remove(max_mmr_idx)

        # Retourner documents sélectionnés dans l'ordre MMR
        return [documents[i] for i in selected_indices]

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calcule la similarité cosine entre deux vecteurs."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))


def process_reranking(
    queries: List[str],
    results: List[List[Dict[str, Any]]],
    config: Dict[str, Any],
    doc_embeddings: Optional[List[np.ndarray]] = None,
) -> List[List[Dict[str, Any]]]:
    """Point d'entrée principal pour le reranking.

    Args:
        queries: Liste des queries textuelles.
        results: Résultats du retrieval pour chaque query.
        config: Configuration complète chargée depuis YAML.
        doc_embeddings: Embeddings des documents (optionnel pour MMR).

    Returns:
        Résultats reranked et diversifiés.
    """
    reranking_config = config.get("reranking", {})

    # Cross-encoder reranking
    cross_encoder = CrossEncoderReranker(config)
    reranked_results = cross_encoder.rerank(queries, results)

    # MMR pour diversité
    mmr = MMR(config)
    final_results = mmr.apply(
        reranked_results,
        doc_embeddings=doc_embeddings,
        top_k=reranking_config.get("final_top_k", 10),
    )

    return final_results


# =============================================================================
# ADVANCED FEATURES - LLM Reranking (RankGPT-style)
# =============================================================================


class LLMReranker:
    """Reranker utilisant un LLM pour reranking listwise.

    RankGPT-style : Utilise le LLM pour ordonner directement les documents
    selon leur pertinence, plutôt que scorer individuellement.

    Plus précis que cross-encoder mais plus lent (+1-2s par query).

    Attributes:
        config: Configuration du LLM reranking.
        client: Client LLM.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le LLM reranker.

        Args:
            config: Configuration complète depuis YAML.
        """
        self.config = config.get("llm_reranking", {})
        self.client: Optional[Any] = None
        self._initialize_client(config)

    def _initialize_client(self, config: Dict[str, Any]) -> None:
        """Initialise le client LLM."""
        if not self.config.get("enabled", False):
            return

        try:
            import openai

            llm_config = self.config.get("llm", {})
            provider = llm_config.get("provider", "ollama")

            if provider == "ollama":
                self.client = openai.OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                )
            elif provider == "openai":
                self.client = openai.OpenAI()
            else:
                self.client = None

        except ImportError:
            print("⚠️  openai library not installed for LLM reranking")
            self.client = None

    def rerank(
        self,
        queries: List[str],
        results: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Rerank avec LLM (listwise).

        Args:
            queries: Liste des queries.
            results: Résultats à reranker.
            top_k: Nombre de documents finaux.

        Returns:
            Documents reranked.
        """
        if not self.config.get("enabled", False) or self.client is None:
            return results

        method = self.config.get("method", "listwise")

        if method == "listwise":
            return self._rerank_listwise(queries, results, top_k)
        elif method == "pairwise":
            return self._rerank_pairwise(queries, results, top_k)
        else:
            return results

    def _rerank_listwise(
        self,
        queries: List[str],
        results: List[List[Dict[str, Any]]],
        top_k: Optional[int],
    ) -> List[List[Dict[str, Any]]]:
        """Reranking listwise (RankGPT-style).

        Le LLM voit tous les documents et les ordonne directement.

        Args:
            queries: Queries.
            results: Résultats.
            top_k: Top-K final.

        Returns:
            Documents reranked.
        """
        reranked_results = []

        for query, query_results in zip(queries, results):
            if not query_results:
                reranked_results.append([])
                continue

            # Limiter nombre de docs à reranker (performance)
            max_docs = self.config.get("max_documents_to_rerank", 10)
            docs_to_rerank = query_results[:max_docs]

            # Construire prompt listwise
            prompt = self._build_listwise_prompt(query, docs_to_rerank)

            # Appeler LLM
            try:
                llm_config = self.config.get("llm", {})
                model = llm_config.get("model", "llama3")
                temperature = llm_config.get("temperature", 0.0)
                max_tokens = llm_config.get("max_tokens", 500)

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )

                ranking = response.choices[0].message.content.strip()

                # Parser ranking (format: "1 > 3 > 2 > 4")
                reranked_docs = self._parse_ranking(ranking, docs_to_rerank)

                # Ajouter docs non reranked à la fin
                remaining_docs = query_results[max_docs:]
                final_docs = reranked_docs + remaining_docs

                # Limiter au top-k si spécifié
                if top_k:
                    final_docs = final_docs[:top_k]

                reranked_results.append(final_docs)

            except Exception as e:
                print(f"⚠️  LLM reranking failed: {e}. Using original order.")
                reranked_results.append(query_results)

        return reranked_results

    def _build_listwise_prompt(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> str:
        """Construit le prompt listwise.

        Args:
            query: Query utilisateur.
            documents: Documents à reranker.

        Returns:
            Prompt formaté.
        """
        prompt_template = self.config.get(
            "prompt_template",
            """Rank the following documents by relevance to the query.
Output ONLY the ranking in format: "1 > 2 > 3 > 4" (most relevant first).

Query: {query}

Documents:
{documents}

Ranking (numbers only, separated by >):""",
        )

        # Formatter documents
        docs_text = ""
        for i, doc in enumerate(documents, start=1):
            content = doc.get("document", "")[:200]  # Limiter longueur
            docs_text += f"[{i}] {content}\n\n"

        return prompt_template.format(query=query, documents=docs_text)

    def _parse_ranking(
        self, ranking_str: str, original_docs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Parse le ranking retourné par le LLM.

        Args:
            ranking_str: String du ranking (ex: "1 > 3 > 2").
            original_docs: Documents originaux.

        Returns:
            Documents réordonnés.
        """
        # Extraire numéros
        import re

        numbers = re.findall(r"\d+", ranking_str)

        # Convertir en indices (1-based → 0-based)
        try:
            indices = [int(n) - 1 for n in numbers]

            # Vérifier validité
            valid_indices = [i for i in indices if 0 <= i < len(original_docs)]

            # Réordonner
            reranked = [original_docs[i] for i in valid_indices]

            # Ajouter docs manquants
            seen_indices = set(valid_indices)
            for i, doc in enumerate(original_docs):
                if i not in seen_indices:
                    reranked.append(doc)

            return reranked

        except (ValueError, IndexError):
            # Si parsing échoue, retourner ordre original
            return original_docs

    def _rerank_pairwise(
        self,
        queries: List[str],
        results: List[List[Dict[str, Any]]],
        top_k: Optional[int],
    ) -> List[List[Dict[str, Any]]]:
        """Reranking pairwise (comparaisons par paires).

        Le LLM compare chaque paire de documents.
        Plus lent que listwise mais potentiellement plus précis.

        Args:
            queries: Queries.
            results: Résultats.
            top_k: Top-K final.

        Returns:
            Documents reranked.
        """
        # Simplification : bubble sort avec LLM
        # (très lent, à utiliser seulement pour petits ensembles)

        reranked_results = []

        for query, query_results in zip(queries, results):
            if len(query_results) <= 1:
                reranked_results.append(query_results)
                continue

            # Limiter nombre de docs
            max_docs = min(5, len(query_results))  # Pairwise très coûteux
            docs_to_compare = query_results[:max_docs].copy()

            # Bubble sort avec LLM comparisons
            n = len(docs_to_compare)
            for i in range(n):
                for j in range(0, n - i - 1):
                    # Comparer docs[j] et docs[j+1]
                    if self._llm_compare(
                        query, docs_to_compare[j], docs_to_compare[j + 1]
                    ):
                        # Swap si j+1 plus pertinent
                        docs_to_compare[j], docs_to_compare[j + 1] = (
                            docs_to_compare[j + 1],
                            docs_to_compare[j],
                        )

            # Ajouter docs restants
            final_docs = docs_to_compare + query_results[max_docs:]

            if top_k:
                final_docs = final_docs[:top_k]

            reranked_results.append(final_docs)

        return reranked_results

    def _llm_compare(
        self, query: str, doc1: Dict[str, Any], doc2: Dict[str, Any]
    ) -> bool:
        """Compare deux documents avec LLM.

        Args:
            query: Query.
            doc1: Premier document.
            doc2: Deuxième document.

        Returns:
            True si doc2 plus pertinent que doc1.
        """
        if self.client is None:
            return False

        prompt = f"""Which document is more relevant to the query?

Query: {query}

Document A: {doc1.get("document", "")[:200]}

Document B: {doc2.get("document", "")[:200]}

Answer with only "A" or "B":"""

        try:
            llm_config = self.config.get("llm", {})
            model = llm_config.get("model", "llama3")

            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=5,
            )

            answer = response.choices[0].message.content.strip().upper()

            return "B" in answer

        except Exception:
            return False


if __name__ == "__main__":
    """Exemple d'exécution du module."""
    print("=" * 80)
    print("RERANKING - EXEMPLE")
    print("=" * 80)
    print("\n⚠️  Ce module nécessite des résultats de retrieval en entrée.")
    print("\nExemple d'utilisation :")
    print("""
    from inference_project.steps.step_02_retrieval import process_retrieval
    from inference_project.steps.step_03_reranking import process_reranking
    from inference_project.utils.config_loader import load_config

    # Charger config
    config = load_config("03_reranking_v2", "config")

    # Queries
    queries = ["What is machine learning?", "How does RAG work?"]

    # Supposons que nous ayons des résultats de retrieval
    retrieval_results = [
        [
            {"id": "doc1", "document": "ML is...", "score": 0.85, "source": "dense"},
            {"id": "doc2", "document": "AI is...", "score": 0.75, "source": "dense"},
        ],
        [
            {"id": "doc3", "document": "RAG combines...", "score": 0.90, "source": "dense"},
            {"id": "doc4", "document": "Retrieval...", "score": 0.80, "source": "dense"},
        ],
    ]

    # Reranking
    reranked_results = process_reranking(queries, retrieval_results, config)

    # Afficher résultats
    for i, query_results in enumerate(reranked_results):
        print(f"\\nQuery {i+1}: {queries[i]}")
        for j, doc in enumerate(query_results, start=1):
            print(f"  {j}. Score: {doc['score']:.4f} | {doc['document'][:100]}")
    """)
    print("=" * 80)
