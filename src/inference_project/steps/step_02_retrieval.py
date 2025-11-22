"""Module pour le retrieval multi-modal (Phase 02).

Ce module implémente trois stratégies de retrieval :
- Dense retrieval : Recherche vectorielle sémantique (ChromaDB, Qdrant, etc.)
- Sparse retrieval : Recherche lexicale (BM25 via Pyserini)
- Hybrid fusion : Combinaison des résultats (RRF, weighted, etc.)

Fonctions:
    DenseRetriever: Classe pour dense retrieval.
    SparseRetriever: Classe pour sparse retrieval (BM25).
    HybridFusion: Classe pour fusionner les résultats.
    process_retrieval: Point d'entrée principal orchestrant le retrieval.
"""

from typing import Any, Dict, List, Optional

import numpy as np

# =============================================================================
# CLASSES DE RETRIEVAL
# =============================================================================


class DenseRetriever:
    """Retriever dense utilisant la recherche vectorielle sémantique.

    Supporte plusieurs vector databases : ChromaDB, Qdrant, Weaviate, etc.

    Attributes:
        config: Configuration du dense retrieval.
        vector_db_client: Client de la vector database.
        collection_name: Nom de la collection/index à interroger.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le dense retriever.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("dense_retrieval", {})
        self.vector_db = self.config.get("vector_db", "chromadb")
        self.collection_name = self.config.get("collection_name", "documents")

        self.client: Optional[Any] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialise le client de la vector database."""
        if not self.config.get("enabled", True):
            return

        if self.vector_db == "chromadb":
            try:
                import chromadb

                # Mode persistent ou in-memory
                persist_directory = self.config.get("persist_directory", None)
                if persist_directory:
                    self.client = chromadb.PersistentClient(path=persist_directory)
                else:
                    self.client = chromadb.Client()

            except ImportError:
                raise ImportError("chromadb not installed. Run: pip install chromadb")

        elif self.vector_db == "qdrant":
            try:
                from qdrant_client import QdrantClient

                host = self.config.get("host", "localhost")
                port = self.config.get("port", 6333)
                self.client = QdrantClient(host=host, port=port)

            except ImportError:
                raise ImportError(
                    "qdrant-client not installed. Run: pip install qdrant-client"
                )

        else:
            raise ValueError(f"Unsupported vector_db: {self.vector_db}")

    def search(
        self, query_embeddings: np.ndarray, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Effectue une recherche dense avec les embeddings de query.

        Args:
            query_embeddings: Embeddings des queries (shape: n_queries x dim).
            top_k: Nombre de résultats à retourner par query.

        Returns:
            Liste de résultats avec métadonnées et scores.
        """
        if not self.config.get("enabled", True):
            return []

        if self.client is None:
            raise ValueError("Vector DB client not initialized")

        # ChromaDB
        if self.vector_db == "chromadb":
            collection = self.client.get_or_create_collection(name=self.collection_name)

            results = collection.query(
                query_embeddings=query_embeddings.tolist(),
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )

            # Formatter résultats
            formatted_results = []
            for i, (ids, documents, metadatas, distances) in enumerate(
                zip(
                    results["ids"],
                    results["documents"],
                    results["metadatas"],
                    results["distances"],
                )
            ):
                query_results = []
                for doc_id, doc, meta, dist in zip(
                    ids, documents, metadatas, distances
                ):
                    # Distance → Score (plus proche = meilleur)
                    # ChromaDB utilise L2 distance, donc on inverse
                    score = 1.0 / (1.0 + dist)

                    query_results.append(
                        {
                            "id": doc_id,
                            "document": doc,
                            "metadata": meta,
                            "score": score,
                            "source": "dense",
                        }
                    )
                formatted_results.append(query_results)

            return formatted_results

        # Qdrant
        elif self.vector_db == "qdrant":
            # TODO: Implémenter Qdrant search
            raise NotImplementedError("Qdrant search not yet implemented")

        return []


class SparseRetriever:
    """Retriever sparse utilisant BM25 (via Pyserini).

    BM25 est une méthode de ranking lexicale basée sur la fréquence des termes.

    Attributes:
        config: Configuration du sparse retrieval.
        searcher: Pyserini BM25 searcher.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le sparse retriever.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("sparse_retrieval", {})
        self.method = self.config.get("method", "bm25")

        self.searcher: Optional[Any] = None
        self._initialize_searcher()

    def _initialize_searcher(self) -> None:
        """Initialise le searcher BM25."""
        if not self.config.get("enabled", True):
            return

        if self.method == "bm25":
            try:
                from pyserini.search.lucene import LuceneSearcher

                index_dir = self.config.get("index_dir", None)
                if not index_dir:
                    raise ValueError("index_dir required for BM25 sparse retrieval")

                self.searcher = LuceneSearcher(index_dir)

            except ImportError:
                raise ImportError("pyserini not installed. Run: pip install pyserini")
        else:
            raise ValueError(f"Unsupported sparse method: {self.method}")

    def search(self, queries: List[str], top_k: int = 10) -> List[List[Dict[str, Any]]]:
        """Effectue une recherche sparse avec BM25.

        Args:
            queries: Liste des queries textuelles.
            top_k: Nombre de résultats à retourner par query.

        Returns:
            Liste de résultats pour chaque query.
        """
        if not self.config.get("enabled", True):
            return []

        if self.searcher is None:
            raise ValueError("Sparse searcher not initialized")

        results_all_queries = []

        for query in queries:
            hits = self.searcher.search(query, k=top_k)

            query_results = []
            for hit in hits:
                query_results.append(
                    {
                        "id": hit.docid,
                        "document": hit.raw if hasattr(hit, "raw") else "",
                        "metadata": {},
                        "score": hit.score,
                        "source": "sparse",
                    }
                )

            results_all_queries.append(query_results)

        return results_all_queries


class HybridFusion:
    """Classe pour fusionner les résultats dense et sparse.

    Implémente plusieurs méthodes de fusion :
    - RRF (Reciprocal Rank Fusion) : 1 / (k + rank)
    - Weighted : Combinaison linéaire des scores normalisés
    - Distribution-based : Normalisation par distribution

    Attributes:
        config: Configuration de la fusion.
        method: Méthode de fusion ('rrf', 'weighted', 'distribution').
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le module de fusion.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("fusion", {})
        self.method = self.config.get("method", "rrf")

    def fuse(
        self,
        dense_results: List[List[Dict[str, Any]]],
        sparse_results: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Fusionne les résultats dense et sparse.

        Args:
            dense_results: Résultats du dense retrieval (liste par query).
            sparse_results: Résultats du sparse retrieval (liste par query).

        Returns:
            Résultats fusionnés et triés par score pour chaque query.
        """
        if self.method == "rrf":
            return self._reciprocal_rank_fusion(dense_results, sparse_results)
        elif self.method == "weighted":
            return self._weighted_fusion(dense_results, sparse_results)
        elif self.method == "distribution":
            return self._distribution_based_fusion(dense_results, sparse_results)
        else:
            raise ValueError(f"Unsupported fusion method: {self.method}")

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[List[Dict[str, Any]]],
        sparse_results: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Reciprocal Rank Fusion (RRF).

        Score RRF = 1 / (k + rank) où k=60 par défaut.
        """
        k = self.config.get("rrf_k", 60)
        fused_results = []

        for dense_res, sparse_res in zip(dense_results, sparse_results):
            # Map doc_id → score RRF
            doc_scores: Dict[str, float] = {}
            doc_data: Dict[str, Dict[str, Any]] = {}

            # Ajouter scores dense
            for rank, doc in enumerate(dense_res, start=1):
                doc_id = doc["id"]
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc

            # Ajouter scores sparse
            for rank, doc in enumerate(sparse_res, start=1):
                doc_id = doc["id"]
                doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
                if doc_id not in doc_data:
                    doc_data[doc_id] = doc

            # Trier par score décroissant
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

            # Reconstruire liste de résultats
            query_results = []
            for doc_id, score in sorted_docs:
                doc = doc_data[doc_id].copy()
                doc["score"] = score
                doc["source"] = "hybrid_rrf"
                query_results.append(doc)

            fused_results.append(query_results)

        return fused_results

    def _weighted_fusion(
        self,
        dense_results: List[List[Dict[str, Any]]],
        sparse_results: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Weighted fusion avec normalisation des scores."""
        dense_weight = self.config.get("dense_weight", 0.7)
        sparse_weight = self.config.get("sparse_weight", 0.3)

        fused_results = []

        for dense_res, sparse_res in zip(dense_results, sparse_results):
            doc_scores: Dict[str, float] = {}
            doc_data: Dict[str, Dict[str, Any]] = {}

            # Normaliser et ajouter scores dense
            if dense_res:
                max_dense = max(doc["score"] for doc in dense_res)
                for doc in dense_res:
                    doc_id = doc["id"]
                    normalized_score = doc["score"] / max_dense if max_dense > 0 else 0
                    doc_scores[doc_id] = (
                        doc_scores.get(doc_id, 0.0) + dense_weight * normalized_score
                    )
                    if doc_id not in doc_data:
                        doc_data[doc_id] = doc

            # Normaliser et ajouter scores sparse
            if sparse_res:
                max_sparse = max(doc["score"] for doc in sparse_res)
                for doc in sparse_res:
                    doc_id = doc["id"]
                    normalized_score = (
                        doc["score"] / max_sparse if max_sparse > 0 else 0
                    )
                    doc_scores[doc_id] = (
                        doc_scores.get(doc_id, 0.0) + sparse_weight * normalized_score
                    )
                    if doc_id not in doc_data:
                        doc_data[doc_id] = doc

            # Trier et reconstruire
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

            query_results = []
            for doc_id, score in sorted_docs:
                doc = doc_data[doc_id].copy()
                doc["score"] = score
                doc["source"] = "hybrid_weighted"
                query_results.append(doc)

            fused_results.append(query_results)

        return fused_results

    def _distribution_based_fusion(
        self,
        dense_results: List[List[Dict[str, Any]]],
        sparse_results: List[List[Dict[str, Any]]],
    ) -> List[List[Dict[str, Any]]]:
        """Distribution-based fusion (normalisation par distribution)."""
        # TODO: Implémenter normalisation par distribution
        # Pour l'instant, fallback sur weighted
        return self._weighted_fusion(dense_results, sparse_results)


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def process_retrieval(
    query_embeddings: np.ndarray,
    queries_text: List[str],
    config: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """Point d'entrée principal pour le retrieval hybride.

    Args:
        query_embeddings: Embeddings denses des queries (n_queries x dim).
        queries_text: Texte des queries (pour sparse retrieval).
        config: Configuration complète chargée depuis YAML.

    Returns:
        Résultats fusionnés pour chaque query.
    """
    retrieval_config = config.get("retrieval", {})

    # Top-K global
    top_k = retrieval_config.get("top_k", 10)

    # Dense retrieval
    dense_retriever = DenseRetriever(config)
    dense_results = (
        dense_retriever.search(query_embeddings, top_k=top_k)
        if dense_retriever.config.get("enabled", True)
        else [[] for _ in queries_text]
    )

    # Sparse retrieval
    sparse_retriever = SparseRetriever(config)
    sparse_results = (
        sparse_retriever.search(queries_text, top_k=top_k)
        if sparse_retriever.config.get("enabled", True)
        else [[] for _ in queries_text]
    )

    # Fusion
    if retrieval_config.get("fusion", {}).get("enabled", True):
        fusion = HybridFusion(config)
        final_results = fusion.fuse(dense_results, sparse_results)
    else:
        # Si fusion désactivée, retourner seulement dense results
        final_results = dense_results

    # Limiter au top_k final après fusion
    final_top_k = retrieval_config.get("final_top_k", top_k)
    final_results = [res[:final_top_k] for res in final_results]

    return final_results


# =============================================================================
# ADVANCED FEATURES - Iterative Retrieval & Metadata Filtering
# =============================================================================


class IterativeRetriever:
    """Retriever itératif pour queries multi-hop.

    Effectue plusieurs tours de retrieval en utilisant les résultats
    précédents pour raffiner la recherche.

    Exemple: "Compare X and Y"
    - Hop 1: Retrieve docs about X
    - Hop 2: Retrieve docs about Y
    - Hop 3: Retrieve docs about "X vs Y"

    Attributes:
        config: Configuration du retrieval itératif.
        dense_retriever: Instance du dense retriever.
        sparse_retriever: Instance du sparse retriever.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le retriever itératif.

        Args:
            config: Configuration complète depuis YAML.
        """
        self.config = config.get("iterative_retrieval", {})
        self.dense_retriever = DenseRetriever(config)
        self.sparse_retriever = SparseRetriever(config)

    def retrieve_iterative(
        self,
        sub_queries: List[str],
        query_embeddings: np.ndarray,
        config: Dict[str, Any],
    ) -> List[List[Dict[str, Any]]]:
        """Effectue retrieval itératif pour sous-questions.

        Args:
            sub_queries: Liste de sous-questions (décomposition).
            query_embeddings: Embeddings de toutes les sub-queries.
            config: Configuration globale.

        Returns:
            Résultats fusionnés de tous les hops.
        """
        if not self.config.get("enabled", False):
            # Retrieval standard (prendre seulement première query)
            first_embedding = query_embeddings[0:1]
            first_query = [sub_queries[0]] if sub_queries else [""]
            return process_retrieval(first_embedding, first_query, config)

        max_hops = self.config.get("max_hops", 3)
        top_k_per_hop = self.config.get("top_k_per_hop", 5)

        all_results: List[Dict[str, Any]] = []
        seen_doc_ids = set()

        # Hop itératif
        for hop in range(min(max_hops, len(sub_queries))):
            # Embedding pour cette sub-query
            if hop >= query_embeddings.shape[0]:
                break

            query_embedding = query_embeddings[hop : hop + 1]
            query_text = [sub_queries[hop]]

            # Retrieval dense
            dense_results = []
            if self.dense_retriever.config.get("enabled", True):
                dense_results = self.dense_retriever.search(
                    query_embedding, top_k=top_k_per_hop
                )

            # Retrieval sparse
            sparse_results = []
            if self.sparse_retriever.config.get("enabled", False):
                sparse_results = self.sparse_retriever.search(
                    query_text, top_k=top_k_per_hop
                )

            # Fusionner résultats de ce hop
            hop_results = self._fuse_hop_results(
                dense_results[0] if dense_results else [],
                sparse_results[0] if sparse_results else [],
            )

            # Ajouter seulement docs non vus
            for doc in hop_results:
                doc_id = doc.get("id", "")
                if doc_id and doc_id not in seen_doc_ids:
                    doc["hop"] = hop + 1
                    doc["sub_query"] = sub_queries[hop]
                    all_results.append(doc)
                    seen_doc_ids.add(doc_id)

        # Limiter au top-K final
        final_top_k = self.config.get("final_top_k", 10)

        # Trier par score décroissant
        all_results_sorted = sorted(
            all_results, key=lambda x: x.get("score", 0.0), reverse=True
        )

        return [all_results_sorted[:final_top_k]]

    def _fuse_hop_results(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Fusionne résultats dense + sparse pour un hop.

        Args:
            dense_results: Résultats dense.
            sparse_results: Résultats sparse.

        Returns:
            Résultats fusionnés.
        """
        # Fusion simple par RRF
        doc_scores: Dict[str, float] = {}
        doc_data: Dict[str, Dict[str, Any]] = {}
        k = 60

        # Scores dense
        for rank, doc in enumerate(dense_results, start=1):
            doc_id = doc["id"]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

        # Scores sparse
        for rank, doc in enumerate(sparse_results, start=1):
            doc_id = doc["id"]
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            if doc_id not in doc_data:
                doc_data[doc_id] = doc

        # Trier
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Reconstruire
        fused_results = []
        for doc_id, score in sorted_docs:
            doc = doc_data[doc_id].copy()
            doc["score"] = score
            fused_results.append(doc)

        return fused_results


class MetadataFilter:
    """Filtre de métadonnées pour retrieval ciblé.

    Permet de filtrer documents selon :
    - Temporal filters (date range)
    - Source filters (specific sources)
    - Domain filters (technical, business, etc.)

    Self-Query : extrait filtres automatiquement de la query.

    Attributes:
        config: Configuration du filtrage.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le filtre.

        Args:
            config: Configuration complète depuis YAML.
        """
        self.config = config.get("metadata_filter", {})

    def extract_filters_from_query(self, query: str) -> Dict[str, Any]:
        """Extrait filtres de métadonnées depuis la query.

        Self-Query : analyse la query pour détecter filtres.

        Args:
            query: Query utilisateur.

        Returns:
            Dict de filtres à appliquer.
        """
        if not self.config.get("enabled", False):
            return {}

        filters: Dict[str, Any] = {}
        query_lower = query.lower()

        # Filtres temporels
        temporal_keywords = {
            "recent": {"days": 30},
            "last week": {"days": 7},
            "last month": {"days": 30},
            "last year": {"days": 365},
            "today": {"days": 1},
        }

        for keyword, filter_val in temporal_keywords.items():
            if keyword in query_lower:
                filters["temporal"] = filter_val
                break

        # Filtres de source
        source_keywords = {
            "documentation": ["docs", "documentation"],
            "blog": ["blog", "article"],
            "paper": ["paper", "research"],
            "code": ["code", "github"],
        }

        for source_type, keywords in source_keywords.items():
            if any(kw in query_lower for kw in keywords):
                filters["source_type"] = source_type
                break

        # Filtres de domaine
        if "technical" in query_lower or "code" in query_lower:
            filters["domain"] = "technical"
        elif "business" in query_lower or "market" in query_lower:
            filters["domain"] = "business"

        return filters

    def apply_filters(
        self,
        results: List[Dict[str, Any]],
        filters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Applique filtres aux résultats de retrieval.

        Args:
            results: Résultats bruts.
            filters: Filtres à appliquer.

        Returns:
            Résultats filtrés.
        """
        if not filters:
            return results

        filtered_results = []

        for doc in results:
            metadata = doc.get("metadata", {})

            # Vérifier chaque filtre
            passes_filters = True

            # Filtre source_type
            if "source_type" in filters:
                source = metadata.get("source", "").lower()
                if filters["source_type"] not in source:
                    passes_filters = False

            # Filtre domain
            if "domain" in filters:
                domain = metadata.get("domain", "").lower()
                if domain and filters["domain"] != domain:
                    passes_filters = False

            # Filtre temporal (simplification : vérifier metadata "date")
            if "temporal" in filters:
                # Nécessiterait parsing de dates réelles
                # Pour simplifier, on accepte tous les docs
                pass

            if passes_filters:
                filtered_results.append(doc)

        return filtered_results


if __name__ == "__main__":
    """Exemple d'exécution du module (nécessite une DB pré-indexée)."""
    print("=" * 80)
    print("RETRIEVAL - EXEMPLE")
    print("=" * 80)
    print("\n⚠️  Ce module nécessite une vector database pré-indexée.")
    print("Veuillez d'abord indexer vos documents avec ChromaDB ou Qdrant.")
    print("\nExemple d'utilisation :")
    print("""
    from inference_project.steps.step_01_embedding_generation import process_embeddings
    from inference_project.steps.step_02_retrieval import process_retrieval
    from inference_project.utils.config_loader import load_config

    # Charger config
    config = load_config("02_retrieval_v2", "config")

    # Queries
    queries = ["What is machine learning?", "How does RAG work?"]

    # Générer embeddings
    embedding_results = process_embeddings(queries, config)
    query_embeddings = embedding_results["dense_embeddings"]

    # Retrieval
    results = process_retrieval(query_embeddings, queries, config)

    # Afficher résultats
    for i, query_results in enumerate(results):
        print(f"\\nQuery {i+1}: {queries[i]}")
        for j, doc in enumerate(query_results[:3], start=1):
            print(f"  {j}. Score: {doc['score']:.4f} | {doc['document'][:100]}")
    """)
    print("=" * 80)
