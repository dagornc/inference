"""Module pour la génération d'embeddings (Phase 01 [1.9]).

Ce module génère les embeddings vectoriels pour les queries traitées.
Il supporte trois types d'embeddings selon la configuration :
- Dense embeddings : Vecteurs denses pour semantic search (BGE-M3, etc.)
- Sparse embeddings : Vecteurs creux pour lexical matching (SPLADE, BM25)
- Late Interaction : Token-level embeddings (ColBERT)

Fonctions:
    generate_dense_embeddings: Génère des embeddings denses.
    generate_sparse_embeddings: Génère des embeddings sparse.
    generate_late_interaction_embeddings: Génère des embeddings late interaction.
    process_embeddings: Point d'entrée principal orchestrant les embeddings.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingGenerator:
    """Générateur d'embeddings avec support multi-modal (dense/sparse/late).

    Cette classe gère la génération d'embeddings selon la configuration.
    Elle initialise les modèles nécessaires et applique les transformations
    configurées (normalisation, pooling, etc.).

    Attributes:
        config: Configuration de l'étape d'embedding.
        dense_model: Modèle pour dense embeddings (si activé).
        sparse_model: Modèle pour sparse embeddings (si activé).
        late_interaction_model: Modèle pour late interaction (si activé).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le générateur d'embeddings.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config
        self.embedding_config = config.get("embedding_generation", {})

        # Initialiser les modèles selon config
        self.dense_model: Optional[SentenceTransformer] = None
        self.sparse_model: Optional[Any] = None
        self.late_interaction_model: Optional[Any] = None

        self._initialize_models()

    def _initialize_models(self) -> None:
        """Initialise les modèles d'embedding selon la configuration."""
        # Dense embedding model
        if self.embedding_config.get("dense_embedding", {}).get("enabled", True):
            dense_config = self.embedding_config.get("dense_embedding", {})
            model_name = dense_config.get("model", "BAAI/bge-m3")
            device = dense_config.get("device", "cpu")

            self.dense_model = SentenceTransformer(model_name, device=device)

        # Sparse embedding model (optionnel pour v1)
        if self.embedding_config.get("sparse_embedding", {}).get("enabled", False):
            # TODO: Implémenter SPLADE ou autre sparse model
            pass

        # Late interaction model (optionnel pour v1)
        if self.embedding_config.get("late_interaction", {}).get("enabled", False):
            # TODO: Implémenter ColBERT
            pass

    def generate_dense_embeddings(
        self, texts: List[str], normalize: bool = True
    ) -> np.ndarray:
        """Génère des embeddings denses pour une liste de textes.

        Args:
            texts: Liste des textes à embedder.
            normalize: Si True, normalise les vecteurs (norme L2 = 1).

        Returns:
            Array numpy de shape (n_texts, embedding_dim) contenant les embeddings.

        Raises:
            ValueError: Si le modèle dense n'est pas initialisé.
        """
        if self.dense_model is None:
            raise ValueError("Dense embedding model not initialized")

        dense_config = self.embedding_config.get("dense_embedding", {})

        # Paramètres d'encodage
        batch_size = dense_config.get("batch_size", 32)
        show_progress_bar = dense_config.get("show_progress_bar", False)
        convert_to_numpy = True
        normalize_embeddings = normalize or dense_config.get("normalize", True)

        # Générer embeddings
        embeddings = self.dense_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )

        # Appliquer pooling si nécessaire (déjà fait par sentence-transformers)
        # Le pooling est configuré dans le modèle lui-même

        return embeddings

    def generate_sparse_embeddings(self, texts: List[str]) -> List[Dict[int, float]]:
        """Génère des embeddings sparse pour une liste de textes.

        Les embeddings sparse sont représentés comme des dictionnaires
        {token_id: weight} pour économiser de la mémoire.

        Args:
            texts: Liste des textes à embedder.

        Returns:
            Liste de dictionnaires sparse {token_id: weight}.

        Raises:
            NotImplementedError: Sparse embeddings non encore implémentés.
        """
        if self.sparse_model is None:
            raise NotImplementedError(
                "Sparse embeddings not yet implemented. "
                "Use dense embeddings or implement SPLADE."
            )

        # TODO: Implémenter avec SPLADE ou BM25 vectorization
        return [{} for _ in texts]

    def generate_late_interaction_embeddings(
        self, texts: List[str]
    ) -> List[np.ndarray]:
        """Génère des embeddings late interaction (token-level).

        Contrairement aux dense embeddings qui produisent 1 vecteur par texte,
        late interaction produit 1 vecteur par token (comme ColBERT).

        Args:
            texts: Liste des textes à embedder.

        Returns:
            Liste d'arrays numpy de shape (n_tokens, embedding_dim).

        Raises:
            NotImplementedError: Late interaction non encore implémenté.
        """
        if self.late_interaction_model is None:
            raise NotImplementedError(
                "Late interaction embeddings not yet implemented. "
                "Use dense embeddings or implement ColBERT."
            )

        # TODO: Implémenter avec ColBERT via ragatouille
        return [np.array([]) for _ in texts]

    def process(self, queries: List[str]) -> Dict[str, Any]:
        """Point d'entrée principal pour générer tous les embeddings configurés.

        Args:
            queries: Liste des queries (déjà expansées) à embedder.

        Returns:
            Dictionnaire contenant :
                - "queries": Les queries originales
                - "dense_embeddings": Embeddings denses (si activé)
                - "sparse_embeddings": Embeddings sparse (si activé)
                - "late_interaction_embeddings": Embeddings late interaction (si activé)
                - "metadata": Métadonnées (dimensions, modèles utilisés, etc.)
        """
        results: Dict[str, Any] = {
            "queries": queries,
            "metadata": {
                "num_queries": len(queries),
                "models_used": [],
            },
        }

        # Dense embeddings
        if self.embedding_config.get("dense_embedding", {}).get("enabled", True):
            dense_embeddings = self.generate_dense_embeddings(queries)
            results["dense_embeddings"] = dense_embeddings
            results["metadata"]["models_used"].append("dense")
            results["metadata"]["dense_dim"] = dense_embeddings.shape[1]

        # Sparse embeddings (optionnel)
        if self.embedding_config.get("sparse_embedding", {}).get("enabled", False):
            try:
                sparse_embeddings = self.generate_sparse_embeddings(queries)
                results["sparse_embeddings"] = sparse_embeddings
                results["metadata"]["models_used"].append("sparse")
            except NotImplementedError:
                # Sparse non implémenté, on continue sans
                pass

        # Late interaction embeddings (optionnel)
        if self.embedding_config.get("late_interaction", {}).get("enabled", False):
            try:
                late_embeddings = self.generate_late_interaction_embeddings(queries)
                results["late_interaction_embeddings"] = late_embeddings
                results["metadata"]["models_used"].append("late_interaction")
            except NotImplementedError:
                # Late interaction non implémenté, on continue sans
                pass

        return results


def process_embeddings(queries: List[str], config: Dict[str, Any]) -> Dict[str, Any]:
    """Fonction helper pour générer les embeddings (interface simplifiée).

    Args:
        queries: Liste des queries à embedder.
        config: Configuration complète chargée depuis YAML.

    Returns:
        Dictionnaire contenant les embeddings et métadonnées.
    """
    generator = EmbeddingGenerator(config)
    return generator.process(queries)


if __name__ == "__main__":
    """Exemple d'exécution du module."""
    from inference_project.utils.config_loader import load_config

    # Charger configuration
    config = load_config("01_embedding_v2", "config")

    # Queries d'exemple
    test_queries = [
        "Quels sont les bénéfices d'un modèle RAG ?",
        "Comment implémenter un système de retrieval hybride ?",
        "Quelle est la différence entre dense et sparse embeddings ?",
    ]

    print("=" * 80)
    print("GÉNÉRATION D'EMBEDDINGS - EXEMPLE")
    print("=" * 80)
    print(f"\nNombre de queries : {len(test_queries)}")
    print(f"Modèle : {config['embedding_generation']['dense_embedding']['model']}")

    # Générer embeddings
    generator = EmbeddingGenerator(config)
    results = generator.process(test_queries)

    # Afficher résultats
    print("\n--- RÉSULTATS ---")
    print(f"Queries traitées : {results['metadata']['num_queries']}")
    print(f"Modèles utilisés : {results['metadata']['models_used']}")

    if "dense_embeddings" in results:
        embeddings = results["dense_embeddings"]
        print(f"\nDense embeddings shape : {embeddings.shape}")
        print(f"Dimension : {results['metadata']['dense_dim']}")
        print(f"Norme du 1er vecteur : {np.linalg.norm(embeddings[0]):.4f}")
        print(f"Similarité query 1-2 : {np.dot(embeddings[0], embeddings[1]):.4f}")
        print(f"Similarité query 1-3 : {np.dot(embeddings[0], embeddings[2]):.4f}")

    print("\n" + "=" * 80)
    print("✅ Embeddings générés avec succès !")
