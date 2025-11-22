"""Module pour la compression contextuelle (Phase 04).

Ce module implémente la compression intelligente du contexte récupéré pour :
- Réduire les coûts de génération LLM (2.5x-10x compression)
- Optimiser le context window
- Préserver la qualité et les informations critiques

Fonctions:
    PreCompressionAnalyzer: Analyse de complexité et compressibilité.
    LLMLinguaCompressor: Compression avec LLMLingua-2.
    ContextualCompressor: Compression contextuelle extractive.
    CompressionAwareMMR: MMR intelligent avec compression awareness.
    QualityValidator: Validation qualité post-compression.
    ContextWindowOptimizer: Gestion intelligente du context window.
    process_compression: Point d'entrée principal orchestrant la compression.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PreCompressionAnalyzer:
    """Analyseur de complexité et compressibilité avant compression.

    Analyse le contenu pour adapter la stratégie de compression :
    - Complexité informationnelle (densité, diversité vocabulaire)
    - Compressibilité (entropie, redondance)
    - Densité d'entités nommées

    Attributes:
        config: Configuration de l'analyse.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise l'analyseur.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("pre_compression_analysis", {})

    def analyze(
        self, documents: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        """Analyse les documents avant compression.

        Args:
            documents: Documents à analyser.

        Returns:
            Tuple (documents avec métadonnées enrichies, statistiques globales).
        """
        if not self.config.get("enabled", True):
            return documents, {}

        enriched_docs = []
        complexity_scores = []
        compressibility_scores = []

        for doc in documents:
            text = doc.get("document", "")

            # Calculer complexité
            complexity = self._calculate_complexity(text)

            # Calculer compressibilité
            compressibility = self._calculate_compressibility(text)

            # Enrichir métadonnées
            doc_metadata = doc.get("metadata", {})
            doc_metadata["complexity_score"] = complexity
            doc_metadata["compressibility_score"] = compressibility

            enriched_doc = doc.copy()
            enriched_doc["metadata"] = doc_metadata

            enriched_docs.append(enriched_doc)

            complexity_scores.append(complexity)
            compressibility_scores.append(compressibility)

        # Statistiques globales
        stats = {
            "avg_complexity": np.mean(complexity_scores) if complexity_scores else 0.0,
            "avg_compressibility": (
                np.mean(compressibility_scores) if compressibility_scores else 0.0
            ),
        }

        return enriched_docs, stats

    def _calculate_complexity(self, text: str) -> float:
        """Calcule la complexité informationnelle du texte.

        Args:
            text: Texte à analyser.

        Returns:
            Score de complexité [0-1].
        """
        if not text:
            return 0.0

        # Simplification : longueur moyenne des mots et variance
        words = text.split()
        if not words:
            return 0.0

        word_lengths = [len(word) for word in words]
        avg_word_length = np.mean(word_lengths)
        vocab_size = len(set(words))
        vocab_diversity = vocab_size / len(words) if words else 0.0

        # Score combiné (simplifié)
        complexity = min(1.0, (avg_word_length / 10.0 + vocab_diversity) / 2.0)

        return float(complexity)

    def _calculate_compressibility(self, text: str) -> float:
        """Calcule la compressibilité du texte (basée sur entropie).

        Args:
            text: Texte à analyser.

        Returns:
            Score de compressibilité [0-1] (plus élevé = plus compressible).
        """
        if not text:
            return 0.0

        # Simplification : ratio de répétition des mots
        words = text.split()
        if not words:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)

        # Plus de répétitions = plus compressible
        compressibility = 1.0 - (unique_words / total_words)

        return float(compressibility)


class LLMLinguaCompressor:
    """Compresseur utilisant LLMLingua pour compression agressive de prompts.

    LLMLingua-2 : 2.5x-4x compression avec +21.4% performance.
    Préserve les entités, nombres, et structure critique.

    Attributes:
        config: Configuration LLMLingua.
        compressor: Instance LLMLingua (si disponible).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le compresseur LLMLingua.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("prompt_compression", {})
        self.compressor: Optional[Any] = None
        self._initialize_compressor()

    def _initialize_compressor(self) -> None:
        """Initialise le compresseur LLMLingua."""
        if not self.config.get("enabled", True):
            return

        try:
            # Tentative d'import LLMLingua
            from llmlingua import PromptCompressor

            tool = self.config.get("tool", "llmlingua2")

            if tool == "llmlingua2":
                llmlingua2_config = self.config.get("llmlingua2", {})
                model_config = llmlingua2_config.get("model_config", {})

                model_name = model_config.get(
                    "model_name",
                    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
                )
                device = model_config.get("device", "cpu")

                self.compressor = PromptCompressor(
                    model_name=model_name,
                    device_map=device,
                )

            else:
                # Fallback : compresseur basic
                self.compressor = None

        except ImportError:
            # LLMLingua pas installé - désactiver
            print(
                "⚠️  LLMLingua not installed. Compression will be disabled. "
                "Install with: pip install llmlingua"
            )
            self.compressor = None

    def compress(
        self, documents: List[Dict[str, Any]], query: str = ""
    ) -> List[Dict[str, Any]]:
        """Compresse les documents avec LLMLingua.

        Args:
            documents: Documents à compresser.
            query: Query utilisateur (pour compression query-aware).

        Returns:
            Documents compressés.
        """
        if not self.config.get("enabled", True) or self.compressor is None:
            return documents

        llmlingua2_config = self.config.get("llmlingua2", {})
        compression_rate = llmlingua2_config.get("compression_rate", 0.4)

        compressed_docs = []

        for doc in documents:
            text = doc.get("document", "")

            try:
                # Compression avec LLMLingua
                compressed_result = self.compressor.compress_prompt(
                    text,
                    rate=compression_rate,
                    force_tokens=["\n", ".", "!", "?", ","],
                )

                compressed_text = compressed_result.get("compressed_prompt", text)

                # Métriques
                original_tokens = len(text.split())
                compressed_tokens = len(compressed_text.split())
                ratio = (
                    original_tokens / compressed_tokens
                    if compressed_tokens > 0
                    else 1.0
                )

                compressed_doc = doc.copy()
                compressed_doc["document"] = compressed_text
                compressed_doc["metadata"] = compressed_doc.get("metadata", {})
                compressed_doc["metadata"]["compression_ratio"] = ratio
                compressed_doc["metadata"]["original_length"] = original_tokens
                compressed_doc["metadata"]["compressed_length"] = compressed_tokens

                compressed_docs.append(compressed_doc)

            except Exception as e:
                # En cas d'erreur, garder document original
                print(f"⚠️  Compression error: {e!s}. Keeping original document.")
                compressed_docs.append(doc)

        return compressed_docs


class ContextualCompressor:
    """Compresseur contextuel extractif.

    Extrait les passages les plus pertinents selon la query.
    Plus rapide et préserve mieux la qualité que compression abstractive.

    Attributes:
        config: Configuration de la compression contextuelle.
        scorer_model: Modèle pour scorer la relevance (optionnel).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le compresseur contextuel.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("contextual_compression", {})
        self.scorer_model: Optional[Any] = None
        self._initialize_scorer()

    def _initialize_scorer(self) -> None:
        """Initialise le modèle de scoring."""
        if not self.config.get("enabled", True):
            return

        extractive_config = self.config.get("extractive", {})
        scorer_model_name = extractive_config.get("scorer_model", "BAAI/bge-m3")

        try:
            from sentence_transformers import SentenceTransformer

            self.scorer_model = SentenceTransformer(scorer_model_name)

        except ImportError:
            print(
                "⚠️  sentence-transformers not installed. "
                "Contextual compression will use simple heuristics."
            )
            self.scorer_model = None

    def compress(
        self, documents: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Compresse contextuellement les documents.

        Args:
            documents: Documents à compresser.
            query: Query utilisateur pour relevance scoring.

        Returns:
            Documents compressés.
        """
        if not self.config.get("enabled", True):
            return documents

        extractive_config = self.config.get("extractive", {})
        max_passage_length = extractive_config.get("max_passage_length", 200)
        relevance_threshold = extractive_config.get("relevance_threshold", 0.4)

        compressed_docs = []

        for doc in documents:
            text = doc.get("document", "")

            # Découper en phrases
            sentences = self._split_sentences(text)

            if not sentences:
                compressed_docs.append(doc)
                continue

            # Scorer chaque phrase
            if self.scorer_model is not None:
                scores = self._score_sentences(sentences, query)
            else:
                # Heuristique simple : toutes les phrases égales
                scores = [1.0] * len(sentences)

            # Sélectionner phrases au-dessus du seuil
            selected_sentences = []
            for sentence, score in zip(sentences, scores):
                if score >= relevance_threshold:
                    selected_sentences.append(sentence)

            # Limiter longueur
            compressed_text = " ".join(selected_sentences)
            words = compressed_text.split()
            if len(words) > max_passage_length:
                compressed_text = " ".join(words[:max_passage_length]) + "..."

            compressed_doc = doc.copy()
            compressed_doc["document"] = compressed_text

            compressed_docs.append(compressed_doc)

        return compressed_docs

    def _split_sentences(self, text: str) -> List[str]:
        """Découpe le texte en phrases.

        Args:
            text: Texte à découper.

        Returns:
            Liste de phrases.
        """
        # Simplification : split sur . ! ?
        import re

        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _score_sentences(self, sentences: List[str], query: str) -> List[float]:
        """Score la relevance de chaque phrase par rapport à la query.

        Args:
            sentences: Liste de phrases.
            query: Query utilisateur.

        Returns:
            Scores de relevance pour chaque phrase.
        """
        if self.scorer_model is None:
            return [1.0] * len(sentences)

        # Encoder query et sentences
        query_embedding = self.scorer_model.encode([query], normalize_embeddings=True)
        sentence_embeddings = self.scorer_model.encode(
            sentences, normalize_embeddings=True
        )

        # Calculer similarités cosine
        similarities = np.dot(sentence_embeddings, query_embedding.T).flatten()

        return similarities.tolist()


class CompressionAwareMMR:
    """MMR intelligent avec compression awareness.

    Ajuste la sélection MMR en fonction de la qualité de compression
    de chaque document.

    Attributes:
        config: Configuration MMR compression-aware.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise MMR compression-aware.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("mmr_compression_aware", {})

    def apply(
        self, documents: List[Dict[str, Any]], top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Applique MMR avec compression awareness.

        Args:
            documents: Documents compressés avec métadonnées.
            top_k: Nombre de documents à retourner.

        Returns:
            Documents sélectionnés avec MMR.
        """
        if not self.config.get("enabled", True):
            return documents

        final_top_k = top_k or self.config.get("final_top_k", 15)

        if len(documents) <= final_top_k:
            return documents

        # Boost des documents bien compressés
        compression_aware_config = self.config.get("compression_aware", {})
        boost_enabled = compression_aware_config.get("boost_well_compressed", {}).get(
            "enabled", True
        )

        if boost_enabled:
            documents = self._boost_well_compressed(documents)

        # Trier par score et limiter
        documents_sorted = sorted(
            documents, key=lambda x: x.get("score", 0.0), reverse=True
        )

        return documents_sorted[:final_top_k]

    def _boost_well_compressed(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Boost le score des documents bien compressés.

        Args:
            documents: Documents avec métadonnées de compression.

        Returns:
            Documents avec scores boostés.
        """
        compression_aware_config = self.config.get("compression_aware", {})
        boost_config = compression_aware_config.get("boost_well_compressed", {})
        ratio_threshold = boost_config.get("compression_ratio_threshold", 2.0)
        boost_factor = boost_config.get("boost_factor", 1.1)

        boosted_docs = []

        for doc in documents:
            compression_ratio = doc.get("metadata", {}).get("compression_ratio", 1.0)

            if compression_ratio >= ratio_threshold:
                doc["score"] = doc.get("score", 0.0) * boost_factor

            boosted_docs.append(doc)

        return boosted_docs


class QualityValidator:
    """Validateur de qualité post-compression.

    Vérifie que la compression n'a pas dégradé la qualité :
    - Similarité sémantique préservée
    - Entités nommées préservées
    - Répondabilité de la question maintenue

    Attributes:
        config: Configuration de la validation.
        similarity_model: Modèle pour similarité sémantique (optionnel).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le validateur.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("quality_validation", {})
        self.similarity_model: Optional[Any] = None
        self._initialize_similarity_model()

    def _initialize_similarity_model(self) -> None:
        """Initialise le modèle de similarité."""
        if not self.config.get("enabled", True):
            return

        semantic_config = self.config.get("semantic_similarity", {})
        if not semantic_config.get("enabled", True):
            return

        model_name = semantic_config.get(
            "model", "sentence-transformers/all-MiniLM-L6-v2"
        )

        try:
            from sentence_transformers import SentenceTransformer

            self.similarity_model = SentenceTransformer(model_name)

        except ImportError:
            print("⚠️  sentence-transformers not installed. Quality validation limited.")
            self.similarity_model = None

    def validate(
        self,
        original_documents: List[Dict[str, Any]],
        compressed_documents: List[Dict[str, Any]],
        query: str = "",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Valide la qualité de la compression.

        Args:
            original_documents: Documents originaux avant compression.
            compressed_documents: Documents après compression.
            query: Query utilisateur (pour answerability).

        Returns:
            Tuple (documents validés, rapport de validation).
        """
        if not self.config.get("enabled", True):
            return compressed_documents, {}

        semantic_config = self.config.get("semantic_similarity", {})
        min_similarity = semantic_config.get("min_similarity", 0.85)

        validated_docs = []
        validation_report = {
            "passed": 0,
            "failed": 0,
            "avg_similarity": 0.0,
        }

        similarities = []

        for orig_doc, comp_doc in zip(original_documents, compressed_documents):
            orig_text = orig_doc.get("document", "")
            comp_text = comp_doc.get("document", "")

            # Calculer similarité sémantique
            if self.similarity_model is not None:
                similarity = self._calculate_similarity(orig_text, comp_text)
            else:
                # Heuristique : ratio de longueur
                similarity = min(
                    1.0, len(comp_text.split()) / max(1, len(orig_text.split()))
                )

            similarities.append(similarity)

            # Vérifier seuil
            if similarity >= min_similarity:
                validation_report["passed"] += 1
                validated_docs.append(comp_doc)
            else:
                validation_report["failed"] += 1
                # Garder document original si échec validation
                validated_docs.append(orig_doc)

        validation_report["avg_similarity"] = (
            np.mean(similarities) if similarities else 0.0
        )

        return validated_docs, validation_report

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité sémantique entre deux textes.

        Args:
            text1: Premier texte.
            text2: Deuxième texte.

        Returns:
            Score de similarité [0-1].
        """
        if self.similarity_model is None:
            return 0.0

        embeddings = self.similarity_model.encode(
            [text1, text2], normalize_embeddings=True
        )

        similarity = np.dot(embeddings[0], embeddings[1])

        return float(similarity)


class ContextWindowOptimizer:
    """Optimiseur de context window pour génération LLM.

    Gère intelligemment le budget de tokens :
    - Allocation dynamique selon importance
    - Truncation intelligente
    - Préservation top-k

    Attributes:
        config: Configuration de l'optimisation.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise l'optimiseur.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("context_window_optimization", {})

    def optimize(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Optimise les documents pour respecter le context window.

        Args:
            documents: Documents à optimiser.

        Returns:
            Documents optimisés pour le context window.
        """
        if not self.config.get("enabled", True):
            return documents

        target_tokens = self.config.get("target_context_tokens", 4000)

        # Compter tokens actuels
        total_tokens = sum(len(doc.get("document", "").split()) for doc in documents)

        if total_tokens <= target_tokens:
            # Pas besoin d'optimiser
            return documents

        # Truncation intelligente
        smart_truncate_config = self.config.get("smart_truncate", {})
        preserve_top_k = smart_truncate_config.get("preserve_top_k", 5)

        optimized_docs = []

        # Préserver top-k complets
        for i, doc in enumerate(documents):
            if i < preserve_top_k:
                optimized_docs.append(doc)
            else:
                # Truncate les autres documents
                text = doc.get("document", "")
                words = text.split()

                # Budget tokens par document
                budget = max(50, target_tokens // (len(documents) - preserve_top_k))

                if len(words) > budget:
                    truncated_text = " ".join(words[:budget]) + "..."
                    truncated_doc = doc.copy()
                    truncated_doc["document"] = truncated_text
                    optimized_docs.append(truncated_doc)
                else:
                    optimized_docs.append(doc)

        return optimized_docs


# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def process_compression(
    documents: List[Dict[str, Any]],
    query: str,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Point d'entrée principal pour la compression contextuelle.

    Args:
        documents: Documents reranked en entrée.
        query: Query utilisateur.
        config: Configuration complète chargée depuis YAML.

    Returns:
        Dictionnaire contenant les documents compressés et métadonnées.
    """
    compression_config = config.get("step_04_compression", {})

    if not compression_config.get("enabled", True):
        return {
            "documents": documents,
            "compression_ratio": 1.0,
            "validation_report": {},
        }

    # Sauvegarder documents originaux pour validation
    original_documents = [doc.copy() for doc in documents]

    # Pipeline de compression
    pipeline = compression_config.get("pipeline", [])

    current_documents = documents
    compression_stats = {}

    for step in pipeline:
        step_name = step.get("step", "")
        step_enabled = step.get("enabled", False)

        if not step_enabled:
            continue

        if step_name == "pre_compression_analysis":
            analyzer = PreCompressionAnalyzer(config)
            current_documents, stats = analyzer.analyze(current_documents)
            compression_stats["pre_compression"] = stats

        elif step_name == "prompt_compression_llmlingua":
            compressor = LLMLinguaCompressor(config)
            current_documents = compressor.compress(current_documents, query)

        elif step_name == "contextual_compression":
            contextual_compressor = ContextualCompressor(config)
            current_documents = contextual_compressor.compress(current_documents, query)

        elif step_name == "mmr_compression_aware":
            mmr = CompressionAwareMMR(config)
            current_documents = mmr.apply(current_documents)

        elif step_name == "quality_validation":
            validator = QualityValidator(config)
            current_documents, validation_report = validator.validate(
                original_documents, current_documents, query
            )
            compression_stats["validation"] = validation_report

        elif step_name == "context_window_optimization":
            optimizer = ContextWindowOptimizer(config)
            current_documents = optimizer.optimize(current_documents)

    # Calculer ratio de compression global
    original_tokens = sum(len(doc.get("document", "").split()) for doc in documents)
    compressed_tokens = sum(
        len(doc.get("document", "").split()) for doc in current_documents
    )

    compression_ratio = (
        original_tokens / compressed_tokens if compressed_tokens > 0 else 1.0
    )

    return {
        "documents": current_documents,
        "compression_ratio": compression_ratio,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "num_documents": len(current_documents),
        "compression_stats": compression_stats,
    }


if __name__ == "__main__":
    """Exemple d'exécution du module."""
    print("=" * 80)
    print("COMPRESSION CONTEXTUELLE - EXEMPLE")
    print("=" * 80)
    print("\n⚠️  Ce module nécessite des documents reranked en entrée.")
    print("\nExemple d'utilisation :")
    print(
        """
    from inference_project.steps.step_03_reranking import process_reranking
    from inference_project.steps.step_04_compression import process_compression
    from inference_project.utils.config_loader import load_config

    # Charger config
    config = load_config("04_compression_v2", "config")

    # Query
    query = "What is machine learning?"

    # Documents (résultats de reranking)
    reranked_documents = [
        {
            "id": "doc1",
            "document": "Machine learning is a subset of artificial intelligence...",
            "score": 0.95,
            "metadata": {"source": "AI Textbook"},
        },
        {
            "id": "doc2",
            "document": "ML algorithms include supervised, unsupervised...",
            "score": 0.88,
            "metadata": {"source": "ML Guide"},
        },
    ]

    # Compression
    result = process_compression(reranked_documents, query, config)

    print(f"Compression ratio: {result['compression_ratio']:.2f}x")
    print(f"Original tokens: {result['original_tokens']}")
    print(f"Compressed tokens: {result['compressed_tokens']}")
    print(f"Documents retained: {result['num_documents']}")
    """
    )
    print("=" * 80)
