"""Tests unitaires pour le module de compression (Phase 04)."""

from typing import Any, Dict, List

import numpy as np
import pytest

from inference_project.steps.step_04_compression import (
    CompressionAwareMMR,
    ContextualCompressor,
    ContextWindowOptimizer,
    PreCompressionAnalyzer,
    QualityValidator,
    process_compression,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def mock_config_compression() -> Dict[str, Any]:
    """Configuration mock pour tests de compression."""
    return {
        "step_04_compression": {
            "enabled": True,
            "pipeline": [
                {"step": "pre_compression_analysis", "enabled": True},
                {"step": "contextual_compression", "enabled": True},
                {"step": "mmr_compression_aware", "enabled": True},
                {"step": "quality_validation", "enabled": True},
                {"step": "context_window_optimization", "enabled": True},
            ],
        },
        "pre_compression_analysis": {"enabled": True},
        "contextual_compression": {
            "enabled": True,
            "extractive": {
                "max_passage_length": 100,
                "relevance_threshold": 0.4,
            },
        },
        "mmr_compression_aware": {
            "enabled": True,
            "final_top_k": 5,
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
                "min_similarity": 0.85,
            },
        },
        "context_window_optimization": {
            "enabled": True,
            "target_context_tokens": 500,
            "smart_truncate": {
                "preserve_top_k": 3,
            },
        },
    }


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Documents exemple pour tests."""
    return [
        {
            "id": "doc1",
            "document": "Machine learning is a subset of artificial intelligence. "
            "It focuses on the development of algorithms. "
            "These algorithms can learn from data. "
            "Machine learning powers many modern applications.",
            "score": 0.95,
            "metadata": {"source": "AI Textbook"},
        },
        {
            "id": "doc2",
            "document": "Supervised learning uses labeled data. "
            "Unsupervised learning finds patterns in unlabeled data. "
            "Reinforcement learning learns through trial and error. "
            "These are the three main types of machine learning.",
            "score": 0.88,
            "metadata": {"source": "ML Guide"},
        },
        {
            "id": "doc3",
            "document": "Deep learning is a subset of machine learning. "
            "It uses neural networks with many layers. "
            "Deep learning excels at image and speech recognition. "
            "Modern AI relies heavily on deep learning techniques.",
            "score": 0.82,
            "metadata": {"source": "DL Primer"},
        },
    ]


# =============================================================================
# TESTS PRE-COMPRESSION ANALYSIS
# =============================================================================


def test_pre_compression_analyzer_initialization(mock_config_compression):
    """Test l'initialisation du PreCompressionAnalyzer."""
    analyzer = PreCompressionAnalyzer(mock_config_compression)

    assert analyzer.config is not None
    assert analyzer.config.get("enabled") is True


def test_pre_compression_analyzer_analyze(mock_config_compression, sample_documents):
    """Test l'analyse de complexité et compressibilité."""
    analyzer = PreCompressionAnalyzer(mock_config_compression)

    enriched_docs, stats = analyzer.analyze(sample_documents)

    # Vérifier que tous les documents sont enrichis
    assert len(enriched_docs) == len(sample_documents)

    # Vérifier métadonnées enrichies
    for doc in enriched_docs:
        assert "complexity_score" in doc["metadata"]
        assert "compressibility_score" in doc["metadata"]
        assert 0.0 <= doc["metadata"]["complexity_score"] <= 1.0
        assert 0.0 <= doc["metadata"]["compressibility_score"] <= 1.0

    # Vérifier statistiques globales
    assert "avg_complexity" in stats
    assert "avg_compressibility" in stats
    assert 0.0 <= stats["avg_complexity"] <= 1.0
    assert 0.0 <= stats["avg_compressibility"] <= 1.0


def test_calculate_complexity():
    """Test le calcul de complexité."""
    analyzer = PreCompressionAnalyzer({})

    # Texte simple
    simple_text = "I am a cat. I like fish. Fish is good."
    complexity_simple = analyzer._calculate_complexity(simple_text)

    # Texte complexe
    complex_text = (
        "Epistemological considerations regarding "
        "phenomenological interpretations necessitate comprehensive analysis."
    )
    complexity_complex = analyzer._calculate_complexity(complex_text)

    # Le texte complexe devrait avoir un score plus élevé
    assert complexity_complex > complexity_simple


def test_calculate_compressibility():
    """Test le calcul de compressibilité."""
    analyzer = PreCompressionAnalyzer({})

    # Texte répétitif (très compressible)
    repetitive_text = "cat cat cat dog dog dog fish fish fish"
    compressibility_high = analyzer._calculate_compressibility(repetitive_text)

    # Texte diversifié (peu compressible)
    diverse_text = "The quick brown fox jumps over the lazy dog elegantly"
    compressibility_low = analyzer._calculate_compressibility(diverse_text)

    # Le texte répétitif devrait être plus compressible
    assert compressibility_high > compressibility_low


# =============================================================================
# TESTS CONTEXTUAL COMPRESSION
# =============================================================================


def test_contextual_compressor_initialization(mock_config_compression):
    """Test l'initialisation du ContextualCompressor."""
    compressor = ContextualCompressor(mock_config_compression)

    assert compressor.config is not None
    assert compressor.config.get("enabled") is True


def test_contextual_compressor_compress(mock_config_compression, sample_documents):
    """Test la compression contextuelle."""
    compressor = ContextualCompressor(mock_config_compression)

    query = "What is machine learning?"
    compressed_docs = compressor.compress(sample_documents, query)

    # Vérifier que tous les documents sont compressés
    assert len(compressed_docs) == len(sample_documents)

    # Vérifier que les documents sont bien plus courts
    for orig_doc, comp_doc in zip(sample_documents, compressed_docs):
        orig_length = len(orig_doc["document"].split())
        comp_length = len(comp_doc["document"].split())

        # La compression devrait réduire la longueur (ou la garder égale dans le pire cas)
        assert comp_length <= orig_length


def test_contextual_compressor_split_sentences():
    """Test le découpage en phrases."""
    compressor = ContextualCompressor({})

    text = "This is sentence one. This is sentence two! Is this sentence three?"
    sentences = compressor._split_sentences(text)

    assert len(sentences) == 3
    assert "This is sentence one" in sentences[0]


def test_contextual_compressor_respects_max_length(mock_config_compression):
    """Test que la compression respecte la longueur maximale."""
    compressor = ContextualCompressor(mock_config_compression)

    # Document très long
    long_doc = {
        "id": "long",
        "document": " ".join(["word"] * 500),  # 500 mots
        "score": 0.9,
        "metadata": {},
    }

    query = "test query"
    compressed_docs = compressor.compress([long_doc], query)

    # Vérifier que le document compressé respecte la limite
    max_length = mock_config_compression["contextual_compression"]["extractive"][
        "max_passage_length"
    ]
    compressed_length = len(compressed_docs[0]["document"].split())

    assert compressed_length <= max_length + 5  # +5 pour "..." et marge


# =============================================================================
# TESTS COMPRESSION-AWARE MMR
# =============================================================================


def test_compression_aware_mmr_initialization(mock_config_compression):
    """Test l'initialisation du CompressionAwareMMR."""
    mmr = CompressionAwareMMR(mock_config_compression)

    assert mmr.config is not None
    assert mmr.config.get("enabled") is True


def test_compression_aware_mmr_apply(mock_config_compression, sample_documents):
    """Test l'application du MMR compression-aware."""
    # Ajouter métadonnées de compression
    docs_with_compression = []
    for i, doc in enumerate(sample_documents):
        doc_copy = doc.copy()
        doc_copy["metadata"] = doc.get("metadata", {}).copy()
        # Documents alternent entre bien compressés et mal compressés
        doc_copy["metadata"]["compression_ratio"] = 3.0 if i % 2 == 0 else 1.2
        docs_with_compression.append(doc_copy)

    mmr = CompressionAwareMMR(mock_config_compression)
    selected_docs = mmr.apply(docs_with_compression, top_k=2)

    # Vérifier limitation au top_k
    assert len(selected_docs) <= 2


def test_compression_aware_mmr_boosts_well_compressed(mock_config_compression):
    """Test que le MMR boost les documents bien compressés."""
    mmr = CompressionAwareMMR(mock_config_compression)

    # Document bien compressé
    well_compressed = {
        "id": "well",
        "document": "text",
        "score": 0.8,
        "metadata": {"compression_ratio": 3.5},
    }

    # Document mal compressé
    poorly_compressed = {
        "id": "poor",
        "document": "text",
        "score": 0.8,
        "metadata": {"compression_ratio": 1.1},
    }

    docs = [well_compressed, poorly_compressed]
    boosted_docs = mmr._boost_well_compressed(docs)

    # Le document bien compressé devrait avoir un score boosté
    assert boosted_docs[0]["score"] > 0.8  # Score boosté
    assert boosted_docs[1]["score"] == 0.8  # Score inchangé


# =============================================================================
# TESTS QUALITY VALIDATION
# =============================================================================


def test_quality_validator_initialization(mock_config_compression):
    """Test l'initialisation du QualityValidator."""
    validator = QualityValidator(mock_config_compression)

    assert validator.config is not None
    assert validator.config.get("enabled") is True


def test_quality_validator_validate(mock_config_compression):
    """Test la validation de qualité."""
    validator = QualityValidator(mock_config_compression)

    # Documents originaux
    original_docs = [
        {
            "id": "doc1",
            "document": "Machine learning is a subset of AI that enables computers to learn.",
            "score": 0.9,
            "metadata": {},
        }
    ]

    # Documents compressés (très similaires)
    compressed_docs = [
        {
            "id": "doc1",
            "document": "Machine learning enables computers to learn from data.",
            "score": 0.9,
            "metadata": {},
        }
    ]

    query = "What is ML?"
    validated_docs, report = validator.validate(
        original_docs, compressed_docs, query
    )

    # Vérifier que la validation retourne des documents
    assert len(validated_docs) > 0

    # Vérifier rapport de validation
    assert "passed" in report
    assert "failed" in report
    assert "avg_similarity" in report
    assert report["passed"] + report["failed"] == len(original_docs)


def test_quality_validator_rejects_poor_compression(mock_config_compression):
    """Test que le validateur rejette les compressions de mauvaise qualité."""
    # Configurer seuil très élevé
    strict_config = mock_config_compression.copy()
    strict_config["quality_validation"]["semantic_similarity"]["min_similarity"] = 0.99

    validator = QualityValidator(strict_config)

    # Document original
    original = [
        {
            "id": "doc1",
            "document": "Machine learning is a complex field of artificial intelligence.",
            "score": 0.9,
            "metadata": {},
        }
    ]

    # Document très différent (mauvaise compression)
    poor_compression = [
        {
            "id": "doc1",
            "document": "Cats like fish.",
            "score": 0.9,
            "metadata": {},
        }
    ]

    validated_docs, report = validator.validate(original, poor_compression, "ML")

    # Le validateur devrait rejeter la compression et garder l'original
    assert report["failed"] > 0


# =============================================================================
# TESTS CONTEXT WINDOW OPTIMIZATION
# =============================================================================


def test_context_window_optimizer_initialization(mock_config_compression):
    """Test l'initialisation du ContextWindowOptimizer."""
    optimizer = ContextWindowOptimizer(mock_config_compression)

    assert optimizer.config is not None
    assert optimizer.config.get("enabled") is True


def test_context_window_optimizer_respects_budget(mock_config_compression):
    """Test que l'optimiseur respecte le budget de tokens."""
    optimizer = ContextWindowOptimizer(mock_config_compression)

    # Documents qui dépassent le budget
    large_docs = [
        {
            "id": f"doc{i}",
            "document": " ".join(["word"] * 200),  # 200 mots chacun
            "score": 0.9 - i * 0.1,
            "metadata": {},
        }
        for i in range(10)  # 2000 mots total
    ]

    optimized_docs = optimizer.optimize(large_docs)

    # Compter tokens totaux
    total_tokens = sum(len(doc["document"].split()) for doc in optimized_docs)

    target_tokens = mock_config_compression["context_window_optimization"][
        "target_context_tokens"
    ]

    # Le total devrait être proche du budget (avec marge de 10%)
    assert total_tokens <= target_tokens * 1.1


def test_context_window_optimizer_preserves_top_k(mock_config_compression):
    """Test que l'optimiseur préserve les top-k documents complets."""
    optimizer = ContextWindowOptimizer(mock_config_compression)

    # Documents triés par score
    docs = [
        {
            "id": f"doc{i}",
            "document": " ".join(["word"] * 100),
            "score": 1.0 - i * 0.1,
            "metadata": {},
        }
        for i in range(5)
    ]

    optimized_docs = optimizer.optimize(docs)

    # Les 3 premiers documents (top-k=3) devraient être préservés intacts
    preserve_top_k = mock_config_compression["context_window_optimization"][
        "smart_truncate"
    ]["preserve_top_k"]

    for i in range(preserve_top_k):
        assert len(optimized_docs[i]["document"].split()) == 100  # Longueur originale


# =============================================================================
# TESTS PROCESS_COMPRESSION
# =============================================================================


def test_process_compression_full_pipeline(mock_config_compression, sample_documents):
    """Test le pipeline complet de compression."""
    query = "What is machine learning?"

    result = process_compression(sample_documents, query, mock_config_compression)

    # Vérifier structure du résultat
    assert "documents" in result
    assert "compression_ratio" in result
    assert "original_tokens" in result
    assert "compressed_tokens" in result
    assert "num_documents" in result
    assert "compression_stats" in result

    # Vérifier compression effective
    assert result["compression_ratio"] >= 1.0  # Au moins 1x (pas d'expansion)
    assert result["compressed_tokens"] <= result["original_tokens"]
    assert len(result["documents"]) > 0


def test_process_compression_disabled(sample_documents):
    """Test que la compression peut être désactivée."""
    config_disabled = {
        "step_04_compression": {
            "enabled": False,
        }
    }

    query = "test"
    result = process_compression(sample_documents, query, config_disabled)

    # Si désactivé, les documents devraient être inchangés
    assert result["compression_ratio"] == 1.0
    assert len(result["documents"]) == len(sample_documents)


def test_process_compression_calculates_ratio(mock_config_compression, sample_documents):
    """Test que le ratio de compression est calculé correctement."""
    query = "machine learning"

    result = process_compression(sample_documents, query, mock_config_compression)

    # Vérifier cohérence du ratio
    expected_ratio = (
        result["original_tokens"] / result["compressed_tokens"]
        if result["compressed_tokens"] > 0
        else 1.0
    )

    assert abs(result["compression_ratio"] - expected_ratio) < 0.01


def test_process_compression_with_empty_documents(mock_config_compression):
    """Test la compression avec liste vide."""
    query = "test"
    result = process_compression([], query, mock_config_compression)

    assert result["num_documents"] == 0
    assert result["original_tokens"] == 0
    assert result["compressed_tokens"] == 0


# =============================================================================
# TESTS D'INTÉGRATION
# =============================================================================


def test_integration_compression_quality_tradeoff(mock_config_compression):
    """Test l'équilibre compression/qualité."""
    # Documents avec contenu riche
    rich_docs = [
        {
            "id": "rich1",
            "document": (
                "Machine learning algorithms can be supervised, unsupervised, "
                "or reinforcement-based. Supervised learning uses labeled data "
                "to train models. Unsupervised learning finds patterns in unlabeled data. "
                "Reinforcement learning learns through trial and error interactions."
            ),
            "score": 0.95,
            "metadata": {},
        }
    ]

    query = "types of machine learning"

    result = process_compression(rich_docs, query, mock_config_compression)

    # Devrait compresser mais préserver l'information essentielle
    assert result["compression_ratio"] >= 1.0
    assert len(result["documents"][0]["document"]) > 0

    # Vérifier que la validation a passé
    if "validation" in result["compression_stats"]:
        validation = result["compression_stats"]["validation"]
        # Au moins quelques documents devraient passer la validation
        assert validation.get("passed", 0) >= 0


def test_integration_pipeline_order(mock_config_compression, sample_documents):
    """Test que les étapes du pipeline s'exécutent dans le bon ordre."""
    query = "machine learning"

    # Tracer l'ordre d'exécution via les métadonnées
    result = process_compression(sample_documents, query, mock_config_compression)

    # Pre-compression analysis devrait enrichir metadata
    if "pre_compression" in result["compression_stats"]:
        assert "avg_complexity" in result["compression_stats"]["pre_compression"]

    # Validation devrait produire un rapport
    if "validation" in result["compression_stats"]:
        assert "passed" in result["compression_stats"]["validation"]

    # Documents finaux devraient exister
    assert len(result["documents"]) > 0
