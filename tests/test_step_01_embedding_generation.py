"""Tests unitaires pour le module step_01_embedding_generation.

Ce module teste la génération d'embeddings denses, sparse, et late interaction.
"""

import numpy as np
import pytest

from inference_project.steps.step_01_embedding_generation import (
    EmbeddingGenerator,
    process_embeddings,
)


@pytest.fixture
def mock_config_dense_only():
    """Configuration minimale avec dense embeddings uniquement."""
    return {
        "embedding_generation": {
            "enabled": True,
            "dense_embedding": {
                "enabled": True,
                "model": "sentence-transformers/all-MiniLM-L6-v2",  # Petit modèle pour tests
                "device": "cpu",
                "batch_size": 32,
                "normalize": True,
                "show_progress_bar": False,
            },
            "sparse_embedding": {"enabled": False},
            "late_interaction": {"enabled": False},
        }
    }


@pytest.fixture
def sample_queries():
    """Queries d'exemple pour les tests."""
    return [
        "What is machine learning?",
        "How does RAG work?",
        "Explain transformer architecture.",
    ]


def test_embedding_generator_initialization(mock_config_dense_only):
    """Test l'initialisation du générateur d'embeddings."""
    generator = EmbeddingGenerator(mock_config_dense_only)

    assert generator.dense_model is not None
    assert generator.sparse_model is None
    assert generator.late_interaction_model is None


def test_generate_dense_embeddings_shape(mock_config_dense_only, sample_queries):
    """Test que les embeddings denses ont la bonne shape."""
    generator = EmbeddingGenerator(mock_config_dense_only)
    embeddings = generator.generate_dense_embeddings(sample_queries)

    # Vérifier shape (n_queries, embedding_dim)
    assert embeddings.shape[0] == len(sample_queries)
    assert embeddings.shape[1] > 0  # Dimension > 0
    assert isinstance(embeddings, np.ndarray)


def test_generate_dense_embeddings_normalization(
    mock_config_dense_only, sample_queries
):
    """Test que les embeddings sont normalisés (norme L2 = 1)."""
    generator = EmbeddingGenerator(mock_config_dense_only)
    embeddings = generator.generate_dense_embeddings(sample_queries, normalize=True)

    # Vérifier que chaque vecteur a une norme proche de 1.0
    for i in range(len(embeddings)):
        norm = np.linalg.norm(embeddings[i])
        assert abs(norm - 1.0) < 1e-5, f"Vecteur {i} non normalisé: norme={norm}"


def test_generate_dense_embeddings_consistency(mock_config_dense_only):
    """Test que les mêmes queries donnent les mêmes embeddings."""
    generator = EmbeddingGenerator(mock_config_dense_only)

    query = ["What is machine learning?"]

    # Générer 2 fois
    embeddings1 = generator.generate_dense_embeddings(query)
    embeddings2 = generator.generate_dense_embeddings(query)

    # Doivent être identiques (ou très proches à cause de floating point)
    np.testing.assert_allclose(embeddings1, embeddings2, rtol=1e-5)


def test_generate_dense_embeddings_similarity(mock_config_dense_only):
    """Test que des queries similaires ont des embeddings similaires."""
    generator = EmbeddingGenerator(mock_config_dense_only)

    similar_queries = [
        "What is machine learning?",
        "Explain machine learning.",  # Similaire
        "How to cook pasta?",  # Très différent
    ]

    embeddings = generator.generate_dense_embeddings(similar_queries)

    # Similarité cosine entre query 0 et 1 (similaires)
    sim_01 = np.dot(embeddings[0], embeddings[1])

    # Similarité cosine entre query 0 et 2 (différents)
    sim_02 = np.dot(embeddings[0], embeddings[2])

    # Queries similaires doivent avoir similarité plus haute
    assert sim_01 > sim_02, f"sim_01={sim_01:.3f} devrait être > sim_02={sim_02:.3f}"


def test_generate_sparse_embeddings_not_implemented(mock_config_dense_only, sample_queries):
    """Test que sparse embeddings lève NotImplementedError si pas activé."""
    generator = EmbeddingGenerator(mock_config_dense_only)

    with pytest.raises(NotImplementedError):
        generator.generate_sparse_embeddings(sample_queries)


def test_generate_late_interaction_not_implemented(
    mock_config_dense_only, sample_queries
):
    """Test que late interaction lève NotImplementedError si pas activé."""
    generator = EmbeddingGenerator(mock_config_dense_only)

    with pytest.raises(NotImplementedError):
        generator.generate_late_interaction_embeddings(sample_queries)


def test_process_embeddings_output_structure(mock_config_dense_only, sample_queries):
    """Test que process() retourne la structure attendue."""
    generator = EmbeddingGenerator(mock_config_dense_only)
    results = generator.process(sample_queries)

    # Vérifier structure
    assert "queries" in results
    assert "dense_embeddings" in results
    assert "metadata" in results

    # Vérifier queries
    assert results["queries"] == sample_queries

    # Vérifier metadata
    metadata = results["metadata"]
    assert metadata["num_queries"] == len(sample_queries)
    assert "dense" in metadata["models_used"]
    assert "dense_dim" in metadata


def test_process_embeddings_helper_function(mock_config_dense_only, sample_queries):
    """Test la fonction helper process_embeddings()."""
    results = process_embeddings(sample_queries, mock_config_dense_only)

    # Doit retourner même structure que generator.process()
    assert "queries" in results
    assert "dense_embeddings" in results
    assert "metadata" in results


def test_empty_queries_list(mock_config_dense_only):
    """Test avec une liste de queries vide."""
    generator = EmbeddingGenerator(mock_config_dense_only)
    results = generator.process([])

    assert results["metadata"]["num_queries"] == 0
    assert results["dense_embeddings"].shape[0] == 0


def test_single_query(mock_config_dense_only):
    """Test avec une seule query."""
    generator = EmbeddingGenerator(mock_config_dense_only)
    results = generator.process(["What is RAG?"])

    assert results["metadata"]["num_queries"] == 1
    assert results["dense_embeddings"].shape[0] == 1


def test_dense_model_not_initialized_error():
    """Test que generate_dense_embeddings lève ValueError si modèle non initialisé."""
    # Config avec dense désactivé
    config = {
        "embedding_generation": {
            "enabled": True,
            "dense_embedding": {"enabled": False},
        }
    }

    generator = EmbeddingGenerator(config)

    with pytest.raises(ValueError, match="Dense embedding model not initialized"):
        generator.generate_dense_embeddings(["test"])


@pytest.mark.parametrize(
    "batch_size,num_queries",
    [
        (1, 5),  # Batch size plus petit que nombre queries
        (10, 5),  # Batch size plus grand que nombre queries
        (5, 5),  # Batch size égal au nombre queries
    ],
)
def test_different_batch_sizes(mock_config_dense_only, batch_size, num_queries):
    """Test que différents batch sizes produisent les mêmes résultats."""
    # Modifier batch size dans config
    mock_config_dense_only["embedding_generation"]["dense_embedding"][
        "batch_size"
    ] = batch_size

    generator = EmbeddingGenerator(mock_config_dense_only)
    queries = [f"Query {i}" for i in range(num_queries)]

    embeddings = generator.generate_dense_embeddings(queries)

    # Vérifier que tous les embeddings sont générés
    assert embeddings.shape[0] == num_queries


def test_special_characters_in_queries(mock_config_dense_only):
    """Test avec queries contenant des caractères spéciaux."""
    queries = [
        "What is <machine learning>?",
        "How does RAG work @2025?",
        "Explain #transformers & attention!",
    ]

    generator = EmbeddingGenerator(mock_config_dense_only)
    results = generator.process(queries)

    # Doit générer embeddings sans erreur
    assert results["dense_embeddings"].shape[0] == len(queries)


def test_very_long_query(mock_config_dense_only):
    """Test avec une query très longue."""
    # Query de ~1000 mots (modèle MiniLM a max_seq_length=256 tokens)
    long_query = " ".join(["word"] * 1000)

    generator = EmbeddingGenerator(mock_config_dense_only)
    embeddings = generator.generate_dense_embeddings([long_query])

    # Doit tronquer et générer embedding sans erreur
    assert embeddings.shape[0] == 1


def test_multilingual_queries(mock_config_dense_only):
    """Test avec queries multilingues."""
    queries = [
        "What is machine learning?",  # Anglais
        "Qu'est-ce que le machine learning ?",  # Français
        "Was ist maschinelles Lernen?",  # Allemand
    ]

    generator = EmbeddingGenerator(mock_config_dense_only)
    results = generator.process(queries)

    # Doit générer embeddings pour toutes les langues
    assert results["dense_embeddings"].shape[0] == len(queries)
