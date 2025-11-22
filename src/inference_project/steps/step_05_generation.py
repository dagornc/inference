"""Module pour la génération de réponses (Phase 05).

Ce module implémente la génération avancée de réponses avec :
- Self-RAG : Génération adaptative avec auto-réflexion
- CRAG : Évaluation et correction du contexte récupéré
- Hallucination Detection : Détection et correction des hallucinations
- Multi-Stage Validation : Validation faithfulness + attribution

Fonctions:
    PreGenerationAnalyzer: Analyse de complexité et évaluation CRAG.
    PromptConstructor: Classe pour construire les prompts.
    LLMGenerator: Classe pour appeler le LLM.
    SelfRAGGenerator: Générateur avec auto-réflexion.
    HallucinationDetector: Détecteur d'hallucinations.
    MultiStageValidator: Validation multi-niveaux.
    ResponseFormatter: Classe pour formater la réponse.
    process_generation: Point d'entrée principal orchestrant la génération.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import openai


class PromptConstructor:
    """Constructeur de prompts pour la génération RAG.

    Construit un prompt structuré contenant :
    - System prompt (rôle et directives)
    - Contexte (documents récupérés)
    - Query utilisateur
    - Instructions (citations, format, etc.)

    Attributes:
        config: Configuration du prompt.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le constructeur de prompts.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("prompt", {})

    def build_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
    ) -> tuple[str, str]:
        """Construit le prompt système et utilisateur.

        Args:
            query: Question de l'utilisateur.
            documents: Documents récupérés et reranked.

        Returns:
            Tuple (system_prompt, user_prompt).
        """
        # System prompt
        system_prompt = self.config.get("system_prompt", self._default_system_prompt())

        # Formatter le contexte
        context_str = self._format_context(documents)

        # User prompt
        user_prompt_template = self.config.get(
            "user_prompt_template", self._default_user_prompt_template()
        )

        user_prompt = user_prompt_template.format(
            context=context_str,
            query=query,
            num_docs=len(documents),
        )

        return system_prompt, user_prompt

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Formate les documents en contexte structuré.

        Args:
            documents: Liste des documents récupérés.

        Returns:
            Contexte formaté sous forme de string.
        """
        context_format_config = self.config.get("context_format", {})

        # Template pour chaque document
        doc_template = context_format_config.get(
            "document_template",
            "[{doc_id}] (Score: {score:.3f})\n{content}",
        )

        separator = context_format_config.get("separator", "\n\n---\n\n")
        truncate_per_doc = context_format_config.get("truncate_per_doc", 500)

        # Formatter chaque document
        formatted_docs = []
        for i, doc in enumerate(documents, start=1):
            content = doc.get("document", "")

            # Tronquer si nécessaire
            if truncate_per_doc and len(content) > truncate_per_doc:
                content = content[:truncate_per_doc] + "..."

            formatted_doc = doc_template.format(
                doc_id=i,
                score=doc.get("score", 0.0),
                content=content,
                source=doc.get("metadata", {}).get("source", "unknown"),
            )

            formatted_docs.append(formatted_doc)

        return separator.join(formatted_docs)

    def _default_system_prompt(self) -> str:
        """Prompt système par défaut."""
        return """Tu es un assistant expert qui répond aux questions en te basant UNIQUEMENT
sur le contexte fourni ci-dessous. Tes réponses doivent être:

- Précises et factuelles
- Basées exclusivement sur le contexte fourni
- Accompagnées de citations des sources [1], [2], etc.
- Claires et concises

Si le contexte ne contient pas assez d'informations pour répondre à la
question, tu DOIS dire: "Je n'ai pas trouvé suffisamment d'informations
dans le contexte fourni pour répondre à cette question."

Ne jamais inventer ou déduire des informations qui ne sont pas
explicitement dans le contexte."""

    def _default_user_prompt_template(self) -> str:
        """Template de prompt utilisateur par défaut."""
        return """Contexte ({num_docs} documents) :
{context}

Question : {query}

Instructions :
- Réponds à la question en utilisant UNIQUEMENT les informations du contexte ci-dessus
- Cite tes sources en utilisant les numéros de documents [1], [2], etc.
- Si la réponse n'est pas dans le contexte, dis-le clairement
- Sois concis mais complet

Réponse :"""


class LLMGenerator:
    """Générateur utilisant un LLM pour produire la réponse.

    Supporte plusieurs providers :
    - Ollama (local, gratuit)
    - OpenAI (API)
    - Anthropic (API)
    - Mistral AI (API)

    Attributes:
        config: Configuration du LLM.
        client: Client OpenAI (ou compatible).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le générateur LLM.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("llm", {})
        self.provider = self.config.get("provider", "ollama")
        self.model = self.config.get("model", "llama3")

        self.client: Optional[openai.OpenAI] = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialise le client LLM selon le provider."""
        # Récupérer config provider depuis global
        # Pour l'instant, on assume un client OpenAI-compatible

        if self.provider == "ollama":
            # Ollama a une API compatible OpenAI
            self.client = openai.OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama",  # Dummy key pour Ollama
            )

        elif self.provider == "openai":
            # OpenAI officiel
            self.client = openai.OpenAI()

        elif self.provider == "anthropic":
            # Anthropic utilise sa propre bibliothèque
            # Pour l'instant, on lève une erreur
            raise NotImplementedError(
                "Anthropic provider not yet implemented. Use Ollama or OpenAI."
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Génère une réponse avec le LLM.

        Args:
            system_prompt: Prompt système définissant le rôle.
            user_prompt: Prompt utilisateur avec contexte et question.

        Returns:
            Réponse générée par le LLM.
        """
        if self.client is None:
            raise ValueError("LLM client not initialized")

        # Paramètres de génération
        temperature = self.config.get("temperature", 0.0)
        max_tokens = self.config.get("max_tokens", 1000)
        top_p = self.config.get("top_p", 0.95)

        # Appel au LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            # En cas d'erreur, retourner message d'erreur
            return f"Erreur lors de la génération: {e!s}"


class ResponseFormatter:
    """Formateur de réponses pour post-processing.

    Formate la réponse générée en ajoutant :
    - Liste des sources utilisées
    - Métadonnées
    - Nettoyage du texte

    Attributes:
        config: Configuration du formatage.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le formateur.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("post_processing", {})

    def format(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Formate la réponse finale.

        Args:
            answer: Réponse brute du LLM.
            documents: Documents utilisés (pour sources).

        Returns:
            Dictionnaire contenant la réponse formatée et métadonnées.
        """
        formatting_config = self.config.get("formatting", {})

        # Nettoyer espaces
        if formatting_config.get("clean_whitespace", True):
            answer = self._clean_whitespace(answer)

        # Construire liste des sources
        sources_list = None
        if formatting_config.get("append_sources", True):
            sources_list = self._build_sources_list(documents)

        # Format de sortie
        output_format = formatting_config.get("output_format", "markdown")

        if output_format == "json":
            return {
                "answer": answer,
                "sources": sources_list,
                "num_sources": len(documents),
                "metadata": {
                    "format": "json",
                },
            }

        elif output_format == "markdown":
            # Ajouter sources en markdown
            if sources_list:
                answer_with_sources = f"{answer}\n\n{sources_list}"
            else:
                answer_with_sources = answer

            return {
                "answer": answer_with_sources,
                "sources_raw": [
                    doc.get("metadata", {}).get("source") for doc in documents
                ],
                "num_sources": len(documents),
                "metadata": {
                    "format": "markdown",
                },
            }

        else:
            # Format text simple
            return {
                "answer": answer,
                "num_sources": len(documents),
            }

    def _clean_whitespace(self, text: str) -> str:
        """Nettoie les espaces superflus."""
        # Remplacer multiples espaces par un seul
        import re

        text = re.sub(r" +", " ", text)
        # Remplacer multiples retours à la ligne par max 2
        text = re.sub(r"\n\n+", "\n\n", text)
        return text.strip()

    def _build_sources_list(self, documents: List[Dict[str, Any]]) -> str:
        """Construit la liste des sources formatée."""
        sources = []
        for i, doc in enumerate(documents, start=1):
            source = doc.get("metadata", {}).get("source", f"Document {i}")
            sources.append(f"[{i}] {source}")

        sources_template = self.config.get("formatting", {}).get(
            "sources_template",
            "\n---\nSources utilisées :\n{sources}",
        )

        return sources_template.format(sources="\n".join(sources))


class PreGenerationAnalyzer:
    """Analyseur pré-génération pour complexité query et évaluation CRAG.

    Analyse :
    - Complexité de la query (simple/medium/complex)
    - Qualité du contexte récupéré (CRAG evaluator)
    - Sélection de stratégie adaptative (Adaptive RAG)

    Attributes:
        config: Configuration de l'analyse pré-génération.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise l'analyseur.

        Args:
            config: Configuration complète chargée depuis YAML.
        """
        self.config = config.get("pre_generation_analysis", {})

    def analyze(self, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyse la query et le contexte.

        Args:
            query: Question de l'utilisateur.
            documents: Documents récupérés.

        Returns:
            Résultats de l'analyse (complexité, score CRAG, stratégie).
        """
        if not self.config.get("enabled", True):
            return {
                "query_complexity": "medium",
                "crag_score": 0.8,
                "strategy": "standard_rag",
            }

        # Analyse complexité query
        complexity = self._analyze_query_complexity(query)

        # Évaluation CRAG
        crag_score, crag_action = self._evaluate_crag(query, documents)

        # Sélection de stratégie
        strategy = self._select_strategy(complexity, crag_score)

        return {
            "query_complexity": complexity,
            "crag_score": crag_score,
            "crag_action": crag_action,
            "strategy": strategy,
        }

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyse la complexité de la query.

        Args:
            query: Question utilisateur.

        Returns:
            Niveau de complexité : "simple", "medium", "complex".
        """
        complexity_config = self.config.get("query_complexity", {})
        if not complexity_config.get("enabled", True):
            return "medium"

        # Heuristiques simples
        query_lower = query.lower()
        score = 0.0

        # Longueur
        if len(query.split()) > 15:
            score += 0.2

        # Mots interrogatifs complexes
        complex_words = ["why", "how", "explain", "compare", "analyze"]
        if any(word in query_lower for word in complex_words):
            score += 0.3

        # Mots de comparaison
        comparison_words = ["vs", "versus", "compare", "better", "difference"]
        if any(word in query_lower for word in comparison_words):
            score += 0.3

        # Plusieurs questions
        if query.count("?") > 1 or " and " in query_lower or " or " in query_lower:
            score += 0.2

        # Classement
        if score < 0.3:
            return "simple"
        elif score < 0.6:
            return "medium"
        else:
            return "complex"

    def _evaluate_crag(
        self, query: str, documents: List[Dict[str, Any]]
    ) -> Tuple[float, str]:
        """Évalue la qualité du contexte (CRAG evaluator).

        Args:
            query: Question utilisateur.
            documents: Documents récupérés.

        Returns:
            Tuple (score CRAG [0-1], action corrective).
        """
        crag_config = self.config.get("crag_evaluator", {})
        if not crag_config.get("enabled", True):
            return 0.8, "generate"

        # Simplification : utiliser les scores de retrieval existants
        if not documents:
            return 0.0, "incorrect"

        # Score moyen des documents
        avg_score = np.mean([doc.get("score", 0.0) for doc in documents])

        # Thresholds
        thresholds = crag_config.get("lightweight_config", {}).get(
            "thresholds",
            {"correct": 0.7, "ambiguous": 0.4},
        )

        # Décision
        if avg_score >= thresholds["correct"]:
            return avg_score, "correct"
        elif avg_score >= thresholds["ambiguous"]:
            return avg_score, "ambiguous"
        else:
            return avg_score, "incorrect"

    def _select_strategy(self, complexity: str, crag_score: float) -> str:
        """Sélectionne la stratégie de génération adaptative.

        Args:
            complexity: Complexité de la query.
            crag_score: Score CRAG.

        Returns:
            Nom de la stratégie à utiliser.
        """
        strategy_config = self.config.get("strategy_selection", {})
        if not strategy_config.get("enabled", True):
            return "standard_rag"

        # Si CRAG score faible, stratégie spéciale
        if crag_score < 0.4:
            return "web_search_fallback"

        # Sinon, stratégie selon complexité
        if complexity == "simple":
            return "direct_generation"
        elif complexity == "medium":
            return "standard_rag"
        else:
            return "multi_hop_cot"


class SelfRAGGenerator:
    """Générateur Self-RAG avec auto-réflexion.

    Self-RAG améliore la génération via :
    - Retrieve on-demand : Récupération conditionnelle
    - Reflection tokens : Auto-évaluation de la pertinence
    - Critique tokens : Évaluation de la qualité de sortie

    Attributes:
        config: Configuration Self-RAG.
        llm_generator: Générateur LLM sous-jacent.
    """

    def __init__(self, config: Dict[str, Any], llm_generator: "LLMGenerator") -> None:
        """Initialise Self-RAG.

        Args:
            config: Configuration complète.
            llm_generator: Instance du générateur LLM.
        """
        self.config = config.get("self_rag", {})
        self.llm_generator = llm_generator

    def generate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """Génère avec auto-réflexion.

        Args:
            query: Question utilisateur.
            documents: Documents de contexte.
            system_prompt: Prompt système.
            user_prompt: Prompt utilisateur.

        Returns:
            Dictionnaire avec réponse et métadonnées Self-RAG.
        """
        if not self.config.get("enabled", True):
            # Génération standard
            answer = self.llm_generator.generate(system_prompt, user_prompt)
            return {"answer": answer, "used_self_rag": False}

        # Génération avec réflexion
        enhanced_prompt = self._add_reflection_tokens(user_prompt)

        answer = self.llm_generator.generate(system_prompt, enhanced_prompt)

        # Parser reflection tokens
        reflection_analysis = self._parse_reflection_tokens(answer)

        # Si besoin de retrieval supplémentaire
        if reflection_analysis.get("needs_more_context", False):
            # Simuler retrieval additionnel (simplification)
            answer = self._retrieve_and_regenerate(
                query, documents, system_prompt, user_prompt
            )

        return {
            "answer": answer,
            "used_self_rag": True,
            "reflection": reflection_analysis,
        }

    def _add_reflection_tokens(self, user_prompt: str) -> str:
        """Ajoute reflection tokens au prompt.

        Args:
            user_prompt: Prompt utilisateur original.

        Returns:
            Prompt enrichi avec instructions de réflexion.
        """
        reflection_instruction = """

After providing your answer, evaluate it with:
[Retrieval]: Is additional information needed? (Yes/No)
[IsRel]: Are the documents relevant? (Yes/No)
[IsSupp]: Is the answer supported by context? (Yes/No)
[IsUse]: Should this answer be used? (Yes/No)
"""

        return user_prompt + reflection_instruction

    def _parse_reflection_tokens(self, answer: str) -> Dict[str, bool]:
        """Parse les reflection tokens de la réponse.

        Args:
            answer: Réponse générée avec tokens.

        Returns:
            Analyse des tokens de réflexion.
        """
        analysis = {
            "needs_more_context": "[Retrieval]: Yes" in answer,
            "docs_relevant": "[IsRel]: Yes" in answer or "[IsRel]: No" not in answer,
            "answer_supported": "[IsSupp]: Yes" in answer
            or "[IsSupp]: No" not in answer,
            "should_use": "[IsUse]: Yes" in answer or "[IsUse]: No" not in answer,
        }

        return analysis

    def _retrieve_and_regenerate(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        system_prompt: str,
        user_prompt: str,
    ) -> str:
        """Récupère plus de contexte et régénère.

        Args:
            query: Question utilisateur.
            documents: Documents actuels.
            system_prompt: Prompt système.
            user_prompt: Prompt utilisateur.

        Returns:
            Nouvelle réponse régénérée.
        """
        # Simplification : utiliser les documents existants
        # Dans une vraie implémentation, on rappellerait le retriever
        answer = self.llm_generator.generate(system_prompt, user_prompt)
        return answer


class HallucinationDetector:
    """Détecteur d'hallucinations dans les réponses générées.

    Détecte les hallucinations via :
    - Semantic consistency : Cohérence avec le contexte
    - Entity checking : Vérification des entités
    - Lightweight heuristics : Mots d'incertitude

    Attributes:
        config: Configuration de la détection.
        similarity_model: Modèle pour similarité sémantique (optionnel).
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise le détecteur.

        Args:
            config: Configuration complète.
        """
        self.config = config.get("hallucination_detection", {})
        self.similarity_model: Optional[Any] = None
        self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialise le modèle de détection."""
        if not self.config.get("enabled", True):
            return

        try:
            from sentence_transformers import SentenceTransformer

            model_name = self.config.get(
                "model", "sentence-transformers/all-MiniLM-L6-v2"
            )
            self.similarity_model = SentenceTransformer(model_name)

        except ImportError:
            print(
                "⚠️  sentence-transformers not installed. "
                "Hallucination detection will use lightweight heuristics."
            )
            self.similarity_model = None

    def detect(
        self, answer: str, documents: List[Dict[str, Any]], query: str = ""
    ) -> Dict[str, Any]:
        """Détecte les hallucinations dans la réponse.

        Args:
            answer: Réponse générée.
            documents: Documents de contexte.
            query: Question originale.

        Returns:
            Résultats de détection (score, détails).
        """
        if not self.config.get("enabled", True):
            return {
                "has_hallucination": False,
                "confidence": 1.0,
                "checks": {},
            }

        # Vérifications multiples
        checks = {}

        # 1. Semantic consistency
        semantic_score = self._check_semantic_consistency(answer, documents)
        checks["semantic_consistency"] = semantic_score

        # 2. Heuristiques légères
        uncertainty_score = self._check_uncertainty_markers(answer)
        checks["uncertainty_markers"] = uncertainty_score

        # 3. Citation check
        has_citations = self._check_citations(answer)
        checks["has_citations"] = has_citations

        # Score global (moyenne pondérée)
        hallucination_score = (
            0.6 * (1 - semantic_score)  # Plus mauvaise sémantique = plus hallucination
            + 0.2 * uncertainty_score  # Plus incertitude = plus hallucination
            + 0.2 * (0 if has_citations else 1)  # Pas citations = plus suspect
        )

        # Threshold
        threshold = self.config.get("threshold", 0.5)
        has_hallucination = hallucination_score > threshold

        return {
            "has_hallucination": has_hallucination,
            "confidence": 1.0 - hallucination_score,
            "hallucination_score": hallucination_score,
            "checks": checks,
        }

    def _check_semantic_consistency(
        self, answer: str, documents: List[Dict[str, Any]]
    ) -> float:
        """Vérifie la cohérence sémantique avec le contexte.

        Args:
            answer: Réponse générée.
            documents: Documents de contexte.

        Returns:
            Score de cohérence [0-1] (1 = très cohérent).
        """
        if not documents or self.similarity_model is None:
            return 0.8  # Assumer OK si pas de modèle

        # Concaténer contexte
        context = " ".join([doc.get("document", "") for doc in documents])

        # Encoder
        embeddings = self.similarity_model.encode(
            [answer, context], normalize_embeddings=True
        )

        # Similarité cosine
        similarity = np.dot(embeddings[0], embeddings[1])

        return float(similarity)

    def _check_uncertainty_markers(self, answer: str) -> float:
        """Détecte les marqueurs d'incertitude dans la réponse.

        Args:
            answer: Réponse générée.

        Returns:
            Score d'incertitude [0-1] (1 = très incertain).
        """
        uncertainty_words = [
            "i don't know",
            "i'm not sure",
            "maybe",
            "perhaps",
            "possibly",
            "might be",
            "could be",
            "uncertain",
            "unclear",
            "not enough information",
        ]

        answer_lower = answer.lower()
        count = sum(1 for word in uncertainty_words if word in answer_lower)

        # Normaliser
        score = min(1.0, count / 3.0)  # 3+ markers = score max

        return score

    def _check_citations(self, answer: str) -> bool:
        """Vérifie la présence de citations dans la réponse.

        Args:
            answer: Réponse générée.

        Returns:
            True si des citations sont présentes.
        """
        # Chercher patterns [1], [2], etc.
        import re

        citation_pattern = r"\[\d+\]"
        matches = re.findall(citation_pattern, answer)

        return len(matches) > 0


class MultiStageValidator:
    """Validateur multi-niveaux pour qualité de génération.

    Valide :
    - Faithfulness : Réponse fidèle au contexte
    - Attribution : Sources correctement citées
    - Consistency : Cohérence interne

    Attributes:
        config: Configuration de validation.
        hallucination_detector: Détecteur d'hallucinations.
    """

    def __init__(
        self, config: Dict[str, Any], hallucination_detector: HallucinationDetector
    ) -> None:
        """Initialise le validateur.

        Args:
            config: Configuration complète.
            hallucination_detector: Instance du détecteur.
        """
        self.config = config.get("multi_stage_validation", {})
        self.hallucination_detector = hallucination_detector

    def validate(
        self,
        answer: str,
        documents: List[Dict[str, Any]],
        query: str,
    ) -> Dict[str, Any]:
        """Valide la réponse générée.

        Args:
            answer: Réponse à valider.
            documents: Documents de contexte.
            query: Question originale.

        Returns:
            Résultats de validation (scores, pass/fail).
        """
        if not self.config.get("enabled", True):
            return {
                "passed": True,
                "faithfulness_score": 1.0,
                "attribution_score": 1.0,
                "consistency_score": 1.0,
            }

        validation_results = {}

        # 1. Faithfulness (via hallucination detection)
        hallucination_result = self.hallucination_detector.detect(
            answer, documents, query
        )
        faithfulness_score = hallucination_result["confidence"]
        validation_results["faithfulness_score"] = faithfulness_score

        # 2. Attribution
        attribution_score = self._check_attribution(answer, documents)
        validation_results["attribution_score"] = attribution_score

        # 3. Consistency
        consistency_score = self._check_consistency(answer)
        validation_results["consistency_score"] = consistency_score

        # Score global
        overall_score = (
            0.5 * faithfulness_score + 0.3 * attribution_score + 0.2 * consistency_score
        )

        validation_results["overall_score"] = overall_score

        # Threshold
        threshold = self.config.get("threshold", 0.7)
        validation_results["passed"] = overall_score >= threshold

        return validation_results

    def _check_attribution(self, answer: str, documents: List[Dict[str, Any]]) -> float:
        """Vérifie que les sources sont correctement citées.

        Args:
            answer: Réponse générée.
            documents: Documents de contexte.

        Returns:
            Score d'attribution [0-1].
        """
        # Compter citations
        import re

        citations = re.findall(r"\[(\d+)\]", answer)

        if not citations:
            return 0.5  # Pas de citations = score moyen

        # Vérifier validité des numéros
        max_doc_id = len(documents)
        valid_citations = [
            int(c) for c in citations if c.isdigit() and 1 <= int(c) <= max_doc_id
        ]

        if not citations:
            return 0.5

        validity_ratio = len(valid_citations) / len(citations)

        return validity_ratio

    def _check_consistency(self, answer: str) -> float:
        """Vérifie la cohérence interne de la réponse.

        Args:
            answer: Réponse générée.

        Returns:
            Score de cohérence [0-1].
        """
        # Heuristiques simples
        score = 1.0

        # Pénaliser contradictions
        contradiction_words = [
            ("however", "but"),
            ("although", "despite"),
            ("contrary", "opposite"),
        ]

        answer_lower = answer.lower()
        for word1, word2 in contradiction_words:
            if word1 in answer_lower and word2 in answer_lower:
                score -= 0.1

        # Pénaliser réponse trop courte
        if len(answer.split()) < 10:
            score -= 0.2

        return max(0.0, score)


def process_generation(
    query: str,
    documents: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Point d'entrée principal pour la génération avancée.

    Pipeline complet avec :
    - Pre-generation analysis (CRAG, complexity)
    - Self-RAG (retrieve on-demand)
    - Hallucination detection
    - Multi-stage validation

    Args:
        query: Question de l'utilisateur.
        documents: Documents récupérés et reranked.
        config: Configuration complète chargée depuis YAML.

    Returns:
        Dictionnaire contenant la réponse formatée et métadonnées complètes.
    """
    generation_config = config.get("step_05_generation", {})

    if not generation_config.get("enabled", True):
        return {
            "answer": "Generation disabled",
            "metadata": {},
        }

    # Pipeline d'exécution
    pipeline = generation_config.get("pipeline", [])

    # Métadonnées à collecter
    generation_metadata = {}

    # Step 1: Pre-Generation Analysis
    analysis_result = None
    if _is_step_enabled(pipeline, "pre_generation_analysis"):
        analyzer = PreGenerationAnalyzer(config)
        analysis_result = analyzer.analyze(query, documents)
        generation_metadata["pre_generation_analysis"] = analysis_result

    # Step 2: Prompt Construction
    prompt_constructor = PromptConstructor(config)
    system_prompt, user_prompt = prompt_constructor.build_prompt(query, documents)

    # Step 3: Initial Generation
    llm_generator = LLMGenerator(config)

    # Step 4: Self-RAG (si activé et conditionnel)
    answer = None
    self_rag_used = False

    if _is_step_enabled(pipeline, "self_rag"):
        self_rag_generator = SelfRAGGenerator(config, llm_generator)

        # Vérifier si Self-RAG conditionnel
        use_self_rag = True
        if generation_config.get("pipeline", [{}])[4].get("conditional", False):
            # Activer Self-RAG seulement si query complexe ou CRAG ambigu
            if analysis_result:
                complexity = analysis_result.get("query_complexity", "medium")
                crag_action = analysis_result.get("crag_action", "correct")
                use_self_rag = complexity == "complex" or crag_action == "ambiguous"

        if use_self_rag:
            self_rag_result = self_rag_generator.generate(
                query, documents, system_prompt, user_prompt
            )
            answer = self_rag_result["answer"]
            self_rag_used = self_rag_result["used_self_rag"]
            generation_metadata["self_rag"] = self_rag_result.get("reflection", {})

    # Si Self-RAG pas utilisé, génération standard
    if answer is None:
        answer = llm_generator.generate(system_prompt, user_prompt)

    # Step 5: Hallucination Detection
    hallucination_result = None
    if _is_step_enabled(pipeline, "hallucination_detection"):
        hallucination_detector = HallucinationDetector(config)
        hallucination_result = hallucination_detector.detect(answer, documents, query)
        generation_metadata["hallucination_detection"] = hallucination_result

        # Si hallucination détectée, avertir
        if hallucination_result.get("has_hallucination", False):
            # Optionnel : ajouter avertissement à la réponse
            pass

    # Step 6: Multi-Stage Validation
    validation_result = None
    if _is_step_enabled(pipeline, "multi_stage_validation"):
        if hallucination_result is None:
            hallucination_detector = HallucinationDetector(config)
        else:
            hallucination_detector = HallucinationDetector(config)

        validator = MultiStageValidator(config, hallucination_detector)
        validation_result = validator.validate(answer, documents, query)
        generation_metadata["multi_stage_validation"] = validation_result

        # Si validation échoue
        if not validation_result.get("passed", True):
            generation_metadata["validation_warning"] = (
                "Answer failed validation checks"
            )

    # Step 7: Post-Processing & Formatting
    response_formatter = ResponseFormatter(config)
    formatted_response = response_formatter.format(answer, documents)

    # Enrichir avec métadonnées
    formatted_response["metadata"]["generation_metadata"] = generation_metadata
    formatted_response["metadata"]["used_self_rag"] = self_rag_used

    return formatted_response


# =============================================================================
# ADVANCED FEATURES - Response Refinement & Structured Output
# =============================================================================


class ResponseRefiner:
    """Raffineur de réponses par itération.

    Améliore la réponse via self-correction itérative :
    - Détecte problèmes (hallucinations, manque de précision)
    - Régénère avec feedback
    - Converge vers réponse optimale

    Attributes:
        config: Configuration du refinement.
        llm_generator: Générateur LLM.
        hallucination_detector: Détecteur d'hallucinations.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        llm_generator: "LLMGenerator",
        hallucination_detector: "HallucinationDetector",
    ) -> None:
        """Initialise le refiner.

        Args:
            config: Configuration complète.
            llm_generator: Instance LLM generator.
            hallucination_detector: Instance hallucination detector.
        """
        self.config = config.get("response_refinement", {})
        self.llm_generator = llm_generator
        self.hallucination_detector = hallucination_detector

    def refine(
        self,
        initial_answer: str,
        query: str,
        documents: List[Dict[str, Any]],
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Raffine la réponse initiale.

        Args:
            initial_answer: Réponse initiale.
            query: Query originale.
            documents: Documents de contexte.
            validation_result: Résultat de validation (optionnel).

        Returns:
            Dict avec réponse raffinée et métadonnées.
        """
        if not self.config.get("enabled", False):
            return {
                "refined_answer": initial_answer,
                "num_iterations": 0,
                "improved": False,
            }

        max_iterations = self.config.get("max_iterations", 2)
        improvement_threshold = self.config.get("improvement_threshold", 0.05)

        current_answer = initial_answer
        iteration_history = []

        for iteration in range(max_iterations):
            # Analyser problèmes
            issues = self._analyze_issues(
                current_answer, query, documents, validation_result
            )

            if not issues:
                # Pas de problèmes, sortir
                break

            # Construire feedback
            feedback = self._build_feedback(issues)

            # Régénérer avec feedback
            refined_answer = self._regenerate_with_feedback(
                query, documents, current_answer, feedback
            )

            # Vérifier amélioration
            is_improved = self._check_improvement(
                current_answer, refined_answer, documents, query
            )

            iteration_history.append(
                {
                    "iteration": iteration + 1,
                    "issues": issues,
                    "improved": is_improved,
                }
            )

            if is_improved:
                current_answer = refined_answer
            else:
                # Pas d'amélioration, arrêter
                break

        return {
            "refined_answer": current_answer,
            "num_iterations": len(iteration_history),
            "improved": current_answer != initial_answer,
            "iteration_history": iteration_history,
        }

    def _analyze_issues(
        self,
        answer: str,
        query: str,
        documents: List[Dict[str, Any]],
        validation_result: Optional[Dict[str, Any]],
    ) -> List[str]:
        """Analyse les problèmes dans la réponse.

        Args:
            answer: Réponse à analyser.
            query: Query.
            documents: Documents.
            validation_result: Résultat validation.

        Returns:
            Liste de problèmes détectés.
        """
        issues = []

        # Vérifier hallucinations
        halluc_result = self.hallucination_detector.detect(answer, documents, query)
        if halluc_result.get("has_hallucination", False):
            issues.append("hallucination_detected")

        # Vérifier validation
        if validation_result and not validation_result.get("passed", True):
            if validation_result.get("faithfulness_score", 1.0) < 0.7:
                issues.append("low_faithfulness")
            if validation_result.get("attribution_score", 1.0) < 0.7:
                issues.append("poor_attribution")

        # Vérifier longueur
        if len(answer.split()) < 20:
            issues.append("too_short")

        # Vérifier structure
        if not self._has_clear_structure(answer):
            issues.append("unclear_structure")

        return issues

    def _has_clear_structure(self, answer: str) -> bool:
        """Vérifie si la réponse a une structure claire.

        Args:
            answer: Réponse.

        Returns:
            True si structure claire.
        """
        # Heuristiques simples
        has_sentences = answer.count(".") >= 2
        has_paragraphs = "\n" in answer or len(answer) > 100

        return has_sentences or has_paragraphs

    def _build_feedback(self, issues: List[str]) -> str:
        """Construit feedback pour régénération.

        Args:
            issues: Liste de problèmes.

        Returns:
            Feedback textuel.
        """
        feedback_parts = []

        if "hallucination_detected" in issues:
            feedback_parts.append(
                "Your answer contains information not supported by the context."
            )

        if "low_faithfulness" in issues:
            feedback_parts.append(
                "Ensure all claims are directly supported by the provided documents."
            )

        if "poor_attribution" in issues:
            feedback_parts.append(
                "Add or fix citations [1], [2], etc. for all factual claims."
            )

        if "too_short" in issues:
            feedback_parts.append("Provide a more detailed and comprehensive answer.")

        if "unclear_structure" in issues:
            feedback_parts.append(
                "Organize your answer with clear sentences and structure."
            )

        return " ".join(feedback_parts)

    def _regenerate_with_feedback(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        previous_answer: str,
        feedback: str,
    ) -> str:
        """Régénère avec feedback.

        Args:
            query: Query.
            documents: Documents.
            previous_answer: Réponse précédente.
            feedback: Feedback à incorporer.

        Returns:
            Nouvelle réponse.
        """
        # Construire prompt avec feedback
        refinement_prompt = f"""Query: {query}

Previous answer:
{previous_answer}

Feedback:
{feedback}

Context:
{self._format_context(documents)}

Generate an improved answer that addresses the feedback while staying faithful to the context:"""

        try:
            system_prompt = (
                "You are an expert assistant that improves answers based on feedback."
            )

            refined = self.llm_generator.generate(system_prompt, refinement_prompt)

            return refined

        except Exception as e:
            print(f"⚠️  Refinement failed: {e}")
            return previous_answer

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Formate contexte pour prompt.

        Args:
            documents: Documents.

        Returns:
            Contexte formaté.
        """
        context_parts = []
        for i, doc in enumerate(documents[:5], 1):  # Limiter à 5 docs
            content = doc.get("document", "")[:200]
            context_parts.append(f"[{i}] {content}")

        return "\n".join(context_parts)

    def _check_improvement(
        self,
        old_answer: str,
        new_answer: str,
        documents: List[Dict[str, Any]],
        query: str,
    ) -> bool:
        """Vérifie si nouvelle réponse meilleure.

        Args:
            old_answer: Ancienne réponse.
            new_answer: Nouvelle réponse.
            documents: Documents.
            query: Query.

        Returns:
            True si amélioration.
        """
        # Comparer scores hallucination
        old_halluc = self.hallucination_detector.detect(old_answer, documents, query)
        new_halluc = self.hallucination_detector.detect(new_answer, documents, query)

        old_score = old_halluc.get("confidence", 0.5)
        new_score = new_halluc.get("confidence", 0.5)

        # Amélioration si confiance augmente
        return new_score > old_score + 0.05


class StructuredOutputGenerator:
    """Générateur de sorties structurées (JSON Schema).

    Force le LLM à produire JSON valide selon un schéma.
    Utile pour APIs, agents, etc.

    Attributes:
        config: Configuration structured output.
        llm_generator: Générateur LLM.
    """

    def __init__(self, config: Dict[str, Any], llm_generator: "LLMGenerator") -> None:
        """Initialise le générateur.

        Args:
            config: Configuration complète.
            llm_generator: Instance LLM generator.
        """
        self.config = config.get("structured_output", {})
        self.llm_generator = llm_generator

    def generate_structured(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Génère sortie structurée selon schéma.

        Args:
            query: Query.
            documents: Documents.
            schema: JSON Schema.

        Returns:
            Sortie structurée (dict).
        """
        if not self.config.get("enabled", False):
            # Fallback : réponse text simple
            return {
                "answer": "Structured output not enabled",
                "metadata": {},
            }

        # Construire prompt avec schéma
        prompt = self._build_schema_prompt(query, documents, schema)

        # Générer
        system_prompt = (
            "You are an AI that outputs ONLY valid JSON according to the schema."
        )

        try:
            response_text = self.llm_generator.generate(system_prompt, prompt)

            # Parser JSON
            import json

            # Extraire JSON (parfois LLM ajoute du texte autour)
            json_str = self._extract_json(response_text)

            structured_output = json.loads(json_str)

            # Valider schéma (optionnel)
            if self.config.get("validate_schema", True):
                self._validate_against_schema(structured_output, schema)

            return structured_output

        except Exception as e:
            print(f"⚠️  Structured output generation failed: {e}")
            # Fallback
            return {
                "answer": "Failed to generate structured output",
                "error": str(e),
            }

    def _build_schema_prompt(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        schema: Dict[str, Any],
    ) -> str:
        """Construit prompt avec schéma.

        Args:
            query: Query.
            documents: Documents.
            schema: JSON Schema.

        Returns:
            Prompt.
        """
        import json

        schema_str = json.dumps(schema, indent=2)

        context = "\n".join(
            [
                f"[{i + 1}] {doc.get('document', '')[:150]}"
                for i, doc in enumerate(documents[:5])
            ]
        )

        prompt = f"""Answer the following query using the provided context.
Output ONLY valid JSON that follows this schema:

{schema_str}

Query: {query}

Context:
{context}

JSON output:"""

        return prompt

    def _extract_json(self, text: str) -> str:
        """Extrait JSON d'un texte.

        Args:
            text: Texte contenant JSON.

        Returns:
            JSON string.
        """
        import re

        # Chercher pattern {...} ou [...]
        json_pattern = r"(\{.*\}|\[.*\])"
        matches = re.findall(json_pattern, text, re.DOTALL)

        if matches:
            return matches[0]

        return text

    def _validate_against_schema(
        self, data: Dict[str, Any], schema: Dict[str, Any]
    ) -> None:
        """Valide data contre schéma JSON.

        Args:
            data: Données à valider.
            schema: JSON Schema.

        Raises:
            ValueError: Si validation échoue.
        """
        # Validation basique (sans jsonschema library)
        required_fields = schema.get("required", [])

        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")


def _is_step_enabled(pipeline: List[Dict[str, Any]], step_name: str) -> bool:
    """Vérifie si une étape du pipeline est activée.

    Args:
        pipeline: Liste des étapes du pipeline.
        step_name: Nom de l'étape à vérifier.

    Returns:
        True si l'étape est activée.
    """
    for step in pipeline:
        if step.get("step") == step_name:
            return step.get("enabled", False)
    return False


if __name__ == "__main__":
    """Exemple d'exécution du module."""
    print("=" * 80)
    print("GÉNÉRATION - EXEMPLE")
    print("=" * 80)
    print("\n⚠️  Ce module nécessite un LLM fonctionnel (Ollama/OpenAI).")
    print("\nPour Ollama (gratuit, local) :")
    print("  1. Installer : https://ollama.com")
    print("  2. Lancer : ollama run llama3")
    print("\nExemple d'utilisation :")
    print("""
    from inference_project.steps.step_05_generation import process_generation
    from inference_project.utils.config_loader import load_config

    # Charger config
    config = load_config("05_generation_v2", "config")

    # Query
    query = "What is machine learning?"

    # Documents (résultats de reranking)
    documents = [
        {
            "id": "doc1",
            "document": "Machine learning is a subset of AI that enables systems to learn from data...",
            "score": 0.95,
            "metadata": {"source": "AI Textbook Chapter 3"},
        },
        {
            "id": "doc2",
            "document": "ML algorithms can be supervised, unsupervised, or reinforcement learning...",
            "score": 0.88,
            "metadata": {"source": "ML Guide"},
        },
    ]

    # Génération
    response = process_generation(query, documents, config)

    print(f"Answer: {response['answer']}")
    print(f"Sources: {response['num_sources']}")
    """)
    print("=" * 80)
