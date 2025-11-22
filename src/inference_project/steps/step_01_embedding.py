"""This module handles the query embedding process, including query expansion techniques.

It uses a language model to rewrite, expand, or generate hypothetical documents
based on the original user query. The goal is to improve the quality and
relevance of the context provided to the Retrieval-Augmented Generation (RAG)
system.

Functions:
    get_llm_client: Creates an LLM client from provider configuration.
    rewrite_query: Rewrites a query using a provided prompt.
    generate_hypothetical_document: Generates a hypothetical document.
    expand_multi_query: Expands a query into multiple variants.
    generate_step_back_question: Generates a step-back question.
    process_query: Processes a query by applying various expansion techniques.
"""

import os
from typing import Any

import openai

from inference_project.utils.config_loader import load_config

# In-memory cache for expanded queries
_query_cache: dict[str, list[str]] = {}


def get_llm_client(provider_config: dict[str, Any]) -> openai.OpenAI:
    """Creates an LLM client from the provider configuration.

    Args:
        provider_config: A dictionary containing the API key and base URL.

    Returns:
        An instance of the openai.OpenAI client.
    """
    return openai.OpenAI(
        api_key=provider_config.get("api_key"),
        base_url=provider_config.get("base_url"),
    )


def _create_chat_completion(
    prompt: str, client: openai.OpenAI, model: str, temperature: float, max_tokens: int
) -> str:
    """Helper function to create a chat completion and return the content.

    Args:
        prompt: The prompt to send to the language model.
        client: The OpenAI client instance.
        model: The name of the model to use.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        The content of the chat completion response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


def rewrite_query(
    prompt: str, client: openai.OpenAI, model: str, temperature: float, max_tokens: int
) -> str:
    """Rewrites a query using a provided prompt.

    Args:
        prompt: The prompt containing the query to rewrite.
        client: The OpenAI client instance.
        model: The name of the model to use.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        The rewritten query.
    """
    return _create_chat_completion(prompt, client, model, temperature, max_tokens)


def generate_hypothetical_document(
    prompt: str, client: openai.OpenAI, model: str, temperature: float, max_tokens: int
) -> str:
    """Generates a hypothetical document using a provided prompt.

    Args:
        prompt: The prompt to generate the document from.
        client: The OpenAI client instance.
        model: The name of the model to use.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        The generated hypothetical document.
    """
    return _create_chat_completion(prompt, client, model, temperature, max_tokens)


def expand_multi_query(
    prompt: str, client: openai.OpenAI, model: str, temperature: float, max_tokens: int
) -> list[str]:
    """Expands a query into multiple variants using a provided prompt.

    Args:
        prompt: The prompt to expand the query from.
        client: The OpenAI client instance.
        model: The name of the model to use.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        A list of query variants.
    """
    content = _create_chat_completion(prompt, client, model, temperature, max_tokens)
    variants = content.split("\n")
    return [v.strip() for v in variants if v.strip()]


def generate_step_back_question(
    prompt: str, client: openai.OpenAI, model: str, temperature: float, max_tokens: int
) -> str:
    """Generates a step-back question using a provided prompt.

    Args:
        prompt: The prompt to generate the question from.
        client: The OpenAI client instance.
        model: The name of the model to use.
        temperature: The sampling temperature.
        max_tokens: The maximum number of tokens to generate.

    Returns:
        The generated step-back question.
    """
    return _create_chat_completion(prompt, client, model, temperature, max_tokens)


def process_query(query: str, config: dict[str, Any]) -> list[str]:
    """Processes a query by applying various expansion techniques.

    Based on the provided configuration, this function can rewrite the query,
    generate a hypothetical document (HyDE), expand it into multiple queries,
    or generate a step-back question.

    Args:
        query: The original user query.
        config: The configuration dictionary, typically loaded from a YAML file.

    Returns:
        A list of strings containing the original query and all its generated
        variants.

    Raises:
        ValueError: If the specified LLM provider is not configured.
    """
    query_expansion_config = config.get("adaptive_query_expansion", {})

    if not query_expansion_config.get("enabled"):
        return [query]

    if query_expansion_config.get("cache_enabled") and query in _query_cache:
        return _query_cache[query]

    llm_config = query_expansion_config.get("llm", {})
    provider_name = llm_config.get("provider")
    provider_config = config.get("llm_providers", {}).get(provider_name)
    if not provider_config:
        raise ValueError(f"Provider '{provider_name}' not configured in global.yaml")

    client = get_llm_client(provider_config)
    model = llm_config.get("model")
    temperature = llm_config.get("temperature", 0.0)
    max_tokens = llm_config.get("max_tokens", 150)

    prompts = query_expansion_config.get("prompts", {})
    techniques = query_expansion_config.get("techniques", {})
    expanded_queries = [query]

    if techniques.get("rewrite", {}).get("enabled"):
        if prompt_template := prompts.get("rewrite"):
            prompt = prompt_template.format(query=query)
            rewritten = rewrite_query(prompt, client, model, temperature, max_tokens)
            expanded_queries.append(rewritten)

    if techniques.get("hyde", {}).get("enabled"):
        if prompt_template := prompts.get("hyde"):
            prompt = prompt_template.format(query=query)
            hypothetical_doc = generate_hypothetical_document(
                prompt, client, model, temperature, max_tokens
            )
            expanded_queries.append(hypothetical_doc)

    if techniques.get("multi_query", {}).get("enabled"):
        if prompt_template := prompts.get("multi_query"):
            num_variants = techniques.get("multi_query", {}).get("num_variants", 4)
            prompt = prompt_template.format(query=query, num_variants=num_variants)
            variants = expand_multi_query(
                prompt, client, model, temperature, max_tokens
            )
            expanded_queries.extend(variants)

    if techniques.get("step_back", {}).get("enabled"):
        if prompt_template := prompts.get("step_back"):
            prompt = prompt_template.format(query=query)
            step_back_question = generate_step_back_question(
                prompt, client, model, temperature, max_tokens
            )
            expanded_queries.append(step_back_question)

    if query_expansion_config.get("cache_enabled"):
        _query_cache[query] = expanded_queries

    return expanded_queries


# =============================================================================
# ADVANCED FEATURES - Query Decomposition & Routing
# =============================================================================


class QueryDecomposer:
    """Décomposeur de queries complexes en sous-questions.

    Pour queries multi-hop nécessitant plusieurs étapes de raisonnement.
    Exemple: "Compare X and Y" → ["What is X?", "What is Y?", "How do X and Y differ?"]

    Attributes:
        config: Configuration de la décomposition.
        client: Client LLM pour génération.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise le décomposeur.

        Args:
            config: Configuration complète depuis YAML.
        """
        self.config = config.get("query_decomposition", {})
        self.client: openai.OpenAI | None = None
        self._initialize_client(config)

    def _initialize_client(self, config: dict[str, Any]) -> None:
        """Initialise le client LLM."""
        if not self.config.get("enabled", False):
            return

        llm_config = self.config.get("llm", {})
        # Check if llm is nested under method (v2 style)
        if not llm_config:
             method_config = self.config.get("method", {})
             if method_config.get("type") == "llm":
                 llm_config = method_config.get("llm", {})

        provider_name = llm_config.get("provider", "ollama")
        provider_config = config.get("llm_providers", {}).get(provider_name)

        if provider_config:
            self.client = get_llm_client(provider_config)

    def decompose(self, query: str) -> list[str]:
        """Décompose une query complexe en sous-questions.

        Args:
            query: Query originale.

        Returns:
            Liste de sous-questions (inclut query originale si pas de décomposition).
        """
        if not self.config.get("enabled", False) or self.client is None:
            return [query]

        # Déterminer si décomposition nécessaire
        if not self._needs_decomposition(query):
            return [query]

        # Décomposer avec LLM
        sub_questions = self._decompose_with_llm(query)

        # Toujours inclure query originale
        if query not in sub_questions:
            sub_questions.insert(0, query)

        return sub_questions

    def _needs_decomposition(self, query: str) -> bool:
        """Vérifie si la query nécessite une décomposition.

        Args:
            query: Query à analyser.

        Returns:
            True si décomposition recommandée.
        """
        query_lower = query.lower()

        # Mots-clés indiquant une query complexe
        complex_keywords = [
            "compare",
            "difference between",
            "versus",
            "vs",
            "how does",
            "explain how",
            "what happens when",
            "relationship between",
            "impact of",
            "effect of",
            "why does",
            "step by step",
        ]

        # Présence de "and" ou "or" multiples
        connectors = self.config.get("trigger", {}).get("detect_connectors", ["and", "or"])
        has_multiple_parts = any(query_lower.count(f" {c} ") > 0 for c in connectors)

        # Présence de mots-clés complexes
        has_complex_keywords = any(kw in query_lower for kw in complex_keywords)

        # Questions multiples
        has_multiple_questions = query.count("?") > 1

        return has_multiple_parts or has_complex_keywords or has_multiple_questions

    def _decompose_with_llm(self, query: str) -> list[str]:
        """Décompose avec LLM.

        Args:
            query: Query à décomposer.

        Returns:
            Liste de sous-questions.
        """
        if self.client is None:
            return [query]

        # Prompt de décomposition
        prompt_template = self.config.get(
            "prompt_template",
            """Break down this complex question into simpler sub-questions that need to be answered step-by-step:

Question: {query}

Provide 2-4 sub-questions, one per line, numbered 1. 2. 3. etc.
Sub-questions:""",
        )

        prompt = prompt_template.format(query=query)

        llm_config = self.config.get("llm", {})
        # Check if llm is nested under method (v2 style)
        if not llm_config:
             method_config = self.config.get("method", {})
             if method_config.get("type") == "llm":
                 llm_config = method_config.get("llm", {})

        model = llm_config.get("model", "llama3")
        temperature = llm_config.get("temperature", 0.0)
        max_tokens = llm_config.get("max_tokens", 200)

        try:
            response = _create_chat_completion(
                prompt, self.client, model, temperature, max_tokens
            )

            # Parser réponse (format: "1. Question 1\n2. Question 2...")
            sub_questions = []
            for line in response.split("\n"):
                line = line.strip()
                # Enlever numérotation
                if line and (line[0].isdigit() or line.startswith("-")):
                    # Enlever "1. " ou "- "
                    question = line.lstrip("0123456789.-) ").strip()
                    if question and len(question) > 10:  # Minimum longueur
                        sub_questions.append(question)

            return sub_questions if sub_questions else [query]

        except Exception as e:
            print(f"⚠️ Query decomposition failed: {e}")
            return [query]


class QueryRouter:
    """Router de queries selon leur type.

    Route les queries vers différentes stratégies selon :
    - Type de question (factual, analytical, comparative, opinion)
    - Domaine (technical, business, general)
    - Complexité

    Attributes:
        config: Configuration du routing.
        client: Client LLM pour classification (optionnel).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise le router.

        Args:
            config: Configuration complète depuis YAML.
        """
        self.config = config.get("query_understanding", {})
        self.client: openai.OpenAI | None = None
        self._initialize_client(config)

    def _initialize_client(self, config: dict[str, Any]) -> None:
        """Initialise le client LLM."""
        if not self.config.get("enabled", False):
            return

        # In v2, routing is part of query_understanding -> type_classification
        # But for backward compatibility or specific routing logic, we might need to adapt.
        # The v2 config has type_classification.
        classification_config = self.config.get("type_classification", {})
        if not classification_config.get("enabled", False):
            return

        routing_method = classification_config.get("classifier", "heuristic")
        if routing_method != "llm":
            return

        llm_config = self.config.get("llm", {})
        provider_name = llm_config.get("provider", "ollama")
        provider_config = config.get("llm_providers", {}).get(provider_name)

        if provider_config:
            self.client = get_llm_client(provider_config)

    def route(self, query: str) -> dict[str, Any]:
        """Route la query et retourne stratégie recommandée.

        Args:
            query: Query à router.

        Returns:
            Dict avec query_type, domain, strategy, confidence.
        """
        classification_config = self.config.get("type_classification", {})
        if not classification_config.get("enabled", False):
            return {
                "query_type": "general",
                "domain": "general",
                "strategy": "standard",
                "confidence": 1.0,
            }

        routing_method = classification_config.get("classifier", "heuristic")

        if routing_method == "heuristic":
            return self._route_heuristic(query)
        elif routing_method == "llm" and self.client is not None:
            return self._route_llm(query)
        else:
            return self._route_heuristic(query)

    def _route_heuristic(self, query: str) -> dict[str, Any]:
        """Routing par heuristiques simples.

        Args:
            query: Query à router.

        Returns:
            Résultat du routing.
        """
        query_lower = query.lower()

        # Use rules from config if available
        classification_config = self.config.get("type_classification", {})
        heuristic_config = classification_config.get("heuristic", {})
        rules = heuristic_config.get("rules", {})

        if rules:
            import re
            for type_name, rule in rules.items():
                patterns = rule.get("patterns", [])
                for pattern in patterns:
                    if re.search(pattern, query_lower):
                        return {
                            "query_type": type_name,
                            "domain": "general", # Domain logic can be improved similarly
                            "strategy": "standard",
                            "confidence": 0.8
                        }
        
        # Fallback to hardcoded if no rules match or no rules provided
        query_type = "factual"  # Default

        if any(
            word in query_lower
            for word in ["why", "how does", "explain", "what causes", "pourquoi", "comment"]
        ):
            query_type = "analytical"
        elif any(
            word in query_lower
            for word in ["compare", "difference", "vs", "versus", "better"]
        ):
            query_type = "comparative"
        elif any(
            word in query_lower for word in ["opinion", "think", "should", "recommend"]
        ):
            query_type = "opinion"

        # Détecter domaine (simplifié)
        domain = "general"

        technical_keywords = [
            "algorithm",
            "code",
            "function",
            "api",
            "database",
            "server",
        ]
        if any(kw in query_lower for kw in technical_keywords):
            domain = "technical"

        business_keywords = [
            "revenue",
            "profit",
            "market",
            "customer",
            "strategy",
            "business",
        ]
        if any(kw in query_lower for kw in business_keywords):
            domain = "business"

        # Stratégie selon type
        strategy_map = self.config.get("strategies", {})
        strategy = strategy_map.get(query_type, "standard")

        return {
            "query_type": query_type,
            "domain": domain,
            "strategy": strategy,
            "confidence": 0.8,  # Heuristique = confiance moyenne
        }

    def _route_llm(self, query: str) -> dict[str, Any]:
        """Routing avec LLM (plus précis).

        Args:
            query: Query à router.

        Returns:
            Résultat du routing.
        """
        if self.client is None:
            return self._route_heuristic(query)

        prompt_template = self.config.get(
            "prompt_template",
            """Classify this query:

Query: "{query}"

Provide classification:
Type: [factual/analytical/comparative/opinion]
Domain: [technical/business/general]
Strategy: [simple/standard/complex]

Format: Type: X, Domain: Y, Strategy: Z""",
        )

        prompt = prompt_template.format(query=query)

        llm_config = self.config.get("llm", {})
        model = llm_config.get("model", "llama3")
        temperature = llm_config.get("temperature", 0.0)
        max_tokens = llm_config.get("max_tokens", 100)

        try:
            response = _create_chat_completion(
                prompt, self.client, model, temperature, max_tokens
            )

            # Parser réponse
            result = {
                "query_type": "factual",
                "domain": "general",
                "strategy": "standard",
                "confidence": 0.9,
            }

            response_lower = response.lower()

            # Extraire type
            if "analytical" in response_lower:
                result["query_type"] = "analytical"
            elif "comparative" in response_lower:
                result["query_type"] = "comparative"
            elif "opinion" in response_lower:
                result["query_type"] = "opinion"

            # Extraire domaine
            if "technical" in response_lower:
                result["domain"] = "technical"
            elif "business" in response_lower:
                result["domain"] = "business"

            # Extraire stratégie
            if "simple" in response_lower:
                result["strategy"] = "simple"
            elif "complex" in response_lower:
                result["strategy"] = "complex"

            return result

        except Exception as e:
            print(f"⚠️ LLM routing failed: {e}. Fallback to heuristic.")
            return self._route_heuristic(query)


if __name__ == "__main__":
    # Example of how to run the query processing.
    # Ensure the OPENAI_API_KEY environment variable is set if using OpenAI.
    # export OPENAI_API_KEY="your_key_here"
    step_config = load_config("01_embedding_v2", "config")
    is_openai = (
        step_config.get("query_expansion", {}).get("llm", {}).get("provider")
        == "openai"
    )

    if is_openai and not os.getenv("OPENAI_API_KEY"):
        print("Please set the OPENAI_API_KEY environment variable.")
    else:
        original_query = "Quels sont les bénéfices d'un modèle RAG ?"

        print("--- First call (should process and cache) ---")
        processed_queries_1 = process_query(original_query, step_config)
        print(f"Original query: {original_query}")
        print(f"Processed queries: {processed_queries_1}")

        print("\n--- Second call (should hit cache) ---\n")
        processed_queries_2 = process_query(original_query, step_config)
        print(f"Original query: {original_query}")
        print(f"Processed queries: {processed_queries_2}")
