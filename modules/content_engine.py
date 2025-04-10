import json
import re
import logging
import asyncio
from typing import Dict, Any, Optional
import os

# Import core objects and services
from core.article_plan import ArticlePlan
from core.generated_article import GeneratedArticle
from modules.llm_service import LLMService, QuotaError # Import LLMService and potential exceptions

# Configure logging (consider using a shared config later)
import config
logging.basicConfig(
    level=config.LOG_LEVEL,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger(__name__)

class ContentEngine:
    """
    Orchestrates the generation of article plans and content using an LLM service.
    Accepts structured input data (keyword, competitor analysis, research) and parameters.
    """

    def __init__(self, llm_service: LLMService):
        """
        Initializes the content engine.

        Args:
            llm_service: An instance of the LLMService for content generation.
        """
        if not isinstance(llm_service, LLMService):
            raise TypeError("llm_service must be an instance of LLMService")
        self.llm_service = llm_service

        # Define prompts (can be loaded from external files later)
        self.prompts = {
            "plan_generation": """
Tu es un expert SEO français chargé de créer un plan d'article CONCIS et COHÉRENT pour le mot-clé: "{keyword}".
Ton plan doit être basé EXCLUSIVEMENT sur les informations fournies ci-dessous concernant l'analyse des concurrents et les recherches complémentaires.

# ANALYSE CONCURRENTIELLE (Résumé)
{competitor_analysis_summary}

# RECHERCHES COMPLÉMENTAIRES (Résumé)
{research_summary}

## DIRECTIVES STRICTES POUR LE PLAN
1.  **Base Factuelle:** Le plan doit refléter les thèmes, structures et informations clés identifiés dans les analyses fournies. NE RIEN INVENTER.
2.  **Structure Logique:** Organise les sections (H2, H3) de manière logique et fluide. Vise {h2_count} sections H2 principales, sauf si l'analyse justifie un nombre différent. Inclus des H3 pertinents si demandé ({include_h3}).
3.  **Cohérence:** Assure une progression naturelle des idées. Évite les redondances.
4.  **Briefs Utiles:** Pour chaque section H2/H3, rédige un bref descriptif indiquant les points clés à couvrir, basés sur les analyses fournies.

## FORMAT DE SORTIE (JSON Strict)
```json
{{
  "keyword": "{keyword}",
  "metatitle": "Méta-titre optimisé basé sur l'analyse (50-60 caractères)",
  "metadescription": "Méta-description engageante basée sur l'analyse (150-160 caractères)",
  "h1": "Titre H1 principal, clair et pertinent",
  "backstory": "Court résumé du contexte et de l'objectif de l'article, basé sur les analyses.",
  "data": {{
    "secondary_keywords": ["Liste", "de mots-clés", "secondaires pertinents", "issus de l'analyse"],
    "key_insights": ["Point clé 1 de l'analyse", "Point clé 2"]
  }},
  "plan": [
    {{
      "h2": "Titre H2 de la Section 1",
      "h3": ["Titre H3.1", "Titre H3.2"], // Inclure seulement si include_h3 est True et pertinent
      "brief": "Points clés à couvrir dans cette section, basés sur les analyses."
    }},
    // ... autres sections H2 ...
  ]
}}
```

Produis UNIQUEMENT le bloc JSON valide, sans texte avant ou après.
""",
            "article_generation": """
# MISSION: RÉDIGER UN ARTICLE SEO PREMIUM

Tu es un rédacteur expert SEO. Ton objectif est de rédiger un article COMPLET et ENGAGEANT basé STRICTEMENT sur le plan détaillé et les informations de recherche fournis.

## PARAMÈTRES DE RÉDACTION
- **Ton:** {tone}
- **Longueur Cible:** Environ {word_count} mots (adapte si nécessaire pour couvrir le plan)
- **Instructions Spécifiques:** {content_instructions}

## PLAN DÉTAILLÉ À SUIVRE (Structure et Briefs)
```json
{plan_json_string}
```

## INFORMATIONS DE RECHERCHE COMPLÉMENTAIRES
Utilise ces informations pour enrichir chaque section pertinente de l'article avec des détails factuels et des données récentes.
```
{research_summary}
```

## EXIGENCES DE QUALITÉ
1.  **Respect du Plan:** Suis la structure H1/H2/H3 et les briefs de chaque section à la lettre.
2.  **Intégration des Recherches:** Incorpore NATURELLEMENT les informations de recherche dans les sections appropriées. Cite les sources si fournies dans les recherches.
3.  **Contenu Factuel:** NE RIEN INVENTER. Base-toi uniquement sur le plan et les recherches.
4.  **Lisibilité:** Paragraphes courts (3-5 lignes), langage clair, transitions fluides.
5.  **SEO:** Intègre les mots-clés secondaires (listés dans le plan) de manière naturelle.
6.  **Format:** Utilise le Markdown standard (Titres #, ##, ###; Listes - ou *; Gras **texte**).

## LIVRABLE
Produis UNIQUEMENT l'article complet au format Markdown. Commence directement par le titre H1. N'ajoute aucune introduction ou conclusion qui ne soit pas explicitement demandée dans le plan.
"""
        }

    async def generate_plan(
        self,
        keyword: str,
        model_settings: Dict[str, Any],
        competitor_analysis_summary: str = "Aucune analyse fournie.",
        research_summary: str = "Aucune recherche complémentaire fournie.",
        generation_params: Optional[Dict[str, Any]] = None
    ) -> ArticlePlan:
        """
        Generates an article plan using the LLM.
        """
        params = generation_params or {}
        h2_count = params.get("h2_count", 7)
        include_h3 = params.get("include_h3", True)

        logger.info(f"Generating plan for keyword: '{keyword}' with H2 count: {h2_count}, Include H3: {include_h3}")

        prompt = self.prompts["plan_generation"].format(
            keyword=keyword,
            competitor_analysis_summary=competitor_analysis_summary,
            research_summary=research_summary,
            h2_count=h2_count,
            include_h3=include_h3
        )

        plan_json_str = "" # Initialize for error logging
        try:
            # Call the LLM service - it now returns a dictionary
            response_dict = await self.llm_service.generate_content(
                prompt=prompt,
                provider=model_settings.get("provider"),
                model=model_settings.get("model"),
                temperature=model_settings.get("temperature", 0.5),
                max_tokens=model_settings.get("max_tokens", 2048)
            )

            # Extract the actual text content from the standardized response dictionary
            response_content = response_dict.get("content")
            if not isinstance(response_content, str):
                 logger.error(f"LLM service did not return a string in 'content'. Got: {type(response_content)}")
                 raise ValueError("Invalid content type received from LLM service.")

            # Extract JSON from the response content string
            json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
            if json_match:
                plan_json_str = json_match.group(1).strip()
            else:
                plan_json_str = response_content.strip()
                if not (plan_json_str.startswith('{') and plan_json_str.endswith('}')):
                     logger.error(f"LLM response content does not contain a valid JSON block. Content: {plan_json_str[:500]}...")
                     raise ValueError("LLM response does not contain a valid JSON block.")

            # Parse the JSON
            plan_data = json.loads(plan_json_str)

            # Basic validation
            if not isinstance(plan_data, dict) or "plan" not in plan_data or "h1" not in plan_data:
                logger.error(f"Generated plan JSON is missing required keys. Received: {plan_json_str}")
                raise ValueError("Generated plan JSON is missing required keys (h1, plan).")

            logger.info(f"Successfully generated plan for keyword: '{keyword}'")
            return ArticlePlan(data=plan_data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response from LLM: {e}")
            logger.error(f"Invalid JSON string received: {plan_json_str}")
            raise ValueError(f"LLM response was not valid JSON: {e}") from e
        except QuotaError as e:
             logger.error(f"LLM quota error during plan generation: {e}")
             raise
        except Exception as e:
            logger.error(f"Error generating plan for '{keyword}': {e}", exc_info=True) # Add exc_info for more details
            raise


    async def generate_article(
        self,
        article_plan: ArticlePlan,
        model_settings: Dict[str, Any],
        research_summary: str = "Aucune recherche complémentaire fournie.",
        generation_params: Optional[Dict[str, Any]] = None
    ) -> GeneratedArticle:
        """
        Generates article content based on a plan and research summary.
        """
        params = generation_params or {}
        tone = params.get("tone", "Informative and professional")
        word_count = params.get("word_count", 1500)
        content_instructions = params.get("content_instructions", "Ensure the article is well-structured and easy to read.")

        logger.info(f"Generating article for keyword: '{article_plan.get('keyword', 'N/A')}' with tone: {tone}, target words: {word_count}")

        try:
            plan_json_string = json.dumps(article_plan.data, indent=2, ensure_ascii=False)

            prompt = self.prompts["article_generation"].format(
                plan_json_string=plan_json_string,
                research_summary=research_summary,
                tone=tone,
                word_count=word_count,
                content_instructions=content_instructions
            )

            # Call the LLM service - it returns a dictionary
            response_dict = await self.llm_service.generate_content(
                prompt=prompt,
                provider=model_settings.get("provider"),
                model=model_settings.get("model"),
                temperature=model_settings.get("temperature", 0.7),
                max_tokens=model_settings.get("max_tokens", 4000)
            )

            # Extract the actual markdown content from the dictionary
            article_markdown = response_dict.get("content")
            if not isinstance(article_markdown, str):
                 logger.error(f"LLM service did not return a string in 'content' for article generation. Got: {type(article_markdown)}")
                 raise ValueError("Invalid content type received from LLM service for article generation.")


            # Basic cleanup
            article_markdown = re.sub(r'^```markdown\s*', '', article_markdown, flags=re.IGNORECASE).strip()
            article_markdown = re.sub(r'\s*```$', '', article_markdown).strip()

            logger.info(f"Successfully generated article for keyword: '{article_plan.get('keyword', 'N/A')}'")
            return GeneratedArticle(content=article_markdown)

        except QuotaError as e:
             logger.error(f"LLM quota error during article generation: {e}")
             raise
        except Exception as e:
            logger.error(f"Error generating article for keyword '{article_plan.get('keyword', 'N/A')}': {e}", exc_info=True) # Add exc_info
            raise


# Example Usage
if __name__ == "__main__":
    async def main():
        print("Testing ContentEngine...")
        from dotenv import load_dotenv
        load_dotenv()
        test_api_keys = {
            "openai": os.getenv("OPENAI_API_KEY"),
        }
        test_api_keys = {k: v for k, v in test_api_keys.items() if v}

        if not test_api_keys or "openai" not in test_api_keys:
            print("OpenAI API key needed for this test. Skipping.")
            return

        llm = LLMService(api_keys=test_api_keys)
        engine = ContentEngine(llm_service=llm)
        test_keyword = "rédacteur web seo"
        test_model_settings = {"provider": "OpenAI", "model": "gpt-3.5-turbo"}

        print(f"\n--- Generating Plan for: '{test_keyword}' ---")
        try:
            competitor_summary = "Focus on skills (writing, SEO) and tools."
            research_summary = "Growing demand for AI-specialized writers."
            plan_params = {"h2_count": 5, "include_h3": False}
            generated_plan = await engine.generate_plan(
                keyword=test_keyword, model_settings=test_model_settings,
                competitor_analysis_summary=competitor_summary, research_summary=research_summary,
                generation_params=plan_params
            )
            print("Plan generated successfully:")
            print(f"  H1: {generated_plan.get('h1')}")
            print(f"  Sections H2: {[s.get('h2') for s in generated_plan.get('plan', [])]}")

            print(f"\n--- Generating Article from Plan ---")
            try:
                article_params = {"tone": "Professional", "word_count": 300}
                generated_article = await engine.generate_article(
                    article_plan=generated_plan, model_settings=test_model_settings,
                    research_summary=research_summary, generation_params=article_params
                )
                print("Article generated successfully:")
                print(f"  Length: {len(generated_article.content)} characters")
                print(f"  Content Snippet:\n---\n{generated_article.content[:300]}...\n---")
            except Exception as e:
                print(f"Error during article generation test: {e}")
                import traceback; traceback.print_exc()
        except Exception as e:
            print(f"Error during plan generation test: {e}")
            import traceback; traceback.print_exc()

    if __name__ == "__main__":
        asyncio.run(main())
