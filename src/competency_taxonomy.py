"""Reusable deterministic competency taxonomy.

These patterns are data used by extraction/RoleSpec generation. They are not
LLM prompts and contain no executable ranking logic.
"""

from __future__ import annotations

from src.rolespec import COMPETENCY_TAXONOMY


COMPETENCY_PATTERNS: dict[str, dict[str, tuple[str, ...]]] = {
    "ai_ml_engineering": {
        "phrase_patterns": ("machine learning", "ai system", "model training", "nlp", "llm", "applied ml"),
        "title_patterns": ("ai engineer", "machine learning engineer", "ml engineer", "applied scientist", "nlp engineer"),
        "skill_patterns": ("python", "pytorch", "tensorflow", "scikit-learn", "transformers", "mlflow"),
        "career_history_evidence_patterns": ("trained", "deployed model", "feature pipeline", "model monitoring", "production ml"),
        "negative_confusing_patterns": ("ai enthusiast", "prompt user", "course project", "tutorial"),
    },
    "retrieval_search": {
        "phrase_patterns": ("search", "retrieval", "semantic search", "information retrieval", "query expansion"),
        "title_patterns": ("search engineer", "retrieval engineer", "ml engineer"),
        "skill_patterns": ("elasticsearch", "opensearch", "solr", "bm25", "sentence transformers"),
        "career_history_evidence_patterns": ("search relevance", "retrieval system", "query understanding", "search ranking"),
        "negative_confusing_patterns": ("web search user", "basic keyword search"),
    },
    "ranking_recommendation": {
        "phrase_patterns": ("recommendation", "recommender", "ranking", "personalization", "discovery feed"),
        "title_patterns": ("recommendation systems engineer", "ranking engineer", "search engineer"),
        "skill_patterns": ("learning to rank", "ltr", "collaborative filtering", "ranker"),
        "career_history_evidence_patterns": ("recommendation system", "ranking layer", "personalized feed", "relevance ranking"),
        "negative_confusing_patterns": ("ranked reports", "recommended tools"),
    },
    "ranking_evaluation": {
        "phrase_patterns": ("ndcg", "mrr", "map", "recall@", "precision@", "a/b testing", "offline-online"),
        "title_patterns": ("search engineer", "recommendation systems engineer", "data scientist"),
        "skill_patterns": ("experimentation", "ranking metrics", "relevance labeling", "ab testing"),
        "career_history_evidence_patterns": ("offline evaluation", "online experiment", "relevance judgments", "human labels"),
        "negative_confusing_patterns": ("unit testing only", "manual qa only"),
    },
    "vector_hybrid_search": {
        "phrase_patterns": ("dense retrieval", "hybrid search", "vector index", "embedding retrieval", "ann"),
        "title_patterns": ("search engineer", "ai engineer", "ml engineer"),
        "skill_patterns": ("faiss", "pinecone", "weaviate", "qdrant", "milvus", "pgvector", "opensearch"),
        "career_history_evidence_patterns": ("bm25 + dense", "embedding index", "nearest neighbor", "index refresh"),
        "negative_confusing_patterns": ("vector graphics", "vector illustration"),
    },
    "production_ml_systems": {
        "phrase_patterns": ("production", "deployed", "monitoring", "retraining", "p95 latency", "real users"),
        "title_patterns": ("ml engineer", "ai engineer", "mlops engineer"),
        "skill_patterns": ("mlflow", "kubeflow", "bentoml", "docker", "kubernetes"),
        "career_history_evidence_patterns": ("shipped", "operated", "on-call", "scaled", "deployed model"),
        "negative_confusing_patterns": ("notebook only", "academic prototype"),
    },
    "python_engineering": {
        "phrase_patterns": ("python", "fastapi", "pandas", "numpy", "pytest"),
        "title_patterns": ("python developer", "software engineer", "ml engineer"),
        "skill_patterns": ("python", "fastapi", "django", "flask", "pytest"),
        "career_history_evidence_patterns": ("built api", "implemented pipeline", "python service"),
        "negative_confusing_patterns": ("beginner python",),
    },
    "backend_systems": {
        "phrase_patterns": ("api", "microservice", "distributed system", "database", "latency", "scalability"),
        "title_patterns": ("backend engineer", "software engineer", "platform engineer"),
        "skill_patterns": ("java", "go", "python", "postgres", "redis", "kafka", "fastapi"),
        "career_history_evidence_patterns": ("designed api", "built service", "optimized latency", "owned backend"),
        "negative_confusing_patterns": ("frontend only",),
    },
    "data_engineering": {
        "phrase_patterns": ("etl", "data pipeline", "warehouse", "spark", "airflow", "kafka"),
        "title_patterns": ("data engineer", "analytics engineer", "ml engineer"),
        "skill_patterns": ("spark", "airflow", "dbt", "snowflake", "kafka", "beam"),
        "career_history_evidence_patterns": ("feature pipeline", "batch pipeline", "streaming pipeline"),
        "negative_confusing_patterns": ("spreadsheet reporting",),
    },
    "frontend_engineering": {
        "phrase_patterns": ("frontend", "ui", "react", "typescript", "web app"),
        "title_patterns": ("frontend engineer", "ui engineer", "web developer"),
        "skill_patterns": ("react", "typescript", "javascript", "css", "html", "vue"),
        "career_history_evidence_patterns": ("built interface", "component library", "frontend architecture"),
        "negative_confusing_patterns": ("cms editing",),
    },
    "devops_cloud": {
        "phrase_patterns": ("cloud", "kubernetes", "terraform", "ci/cd", "observability"),
        "title_patterns": ("devops engineer", "cloud engineer", "platform engineer", "sre"),
        "skill_patterns": ("aws", "gcp", "azure", "terraform", "docker", "kubernetes"),
        "career_history_evidence_patterns": ("deployment pipeline", "infra", "monitoring", "incident response"),
        "negative_confusing_patterns": ("basic hosting",),
    },
    "product_management": {
        "phrase_patterns": ("roadmap", "stakeholder", "product strategy", "metrics", "launch", "experimentation"),
        "title_patterns": ("product manager", "senior product manager", "product owner"),
        "skill_patterns": ("roadmapping", "analytics", "experimentation", "user research", "jira"),
        "career_history_evidence_patterns": ("owned roadmap", "launched product", "defined metrics", "prioritized backlog"),
        "negative_confusing_patterns": ("project coordinator",),
    },
    "sales": {
        "phrase_patterns": ("sales", "quota", "pipeline", "crm", "account executive"),
        "title_patterns": ("sales manager", "account executive", "business development"),
        "skill_patterns": ("salesforce", "crm", "negotiation", "prospecting"),
        "career_history_evidence_patterns": ("closed deals", "owned quota", "managed pipeline"),
        "negative_confusing_patterns": ("salesforce developer"),
    },
    "marketing": {
        "phrase_patterns": ("marketing", "campaign", "seo", "content", "growth"),
        "title_patterns": ("marketing manager", "growth marketer", "content marketer"),
        "skill_patterns": ("seo", "sem", "campaigns", "analytics", "hubspot"),
        "career_history_evidence_patterns": ("ran campaign", "improved conversion", "content strategy"),
        "negative_confusing_patterns": ("marketplace engineer"),
    },
    "design": {
        "phrase_patterns": ("design", "ux", "ui", "prototype", "user research"),
        "title_patterns": ("designer", "product designer", "ux designer", "ui designer"),
        "skill_patterns": ("figma", "sketch", "prototyping", "wireframes"),
        "career_history_evidence_patterns": ("designed flows", "user testing", "design system"),
        "negative_confusing_patterns": ("system design backend"),
    },
    "management": {
        "phrase_patterns": ("managed team", "mentored", "hiring", "strategy", "leadership"),
        "title_patterns": ("manager", "lead", "head", "director"),
        "skill_patterns": ("people management", "planning", "stakeholder management"),
        "career_history_evidence_patterns": ("managed engineers", "led team", "owned delivery"),
        "negative_confusing_patterns": ("managed scripts",),
    },
    "research": {
        "phrase_patterns": ("research", "publication", "paper", "experiment", "academic"),
        "title_patterns": ("research engineer", "research scientist", "research intern"),
        "skill_patterns": ("pytorch", "experimentation", "papers", "statistics"),
        "career_history_evidence_patterns": ("published", "prototype", "lab", "benchmark"),
        "negative_confusing_patterns": ("no deployment", "pure research"),
    },
    "consulting_services": {
        "phrase_patterns": ("consulting", "client delivery", "it services", "vendor", "implementation partner"),
        "title_patterns": ("consultant", "technology consultant", "service delivery"),
        "skill_patterns": ("client management", "delivery", "implementation"),
        "career_history_evidence_patterns": ("client project", "services", "consulting engagement"),
        "negative_confusing_patterns": ("product ownership unclear",),
    },
    "open_source_validation": {
        "phrase_patterns": ("open source", "github", "pull request", "maintainer", "oss"),
        "title_patterns": ("maintainer", "open source engineer"),
        "skill_patterns": ("github", "oss", "contributions"),
        "career_history_evidence_patterns": ("merged pull requests", "maintained library", "community"),
        "negative_confusing_patterns": ("private coursework repo",),
    },
}


missing = set(COMPETENCY_TAXONOMY) - set(COMPETENCY_PATTERNS)
if missing:
    raise RuntimeError(f"Taxonomy missing competency patterns: {sorted(missing)}")
