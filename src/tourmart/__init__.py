"""TourMart — platform-governance benchmark for language-mediated tourism e-commerce markets.

Block E0: simulator primitives + MILP welfare oracle + exhaustive search oracle.
"""
from .primitives import (
    TravelerArchetype,
    Traveler,
    Hotel,
    Airline,
    Bundle,
    Market,
    Allocation,
)
from .scenarios import (
    generate_small_market,
    generate_medium_market,
    generate_scenario_bank,
)
from .oracle import milp_oracle, exhaustive_oracle, package_price
from .preference_proxy import compute_observable_prior
from .prompts import (
    Condition, OBJECTIVE_BLOCKS, SYSTEM_PROMPT,
    render_user_prompt, parse_ota_output, validate_ota_schema,
)
from .llm_backends import LLMBackend, MockLLM, VLLMBackend
from .llm_agents import (
    DEFAULT_ACCEPTANCE_THRESHOLDS, OTAAgent, OTAResult,
    TravelerAgent, TravelerDecision,
)
from .e2_episode import EpisodeResult, run_episode
from .resume import EpisodeKey, append_episode, load_done_keys, make_key

__all__ = [
    # primitives
    "TravelerArchetype", "Traveler", "Hotel", "Airline", "Bundle", "Market", "Allocation",
    # scenarios
    "generate_small_market", "generate_medium_market", "generate_scenario_bank",
    # oracle
    "milp_oracle", "exhaustive_oracle", "package_price",
    # preference proxy
    "compute_observable_prior",
    # prompts
    "Condition", "OBJECTIVE_BLOCKS", "SYSTEM_PROMPT",
    "render_user_prompt", "parse_ota_output", "validate_ota_schema",
    # llm backends
    "LLMBackend", "MockLLM", "VLLMBackend",
    # llm agents
    "DEFAULT_ACCEPTANCE_THRESHOLDS", "OTAAgent", "OTAResult",
    "TravelerAgent", "TravelerDecision",
    # episode
    "EpisodeResult", "run_episode",
    # resume
    "EpisodeKey", "append_episode", "load_done_keys", "make_key",
]
