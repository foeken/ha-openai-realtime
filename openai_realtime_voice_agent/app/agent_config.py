"""Agent profile and wake-word routing configuration."""
import json
import logging
import re
from dataclasses import dataclass
from typing import Mapping, Optional

logger = logging.getLogger(__name__)


def normalize_route_key(value: Optional[str]) -> Optional[str]:
    """Normalize wake words and agent names for routing lookups."""
    if not value:
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    normalized = normalized.strip("_")
    return normalized or None


def _parse_tools(value) -> Optional[frozenset[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value or value in ("*", "all"):
            return None
        tools = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        tools = [str(part).strip() for part in value]
    else:
        logger.warning("Ignoring invalid tools value %r", value)
        return None

    filtered = frozenset(tool for tool in tools if tool)
    return filtered or None


@dataclass(frozen=True)
class AgentProfile:
    """Realtime agent settings selected per satellite session."""

    name: str
    instructions: str
    voice: str
    tools: Optional[frozenset[str]] = None

    def allows_tool(self, tool_name: str) -> bool:
        return self.tools is None or tool_name in self.tools


class AgentRegistry:
    """Resolve agent profiles from explicit agent names or wake words."""

    def __init__(
        self,
        profiles: Mapping[str, AgentProfile],
        wake_word_map: Mapping[str, str],
        default_agent: str,
    ):
        self._profiles = dict(profiles)
        self._wake_word_map = dict(wake_word_map)
        self._default_agent = normalize_route_key(default_agent) or "default"

    @property
    def profiles(self) -> Mapping[str, AgentProfile]:
        return self._profiles

    @property
    def wake_word_map(self) -> Mapping[str, str]:
        return self._wake_word_map

    def resolve(
        self,
        *,
        agent_name: Optional[str] = None,
        wake_word: Optional[str] = None,
    ) -> AgentProfile:
        resolved_name = normalize_route_key(agent_name)
        if not resolved_name and wake_word:
            resolved_name = self._wake_word_map.get(normalize_route_key(wake_word))
        if not resolved_name:
            resolved_name = self._default_agent

        if resolved_name in self._profiles:
            return self._profiles[resolved_name]

        default_profile = self._profiles[self._default_agent]
        logger.info("Using default agent settings for unknown agent route '%s'", resolved_name)
        return AgentProfile(
            name=resolved_name,
            instructions=default_profile.instructions,
            voice=default_profile.voice,
            tools=default_profile.tools,
        )


def _parse_mapping(raw: Optional[str]) -> dict[str, str]:
    if not raw:
        return {}

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return {
            normalize_route_key(key): normalize_route_key(value)
            for key, value in parsed.items()
            if normalize_route_key(key) and normalize_route_key(value)
        }

    mapping: dict[str, str] = {}
    for item in raw.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        normalized_key = normalize_route_key(key)
        normalized_value = normalize_route_key(value)
        if normalized_key and normalized_value:
            mapping[normalized_key] = normalized_value
    return mapping


def load_agent_registry(
    env: Mapping[str, str],
    *,
    default_instructions: str,
    default_voice: str,
) -> AgentRegistry:
    """Load agent profiles from environment variables.

    Supported env:
    - DEFAULT_AGENT: default route name.
    - AGENTS_JSON / AGENT_CONFIG_JSON: JSON object of agent profiles.
    - WAKE_WORD_AGENT_MAP: JSON object or comma list, e.g. hey_jarvis=jarvis.
    """
    default_agent = normalize_route_key(env.get("DEFAULT_AGENT")) or "default"
    profiles: dict[str, AgentProfile] = {
        default_agent: AgentProfile(
            name=default_agent,
            instructions=default_instructions,
            voice=default_voice,
        )
    }

    raw_agents = env.get("AGENTS_JSON") or env.get("AGENT_CONFIG_JSON")
    if raw_agents:
        try:
            parsed_agents = json.loads(raw_agents)
        except json.JSONDecodeError as exc:
            logger.warning("Ignoring invalid AGENTS_JSON: %s", exc)
            parsed_agents = None

        if isinstance(parsed_agents, dict):
            agent_items = parsed_agents.get("agents", parsed_agents)
            if isinstance(agent_items, dict):
                for raw_name, config in agent_items.items():
                    name = normalize_route_key(raw_name)
                    if not name or not isinstance(config, dict):
                        continue
                    profiles[name] = AgentProfile(
                        name=name,
                        instructions=config.get("instructions") or default_instructions,
                        voice=(config.get("voice") or default_voice).lower(),
                        tools=_parse_tools(config.get("tools")),
                    )

    wake_word_map = {
        "hey_jarvis": "jarvis",
        "okay_nabu": "nabu",
    }
    wake_word_map.update(_parse_mapping(env.get("WAKE_WORD_AGENT_MAP")))

    if default_agent not in profiles:
        first_profile = next(iter(profiles.values()))
        profiles[default_agent] = AgentProfile(
            name=default_agent,
            instructions=first_profile.instructions,
            voice=first_profile.voice,
            tools=first_profile.tools,
        )

    logger.info(
        "Loaded agent routes: default=%s profiles=%s wake_words=%s",
        default_agent,
        sorted(profiles.keys()),
        wake_word_map,
    )
    return AgentRegistry(profiles, wake_word_map, default_agent)
