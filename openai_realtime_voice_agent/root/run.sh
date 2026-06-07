#!/usr/bin/with-contenv bashio
set -e

# Get configuration
OPENAI_API_KEY=$(bashio::config 'openai_api_key')
WEBSOCKET_PORT=$(bashio::config 'websocket_port')
HA_MCP_URL=$(bashio::config 'ha_mcp_url')
LONGLIVED_TOKEN=$(bashio::config 'longlived_token')

# Get turn detection settings
VAD_THRESHOLD=$(bashio::config 'vad_threshold')
VAD_PREFIX_PADDING_MS=$(bashio::config 'vad_prefix_padding_ms')
VAD_SILENCE_DURATION_MS=$(bashio::config 'vad_silence_duration_ms')
VAD_IDLE_TIMEOUT_MS=$(bashio::config 'vad_idle_timeout_ms')

# Get OpenAI Realtime settings
OPENAI_REALTIME_MODEL=$(bashio::config 'openai_realtime_model')
OPENAI_REALTIME_VOICE=$(bashio::config 'openai_realtime_voice')
OPENAI_TRANSCRIPTION_MODEL=$(bashio::config 'openai_transcription_model')
OPENAI_NOISE_REDUCTION=$(bashio::config 'openai_noise_reduction')

# Get instructions
INSTRUCTIONS=$(bashio::config 'instructions')

# Get agent routing settings
DEFAULT_AGENT=$(bashio::config 'default_agent')
WAKE_WORD_AGENT_MAP=$(bashio::config 'wake_word_agent_map')
AGENT_PROMPTS_JSON=$(bashio::config 'agent_prompts_json')
AGENTS_JSON=$(bashio::config 'agents_json')
CLIENT_METADATA_TIMEOUT_SECONDS=$(bashio::config 'client_metadata_timeout_seconds')
AUTO_DISCONNECT_AFTER_RESPONSE_SECONDS=$(bashio::config 'auto_disconnect_after_response_seconds')

# Get session management settings
SESSION_REUSE_TIMEOUT_SECONDS=$(bashio::config 'session_reuse_timeout_seconds')

# Get audio recording setting
ENABLE_RECORDING=$(bashio::config 'enable_recording')

# Validate required configuration
if [ -z "$OPENAI_API_KEY" ]; then
    bashio::log.error "OPENAI_API_KEY is required but not set"
    exit 1
fi

# Export environment variables
export OPENAI_API_KEY
export WEBSOCKET_PORT
export LONGLIVED_TOKEN

# Export turn detection settings
export VAD_THRESHOLD
export VAD_PREFIX_PADDING_MS
export VAD_SILENCE_DURATION_MS
export VAD_IDLE_TIMEOUT_MS

# Export OpenAI Realtime settings
export OPENAI_REALTIME_MODEL
export OPENAI_REALTIME_VOICE
export OPENAI_TRANSCRIPTION_MODEL
export OPENAI_NOISE_REDUCTION

# Export instructions
export INSTRUCTIONS

# Export agent routing settings
export DEFAULT_AGENT
export WAKE_WORD_AGENT_MAP
export AGENT_PROMPTS_JSON
export AGENTS_JSON
export CLIENT_METADATA_TIMEOUT_SECONDS
export AUTO_DISCONNECT_AFTER_RESPONSE_SECONDS

# Export session management settings
export SESSION_REUSE_TIMEOUT_SECONDS

# Export audio recording setting
export ENABLE_RECORDING

# Export HA_MCP_URL if set (empty string means use default in main.py)
if [ -n "$HA_MCP_URL" ]; then
    export HA_MCP_URL
fi

# SUPERVISOR_TOKEN is automatically provided by Home Assistant when homeassistant_api: true

# Start the application
export PYTHONUNBUFFERED=1
exec python3 -m app.main
