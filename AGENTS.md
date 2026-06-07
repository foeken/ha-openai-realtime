# AGENTS.md

## Voice Agent Development Loop

- Run the server locally before deploying to Home Assistant; HA updates are slow, so use the local websocket target for iteration.
- When launching the local server from Codex, load `/Users/andre.foeken/Code/ha-openai-realtime/.env` with override enabled so repo secrets win over any inherited terminal environment variables.
- Keep the ESP client flashed against the local server URL while testing, then reflash it back to the Home Assistant add-on URL before handing the device back.
- When the user asks for a quick validation loop, flash local server URL, run the local server, trigger the device with Sonos TTS, confirm the command from server logs plus HA state, then reflash the HA URL version.
- For wake-word tests, say only `Hey Jarvis` first, then inspect server logs to confirm the trigger produced a client turn.
- Wait briefly for the websocket connection to appear before deciding the wake was missed; the ESP can connect a moment after the spoken wake word finishes.
- If the trigger still did not show up in the server logs and the ESP is still idle/disconnected, repeat only the wake word. Do not send the command until the wake trigger is confirmed.
- After the wake trigger is confirmed, wait about 1 second, then speak the test command.
- Watch both server logs and ESPHome serial logs for the full loop: wake, websocket connection, `Voice user transcript`, model/tool response, `Voice assistant response`, audio playback, stop, and return to idle.
- Repeat from idle for follow-up tests; avoid overlapping new TTS with assistant speech.

## Parallel Satellite Architecture

- Treat each ESP satellite wake session as isolated state: one websocket, one Pipecat pipeline, one OpenAI Realtime service, one context aggregator pair, and one optional recorder pair.
- Do not reintroduce a global `OpenAIRealtimeLLMService`, global websocket transport, or global recorder for active client sessions; those make rooms cancel or overwrite each other.
- Route wake words by session metadata (`session_start` with `wake_word`) or by websocket URL path/query (`/jarvis`, `?agent=jarvis`).
- Keep `disconnect_client` scoped to the current websocket only, so "stop listening" in one room does not close another room's session.
- Keep wake sessions single-turn by default: arm server-side auto-disconnect after the final assistant response and close the current websocket only after `BotStoppedSpeakingFrame`, so audio is not clipped and the ESP returns to idle for the next wake word.
