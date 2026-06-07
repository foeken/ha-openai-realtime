# AGENTS.md

## Voice Agent Development Loop

- Run the server locally before deploying to Home Assistant; HA updates are slow, so use the local websocket target for iteration.
- Keep the ESP client flashed against the local server URL while testing.
- For wake-word tests, say only `Hey Jarvis` first, then inspect server logs to confirm the trigger produced a client turn.
- Wait briefly for the websocket connection to appear before deciding the wake was missed; the ESP can connect a moment after the spoken wake word finishes.
- If the trigger still did not show up in the server logs and the ESP is still idle/disconnected, repeat only the wake word. Do not send the command until the wake trigger is confirmed.
- After the wake trigger is confirmed, wait about 1 second, then speak the test command.
- Watch both server logs and ESPHome serial logs for the full loop: wake, websocket connection, `Voice user transcript`, model/tool response, `Voice assistant response`, audio playback, stop, and return to idle.
- Repeat from idle for follow-up tests; avoid overlapping new TTS with assistant speech.
