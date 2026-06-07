"""Main application entry point using Pipecat."""
import os
import sys
import asyncio
import json
import logging
from contextlib import suppress
from typing import Callable, Literal, Optional
from urllib.parse import parse_qs, urlparse
import dotenv
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection
from pipecat.frames.frames import UserStartedSpeakingFrame, UserStoppedSpeakingFrame
from pipecat.services.openai.realtime import events as realtime_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.server import WebsocketServerParams
from websockets.asyncio.client import connect as websocket_connect
from websockets.asyncio.server import serve as websocket_serve
from app.agent_config import AgentProfile, AgentRegistry, load_agent_registry, normalize_route_key
from app.client_websocket_transport import ClientWebsocketTransport
from app.mcp_service import HomeAssistantMCPService
from app.disconnect_tool import get_disconnect_tool_definition, create_disconnect_tool_handler
from app.audio_recording_service import AudioRecordingService
from app.raw_audio_serializer import RawAudioSerializer
from app.session_manager import SessionManager
from app.websocket_handler import SessionAutoDisconnect, WebSocketHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce verbosity of noisy loggers
logging.getLogger("aiortc").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("__main__").setLevel(logging.INFO)

dotenv.load_dotenv()


DEFAULT_INSTRUCTIONS = (
    "You are the Home Assistant Voice Agent and can control the smart home. "
    "Respond in English unless the user explicitly asks for another language. "
    "Hey Mycroft is currently disabled on the ESP client because the V2 wake-word "
    "model tensor arena no longer fits after tensor-size changes. "
    "When a tool is needed, call the tool without filler like 'let me check', "
    "then speak a concise answer after the tool result returns."
)


class InputAudioBufferTimeoutTriggered(realtime_events.ServerEvent):
    type: Literal["input_audio_buffer.timeout_triggered"]
    audio_start_ms: int
    audio_end_ms: int
    item_id: Optional[str] = None


realtime_events._server_event_types.setdefault(
    "input_audio_buffer.timeout_triggered",
    InputAudioBufferTimeoutTriggered,
)


class PatchedOpenAIRealtimeLLMService(OpenAIRealtimeLLMService):
    """Adds session.update fields not yet modeled by every Pipecat release."""

    def __init__(
        self,
        *args,
        session_update_patch: Optional[dict] = None,
        ping_interval_seconds: Optional[float] = 30.0,
        ping_timeout_seconds: Optional[float] = 60.0,
        on_final_response_done: Optional[Callable[[], None]] = None,
        on_user_started_speaking: Optional[Callable[[], object]] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._session_update_patch = session_update_patch or {}
        self._ping_interval_seconds = ping_interval_seconds
        self._ping_timeout_seconds = ping_timeout_seconds
        self._on_final_response_done = on_final_response_done
        self._on_user_started_speaking = on_user_started_speaking
        self._reconnect_lock = asyncio.Lock()
        self._assistant_response_text_parts = []
        self._create_response_after_active_done = False

    async def _receive_task_handler(self):
        async for message in self._websocket:
            evt = realtime_events.parse_server_event(message)
            if evt.type == "session.created":
                await self._handle_evt_session_created(evt)
            elif evt.type == "session.updated":
                await self._handle_evt_session_updated(evt)
            elif evt.type == "response.output_audio.delta":
                await self._handle_evt_audio_delta(evt)
            elif evt.type == "response.output_audio.done":
                await self._handle_evt_audio_done(evt)
            elif evt.type == "conversation.item.added":
                await self._handle_evt_conversation_item_added(evt)
            elif evt.type == "conversation.item.done":
                await self._handle_evt_conversation_item_done(evt)
            elif evt.type == "conversation.item.input_audio_transcription.delta":
                await self._handle_evt_input_audio_transcription_delta(evt)
            elif evt.type == "conversation.item.input_audio_transcription.completed":
                await self.handle_evt_input_audio_transcription_completed(evt)
            elif evt.type == "conversation.item.retrieved":
                await self._handle_conversation_item_retrieved(evt)
            elif evt.type == "response.done":
                await self._handle_evt_response_done(evt)
            elif evt.type == "input_audio_buffer.speech_started":
                await self._handle_evt_speech_started(evt)
            elif evt.type == "input_audio_buffer.speech_stopped":
                await self._handle_evt_speech_stopped(evt)
            elif evt.type == "response.output_text.delta":
                await self._handle_evt_text_delta(evt)
            elif evt.type == "response.output_audio_transcript.delta":
                await self._handle_evt_audio_transcript_delta(evt)
            elif evt.type == "response.output_audio_transcript.done":
                await self._handle_evt_audio_transcript_done(evt)
            elif evt.type == "response.function_call_arguments.done":
                await self._handle_evt_function_call_arguments_done(evt)
            elif evt.type == "error":
                if await self._maybe_handle_evt_retrieve_conversation_item_error(evt):
                    continue
                fatal = await self._handle_evt_error(evt)
                if fatal:
                    return

    async def start(self, frame):
        """Start the processor without opening an idle Realtime websocket."""
        await super(OpenAIRealtimeLLMService, self).start(frame)

    async def _handle_context(self, context):
        if not self._context:
            self._context = context
            await self._process_completed_function_calls(send_new_results=False)
            if self._is_live_audio_session_active():
                self._llm_needs_conversation_setup = False
                logger.debug("%s using live Realtime audio context without replay", self)
                return

            await self._create_response()
        else:
            self._context = context
            await self._process_completed_function_calls(send_new_results=True)

    def _is_live_audio_session_active(self):
        return bool(self._websocket and not self._audio_input_paused)

    async def send_client_event(self, event):
        payload = event.model_dump(exclude_none=True)
        if payload.get("type") == "session.update":
            self._deep_merge(payload.setdefault("session", {}), self._session_update_patch)
        await self._ws_send(payload)

    async def _handle_evt_error(self, evt):
        if evt.error.code == "response_cancel_not_active":
            logger.debug("%s %s", self, evt.error.message)
            return False

        if evt.error.code == "conversation_already_has_active_response":
            logger.debug("%s %s", self, evt.error.message)
            self._create_response_after_active_done = True
            return False

        if (
            evt.error.code == "invalid_value"
            and evt.error.message
            and "Audio content of" in evt.error.message
            and "already shorter than" in evt.error.message
        ):
            logger.debug("%s ignoring stale audio truncate error: %s", self, evt.error.message)
            self._current_audio_response = None
            return False

        await super()._handle_evt_error(evt)
        return True

    async def handle_evt_input_audio_transcription_completed(self, evt):
        transcript = evt.transcript.strip()
        if transcript:
            logger.info("Voice user transcript: %s", transcript)

        await super().handle_evt_input_audio_transcription_completed(evt)

    async def _handle_evt_input_audio_transcription_delta(self, evt):
        await super()._handle_evt_input_audio_transcription_delta(evt)

    async def _handle_evt_speech_started(self, evt):
        await super()._handle_evt_speech_started(evt)
        await self._notify_user_started_speaking()
        await self.push_frame(UserStartedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _handle_evt_speech_stopped(self, evt):
        await super()._handle_evt_speech_stopped(evt)
        await self.push_frame(UserStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    async def _handle_evt_audio_transcript_delta(self, evt):
        if evt.delta:
            self._assistant_response_text_parts.append(evt.delta)

        await super()._handle_evt_audio_transcript_delta(evt)

    async def _handle_evt_audio_transcript_done(self, evt):
        transcript = evt.transcript.strip()
        if transcript:
            self._assistant_response_text_parts = [transcript]

    async def _handle_evt_response_done(self, evt):
        has_function_call = self._response_has_function_call(evt.response)
        await super()._handle_evt_response_done(evt)
        if not self._assistant_response_text_parts:
            text = self._extract_assistant_response_text(evt.response)
            if text:
                self._assistant_response_text_parts = [text]
        self._log_assistant_response_text()

        if (
            not has_function_call
            and not self._create_response_after_active_done
            and getattr(evt.response, "status", None) != "failed"
        ):
            self._notify_final_response_done()

        if self._create_response_after_active_done:
            self._create_response_after_active_done = False
            await self._create_response()

    def _notify_final_response_done(self):
        if not self._on_final_response_done:
            return

        try:
            self._on_final_response_done()
        except Exception as exc:
            logger.warning("Failed to notify final response completion: %s", exc)

    async def _notify_user_started_speaking(self):
        if not self._on_user_started_speaking:
            return

        try:
            result = self._on_user_started_speaking()
            if hasattr(result, "__await__"):
                await result
        except Exception as exc:
            logger.warning("Failed to notify user speech start: %s", exc)

    def _response_has_function_call(self, response) -> bool:
        for item in response.output or []:
            item_type = getattr(item, "type", None)
            if item_type == "function_call":
                return True
            if getattr(item, "call_id", None) and getattr(item, "name", None):
                return True
        return False

    def _log_assistant_response_text(self):
        text = "".join(self._assistant_response_text_parts).strip()
        self._assistant_response_text_parts = []
        if text:
            logger.info("Voice assistant response: %s", " ".join(text.split()))

    def _extract_assistant_response_text(self, response):
        parts = []
        for item in response.output or []:
            if getattr(item, "role", None) != "assistant" or not getattr(item, "content", None):
                continue
            for content in item.content:
                for field in ("transcript", "text"):
                    value = getattr(content, field, None)
                    if value:
                        parts.append(value)
        return " ".join(parts).strip()

    async def _connect(self):
        try:
            if self._websocket:
                return

            self._disconnecting = False
            self._websocket = await websocket_connect(
                uri=self.base_url,
                additional_headers={
                    "Authorization": f"Bearer {self.api_key}",
                },
                ping_interval=self._ping_interval_seconds,
                ping_timeout=self._ping_timeout_seconds,
            )
            self._receive_task = self.create_task(self._receive_task_handler())
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting: {e}", exception=e)
            self._websocket = None

    async def _ws_send(self, realtime_message):
        if self._disconnecting:
            return

        if not self._websocket:
            if not await self._ensure_realtime_websocket():
                await self.push_error(
                    error_msg="Error sending client event: OpenAI Realtime websocket is not connected"
                )
                return
            await self._wait_for_session_ready(realtime_message)

        websocket = self._websocket
        try:
            await self._send_realtime_message(realtime_message)
            return
        except Exception as e:
            if self._disconnecting:
                return

            if not await self._recover_realtime_websocket(e, websocket):
                await self.push_error(error_msg=f"Error sending client event: {e}", exception=e)
                return

        await self._wait_for_session_ready(realtime_message)

        try:
            await self._send_realtime_message(realtime_message)
        except Exception as e:
            await self.push_error(
                error_msg=f"Error sending client event after reconnect: {e}",
                exception=e,
            )

    async def _ensure_realtime_websocket(self) -> bool:
        async with self._reconnect_lock:
            if self._disconnecting:
                return False

            if self._websocket:
                return True

            self._api_session_ready = False
            self._llm_needs_conversation_setup = True
            self._context = None
            self._completed_tool_calls = set()
            self._pending_function_calls.clear()
            self._current_audio_response = None
            self._assistant_response_text_parts = []
            self._create_response_after_active_done = False

            await self._connect()
            return self._websocket is not None

    async def _send_realtime_message(self, realtime_message):
        if not self._websocket:
            raise ConnectionError("OpenAI Realtime websocket is not connected")

        await self._websocket.send(json.dumps(realtime_message))

    async def _recover_realtime_websocket(self, error: Exception, failed_websocket) -> bool:
        async with self._reconnect_lock:
            if self._disconnecting:
                return False

            if self._websocket is not failed_websocket and self._websocket:
                return True

            logger.warning("OpenAI Realtime websocket send failed; reconnecting session: %s", error)

            receive_task = self._receive_task
            self._receive_task = None
            if receive_task:
                await self.cancel_task(receive_task, timeout=1.0)

            websocket = self._websocket
            self._websocket = None
            if websocket:
                with suppress(Exception):
                    await websocket.close()

            self._api_session_ready = False
            self._llm_needs_conversation_setup = True
            self._current_audio_response = None
            self._assistant_response_text_parts = []
            self._create_response_after_active_done = False

            await self._connect()
            return self._websocket is not None

    async def park_realtime_connection(self):
        """Close the upstream Realtime websocket between ESP wake sessions."""
        async with self._reconnect_lock:
            if not self._websocket and not self._receive_task:
                return

            logger.info("Parking OpenAI Realtime websocket until the next wake session")
            await self._disconnect()
            self._api_session_ready = False
            self._run_llm_when_api_session_ready = False
            self._llm_needs_conversation_setup = True
            self._context = None
            self._completed_tool_calls = set()
            self._pending_function_calls.clear()
            self._current_audio_response = None
            self._current_assistant_response = None
            self._assistant_response_text_parts = []
            self._create_response_after_active_done = False

    async def _wait_for_session_ready(self, realtime_message):
        if realtime_message.get("type") == "session.update":
            return

        deadline = asyncio.get_running_loop().time() + 5.0
        while not self._api_session_ready and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.05)

        if not self._api_session_ready:
            logger.warning("OpenAI Realtime session did not confirm readiness after reconnect")

    @classmethod
    def _deep_merge(cls, target: dict, patch: dict):
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                cls._deep_merge(target[key], value)
            else:
                target[key] = value


class Application:
    """Main application class using Pipecat."""
    
    def __init__(self):
        """Initialize application."""
        self.websocket_handler: Optional[WebSocketHandler] = None
        self.mcp_service: Optional[HomeAssistantMCPService] = None
        self.audio_recording_service: Optional[AudioRecordingService] = None
        self.session_manager: Optional[SessionManager] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.active_sessions: dict[str, dict] = {}
        self._server = None
        
    async def initialize(self) -> None:
        """Initialize all components."""
        # Get configuration from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        websocket_port = int(os.environ.get("WEBSOCKET_PORT", "8080"))
        websocket_host = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
        client_metadata_timeout = float(os.environ.get("CLIENT_METADATA_TIMEOUT_SECONDS", "1.0"))
        auto_disconnect_after_response_seconds = float(
            os.environ.get("AUTO_DISCONNECT_AFTER_RESPONSE_SECONDS", "5.0")
        )
        
        # Get turn detection settings with defaults
        vad_threshold = float(os.environ.get("VAD_THRESHOLD", "0.5"))
        vad_prefix_padding_ms = int(os.environ.get("VAD_PREFIX_PADDING_MS", "300"))
        vad_silence_duration_ms = int(os.environ.get("VAD_SILENCE_DURATION_MS", "500"))
        vad_idle_timeout_ms = int(os.environ.get("VAD_IDLE_TIMEOUT_MS", "0"))

        # Get OpenAI Realtime settings with defaults
        openai_realtime_model = os.environ.get("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
        openai_realtime_voice = os.environ.get("OPENAI_REALTIME_VOICE", "cedar").lower()
        openai_transcription_model = os.environ.get("OPENAI_TRANSCRIPTION_MODEL", "gpt-realtime-whisper")
        openai_noise_reduction = os.environ.get("OPENAI_NOISE_REDUCTION", "far_field")
        
        # Get instructions with default
        instructions = os.environ.get("INSTRUCTIONS", DEFAULT_INSTRUCTIONS)
        
        # Get recording setting (optional, defaults to false)
        enable_recording = os.environ.get("ENABLE_RECORDING", "false").lower() == "true"
        
        # Get session reuse timeout and initialize session manager
        session_reuse_timeout = float(os.environ.get("SESSION_REUSE_TIMEOUT_SECONDS", "300"))
        self.session_manager = SessionManager(reuse_timeout=session_reuse_timeout)
        logger.info(f"Session reuse timeout: {session_reuse_timeout} seconds")
        
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize Home Assistant MCP Service
        mcp_client = None
        try:
            supervisor_token = os.environ.get("LONGLIVED_TOKEN") or os.environ.get("SUPERVISOR_TOKEN")
            ha_mcp_url = os.environ.get("HA_MCP_URL", "http://supervisor/core/api/mcp")
            if supervisor_token:
                logger.info("Loading Home Assistant MCP tools...")
                self.mcp_service = HomeAssistantMCPService(url=ha_mcp_url, access_token=supervisor_token)
                mcp_client = await self.mcp_service.initialize()
                logger.info("✅ Home Assistant MCP Client initialized")
            else:
                logger.warning("⚠️ SUPERVISOR_TOKEN not set, skipping Home Assistant MCP integration")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Home Assistant MCP Client: {e}")
        
        # Initialize audio recording service (optional)
        self.audio_recording_service = AudioRecordingService(
            enable_recording=enable_recording,
            sample_rate=24000,
            chunk_duration_seconds=30,
            output_dir="recordings"
        )

        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(
            host=websocket_host,
            port=websocket_port,
            session_manager=self.session_manager,
            audio_recording_service=self.audio_recording_service
        )
        
        # Store configuration for session creation
        self.openai_api_key = openai_api_key
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        self.client_metadata_timeout = client_metadata_timeout
        self.auto_disconnect_after_response_seconds = auto_disconnect_after_response_seconds
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_idle_timeout_ms = vad_idle_timeout_ms
        self.openai_realtime_model = openai_realtime_model
        self.openai_transcription_model = openai_transcription_model
        self.openai_noise_reduction = openai_noise_reduction
        self.instructions = instructions
        self.mcp_client = mcp_client
        self.agent_registry = load_agent_registry(
            os.environ,
            default_instructions=instructions,
            default_voice=openai_realtime_voice,
        )
        
        logger.info("✅ Application initialized - ready to accept WebSocket connections")
    
    def _update_session_activity(self):
        """Update session activity timestamp (called by SessionActivityTracker)."""
        pass

    def _transport_params(self) -> WebsocketServerParams:
        return WebsocketServerParams(
            serializer=RawAudioSerializer(),
            audio_in_enabled=True,
            audio_out_enabled=True,
        )

    def _request_path(self, websocket) -> str:
        request = getattr(websocket, "request", None)
        return getattr(request, "path", None) or getattr(websocket, "path", "") or "/"

    def _metadata_from_path(self, websocket) -> dict[str, str]:
        parsed = urlparse(self._request_path(websocket))
        query = parse_qs(parsed.query)
        metadata: dict[str, str] = {}

        for key in ("agent", "wake_word", "client_id"):
            values = query.get(key)
            if values and values[0]:
                metadata[key] = values[0]

        path_parts = [part for part in parsed.path.split("/") if part]
        if path_parts:
            if path_parts[0] in ("agent", "agents") and len(path_parts) > 1:
                metadata.setdefault("agent", path_parts[1])
            else:
                metadata.setdefault("agent", path_parts[0])

        return metadata

    def _parse_control_message(self, message: str) -> Optional[dict]:
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None

    async def _read_initial_metadata(self, websocket) -> tuple[dict[str, str], list[bytes | str]]:
        metadata = self._metadata_from_path(websocket)
        pending_messages: list[bytes | str] = []

        try:
            message = await asyncio.wait_for(websocket.recv(), timeout=self.client_metadata_timeout)
        except asyncio.TimeoutError:
            return metadata, pending_messages
        except Exception:
            return metadata, pending_messages

        if isinstance(message, str):
            parsed = self._parse_control_message(message)
            if parsed and parsed.get("type") == "session_start":
                for key in ("agent", "wake_word", "client_id"):
                    value = parsed.get(key)
                    if isinstance(value, str) and value:
                        metadata[key] = value
                return metadata, pending_messages

        pending_messages.append(message)
        return metadata, pending_messages

    def _extract_client_id(self, websocket, metadata: dict[str, str], agent: AgentProfile) -> str:
        remote_host = None
        remote = getattr(websocket, "remote_address", None)
        if isinstance(remote, tuple) and remote:
            remote_host = str(remote[0])

        client_id = metadata.get("client_id")
        if not client_id:
            client_id = remote_host or "unknown"
        elif remote_host:
            client_id = f"{client_id}_{remote_host}"

        normalized_client = normalize_route_key(client_id) or "unknown"
        return f"{normalized_client}:{agent.name}"

    def _filter_mcp_tools(
        self,
        mcp_tools_schema: Optional[ToolsSchema],
        agent: AgentProfile,
    ) -> Optional[ToolsSchema]:
        if not mcp_tools_schema:
            return None
        filtered_tools = [
            tool for tool in mcp_tools_schema.standard_tools
            if agent.allows_tool(tool.name)
        ]
        return ToolsSchema(
            standard_tools=filtered_tools,
            custom_tools=mcp_tools_schema.custom_tools,
        )

    async def _create_openai_service(
        self,
        *,
        client_id: str,
        agent: AgentProfile,
        transport: ClientWebsocketTransport,
        on_final_response_done: Optional[Callable[[], None]] = None,
        on_user_started_speaking: Optional[Callable[[], object]] = None,
    ) -> OpenAIRealtimeLLMService:
        """Create a new OpenAI Realtime service for one client connection."""
        logger.info("🆕 Creating OpenAI session for %s using agent '%s'", client_id, agent.name)

        if self.session_manager:
            self.session_manager.cleanup_before_new_session(client_id)

        from pipecat.services.openai.realtime.events import (
            AudioConfiguration,
            AudioInput,
            AudioOutput,
            InputAudioNoiseReduction,
            InputAudioTranscription,
            PCMAudioFormat,
            SessionProperties,
            TurnDetection,
        )

        disconnect_tool_def = get_disconnect_tool_definition()
        all_tools = [disconnect_tool_def]

        mcp_tools_schema = None
        filtered_mcp_tools_schema = None
        if self.mcp_client:
            try:
                logger.info("🔧 Fetching MCP tool definitions for agent '%s'...", agent.name)
                mcp_tools_schema = await self.mcp_client.get_tools_schema()
                filtered_mcp_tools_schema = self._filter_mcp_tools(mcp_tools_schema, agent)

                for function_schema in filtered_mcp_tools_schema.standard_tools:
                    all_tools.append(
                        {
                            "type": "function",
                            "name": function_schema.name,
                            "description": function_schema.description,
                            "parameters": {
                                "type": "object",
                                "properties": function_schema.properties,
                                "required": function_schema.required,
                            },
                        }
                    )

                logger.info(
                    "✅ Agent '%s' has %s/%s MCP tools",
                    agent.name,
                    len(filtered_mcp_tools_schema.standard_tools),
                    len(mcp_tools_schema.standard_tools),
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to fetch MCP tool definitions: {e}")

        session_properties = SessionProperties(
            instructions=agent.instructions,
            output_modalities=["audio"],
            audio=AudioConfiguration(
                input=AudioInput(
                    format=PCMAudioFormat(),
                    transcription=InputAudioTranscription(
                        model=self.openai_transcription_model
                    ),
                    noise_reduction=InputAudioNoiseReduction(
                        type=self.openai_noise_reduction
                    ),
                    turn_detection=TurnDetection(
                        type="server_vad",
                        threshold=self.vad_threshold,
                        prefix_padding_ms=self.vad_prefix_padding_ms,
                        silence_duration_ms=self.vad_silence_duration_ms,
                    ),
                ),
                output=AudioOutput(
                    format=PCMAudioFormat(),
                    voice=agent.voice,
                ),
            ),
            tools=all_tools,
            max_output_tokens="inf",
        )
        session_update_patch = {}
        if self.vad_idle_timeout_ms > 0:
            session_update_patch = {
                "audio": {
                    "input": {
                        "turn_detection": {
                            "idle_timeout_ms": self.vad_idle_timeout_ms,
                        }
                    }
                }
            }

        logger.info(
            "🔧 Realtime config client=%s agent=%s model=%s voice=%s transcription=%s noise_reduction=%s tools=%s",
            client_id,
            agent.name,
            self.openai_realtime_model,
            agent.voice,
            self.openai_transcription_model,
            self.openai_noise_reduction,
            [tool.get("name", "unknown") for tool in all_tools],
        )

        service = PatchedOpenAIRealtimeLLMService(
            api_key=self.openai_api_key,
            model=self.openai_realtime_model,
            session_properties=session_properties,
            session_update_patch=session_update_patch,
            start_audio_paused=False,
            on_final_response_done=on_final_response_done,
            on_user_started_speaking=on_user_started_speaking,
        )

        async def disconnect_client_callback():
            await transport.disconnect_client()

        service.register_function(
            "disconnect_client",
            create_disconnect_tool_handler(disconnect_callback=disconnect_client_callback),
        )
        logger.info("✅ Registered disconnect tool handler for %s", client_id)

        if self.mcp_client and filtered_mcp_tools_schema:
            try:
                await self.mcp_client.register_tools_schema(filtered_mcp_tools_schema, service)
                logger.info(
                    "✅ Registered %s MCP tool handlers for %s",
                    len(filtered_mcp_tools_schema.standard_tools),
                    client_id,
                )
            except Exception as e:
                logger.warning(f"⚠️ Failed to register MCP tool handlers: {e}")

        if self.session_manager:
            self.session_manager.set_current_service(client_id, service)

        return service

    async def _handle_control_message(
        self,
        *,
        client_id: str,
        service: OpenAIRealtimeLLMService,
        message: str,
    ):
        data = self._parse_control_message(message)
        if not data:
            logger.debug("📨 Received non-JSON text message from %s: %s", client_id, message[:100])
            return

        message_type = data.get("type")
        if message_type != "interrupt":
            logger.debug("📨 Received message from %s: %s", client_id, message_type)
            return

        logger.info("🛑 Interrupt received from client %s", client_id)
        try:
            if hasattr(service, "send_interrupt"):
                await service.send_interrupt()
            elif hasattr(service, "push_event"):
                await service.push_event({"type": "response.interrupt"})
            elif hasattr(service, "_send_event"):
                await service._send_event({"type": "response.interrupt"})
            else:
                logger.warning("⚠️ No interrupt method found for service %s", client_id)
        except Exception as e:
            logger.error("❌ Error sending interrupt to OpenAI service for %s: %s", client_id, e, exc_info=True)

    async def _handle_client(self, websocket):
        metadata, initial_messages = await self._read_initial_metadata(websocket)
        agent = self.agent_registry.resolve(
            agent_name=metadata.get("agent"),
            wake_word=metadata.get("wake_word"),
        )
        client_id = self._extract_client_id(websocket, metadata, agent)
        logger.info(
            "🔗 Client %s connected route=%s wake_word=%s agent=%s",
            client_id,
            self._request_path(websocket),
            metadata.get("wake_word"),
            agent.name,
        )

        service_holder: dict[str, OpenAIRealtimeLLMService] = {}

        async def text_handler(message: str):
            service = service_holder.get("service")
            if service:
                await self._handle_control_message(
                    client_id=client_id,
                    service=service,
                    message=message,
                )

        transport = ClientWebsocketTransport(
            websocket,
            self._transport_params(),
            initial_messages=initial_messages,
            text_message_handler=text_handler,
            input_name=f"{client_id}-input",
            output_name=f"{client_id}-output",
        )

        recording_session = None
        if self.audio_recording_service:
            recording_session = self.audio_recording_service.create_session(client_id)

        auto_disconnect = None
        if self.auto_disconnect_after_response_seconds >= 0:
            async def disconnect_after_response():
                await transport.disconnect_client(reason="assistant_response_complete")

            auto_disconnect = SessionAutoDisconnect(
                client_id=client_id,
                disconnect_callback=disconnect_after_response,
                delay_seconds=self.auto_disconnect_after_response_seconds,
            )

        service = await self._create_openai_service(
            client_id=client_id,
            agent=agent,
            transport=transport,
            on_final_response_done=auto_disconnect.arm if auto_disconnect else None,
            on_user_started_speaking=(
                lambda: auto_disconnect.cancel_pending("user speech")
                if auto_disconnect
                else None
            ),
        )
        service_holder["service"] = service

        pipeline, runner, task = self.websocket_handler.build_pipeline(
            transport=transport,
            openai_service=service,
            client_id=client_id,
            activity_callback=self._update_session_activity,
            recording_session=recording_session,
            auto_disconnect=auto_disconnect,
        )
        self.active_sessions[client_id] = {
            "agent": agent.name,
            "runner": runner,
            "task": task,
            "service": service,
            "transport": transport,
            "recording_session": recording_session,
            "auto_disconnect": auto_disconnect,
        }

        service.set_audio_input_paused(False)
        run_task = asyncio.create_task(runner.run(task))

        try:
            await transport.wait_closed()
        finally:
            service.set_audio_input_paused(True)
            if hasattr(service, "park_realtime_connection"):
                await service.park_realtime_connection()
            if recording_session:
                recording_session.stop()
            if self.session_manager:
                self.session_manager.handle_client_disconnect(client_id, service)
                self.session_manager.remove_context_aggregator(client_id)
            await runner.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await run_task
            self.active_sessions.pop(client_id, None)
            logger.info("🔌 Client %s disconnected", client_id)
    
    async def run(self) -> None:
        """Run the application."""
        await self.initialize()

        try:
            logger.info("✅ Starting multi-client WebSocket server on %s:%s", self.websocket_host, self.websocket_port)
            async with websocket_serve(
                self._handle_client,
                self.websocket_host,
                self.websocket_port,
                ping_interval=20,
                ping_timeout=10,
            ) as server:
                self._server = server
                await asyncio.Future()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
            raise
        finally:
            await self.cleanup()
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up application...")

        for client_id, session in list(self.active_sessions.items()):
            try:
                await session["runner"].cancel()
            except Exception as e:
                logger.warning("⚠️ Error cancelling runner for %s: %s", client_id, e)
            recording_session = session.get("recording_session")
            if recording_session:
                recording_session.stop()
            self.active_sessions.pop(client_id, None)
        
        if self.audio_recording_service:
            self.audio_recording_service.cleanup()
        
        logger.info("✅ Application cleanup complete")


async def main() -> None:
    """Main entry point."""
    app = Application()
    
    try:
        await app.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
