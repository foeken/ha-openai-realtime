"""Main application entry point using Pipecat."""
import os
import sys
import asyncio
import json
import logging
from contextlib import suppress
from typing import Literal, Optional
import dotenv
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.openai.realtime import events as realtime_events
from pipecat.services.openai.realtime.llm import OpenAIRealtimeLLMService
from pipecat.transports.websocket.server import WebsocketServerTransport
from websockets.asyncio.client import connect as websocket_connect
from app.mcp_service import HomeAssistantMCPService
from app.disconnect_tool import get_disconnect_tool_definition, create_disconnect_tool_handler
from app.audio_recording_service import AudioRecordingService
from app.session_manager import SessionManager
from app.websocket_handler import WebSocketHandler

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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._session_update_patch = session_update_patch or {}
        self._ping_interval_seconds = ping_interval_seconds
        self._ping_timeout_seconds = ping_timeout_seconds
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

    async def _handle_evt_audio_transcript_delta(self, evt):
        if evt.delta:
            self._assistant_response_text_parts.append(evt.delta)

        await super()._handle_evt_audio_transcript_delta(evt)

    async def _handle_evt_audio_transcript_done(self, evt):
        transcript = evt.transcript.strip()
        if transcript:
            self._assistant_response_text_parts = [transcript]

    async def _handle_evt_response_done(self, evt):
        await super()._handle_evt_response_done(evt)
        if not self._assistant_response_text_parts:
            text = self._extract_assistant_response_text(evt.response)
            if text:
                self._assistant_response_text_parts = [text]
        self._log_assistant_response_text()

        if self._create_response_after_active_done:
            self._create_response_after_active_done = False
            await self._create_response()

    def _log_assistant_response_text(self):
        text = "".join(self._assistant_response_text_parts).strip()
        self._assistant_response_text_parts = []
        if text:
            logger.info("Voice assistant response: %s", " ".join(text.split()))

    def _extract_assistant_response_text(self, response):
        parts = []
        for item in response.output or []:
            if item.role != "assistant" or not item.content:
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
        self.pipeline: Optional[Pipeline] = None
        self.runner: Optional[PipelineRunner] = None
        self.websocket_handler: Optional[WebSocketHandler] = None
        self.websocket_transport: Optional[WebsocketServerTransport] = None
        self.openai_service: Optional[OpenAIRealtimeLLMService] = None
        self.mcp_service: Optional[HomeAssistantMCPService] = None
        self.audio_recording_service: Optional[AudioRecordingService] = None
        self.session_manager: Optional[SessionManager] = None
        self.current_task: Optional[PipelineTask] = None
        self._pipeline_lock: Optional[asyncio.Lock] = None
        
    async def initialize(self) -> None:
        """Initialize all components."""
        # Get configuration from environment
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        websocket_port = int(os.environ.get("WEBSOCKET_PORT", "8080"))
        websocket_host = os.environ.get("WEBSOCKET_HOST", "0.0.0.0")
        
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
        
        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(
            host=websocket_host,
            port=websocket_port,
            session_manager=self.session_manager,
            audio_recording_service=self.audio_recording_service
        )
        self.websocket_transport = self.websocket_handler.create_transport()
        
        # Store configuration for session creation
        self.openai_api_key = openai_api_key
        self.vad_threshold = vad_threshold
        self.vad_prefix_padding_ms = vad_prefix_padding_ms
        self.vad_silence_duration_ms = vad_silence_duration_ms
        self.vad_idle_timeout_ms = vad_idle_timeout_ms
        self.openai_realtime_model = openai_realtime_model
        self.openai_realtime_voice = openai_realtime_voice
        self.openai_transcription_model = openai_transcription_model
        self.openai_noise_reduction = openai_noise_reduction
        self.instructions = instructions
        self.mcp_client = mcp_client
        
        # Initialize audio recording service (optional)
        self.audio_recording_service = AudioRecordingService(
            enable_recording=enable_recording,
            sample_rate=24000,
            chunk_duration_seconds=30,
            output_dir="recordings"
        )
        
        logger.info("✅ Application initialized - ready to accept WebSocket connections")
    
    def _build_pipeline_for_transport(self, transport: WebsocketServerTransport, client_id: str):
        """
        Build pipeline for a WebSocket transport connection.
        
        Args:
            transport: The WebSocket transport instance
            client_id: Unique identifier for the client device
        """
        # Ensure OpenAI service exists
        if self.openai_service is None:
            raise RuntimeError("OpenAI service must be created before building pipeline")
        
        # Use WebSocket handler to build pipeline
        self.pipeline, self.runner, self.current_task = self.websocket_handler.build_pipeline(
            transport=transport,
            openai_service=self.openai_service,
            client_id=client_id,
            activity_callback=self._update_session_activity
        )
    
    def _update_session_activity(self):
        """Update session activity timestamp (called by SessionActivityTracker)."""
        pass
    
    async def _ensure_openai_service(self, client_id: Optional[str] = None):
        """Create a new OpenAI service instance for a client.
        
        Args:
            client_id: Optional client ID for session management
        """
        if self._pipeline_lock is None:
            self._pipeline_lock = asyncio.Lock()
        
        async with self._pipeline_lock:
            if client_id is None:
                logger.warning("⚠️ No client_id provided to _ensure_openai_service")
            
            # Create new session
            if client_id:
                logger.info(f"🆕 Creating new OpenAI Session for Client {client_id}...")
            else:
                logger.info("🆕 Creating new OpenAI Session...")
            
            # Cache context from old service before creating new one
            if client_id and self.openai_service is not None:
                try:
                    self.session_manager.cleanup_before_new_session(client_id)
                    logger.debug(f"Cached context from previous session for client {client_id}")
                except Exception as e:
                    logger.warning(f"⚠️ Error caching context from old service for client {client_id}: {e}")
            
            # Create session properties with audio configuration
            from pipecat.services.openai.realtime.events import (
                SessionProperties,
                AudioConfiguration,
                AudioInput,
                AudioOutput,
                InputAudioNoiseReduction,
                InputAudioTranscription,
                PCMAudioFormat,
                TurnDetection
            )
            
            # Create disconnect tool definition
            disconnect_tool_def = get_disconnect_tool_definition()
            
            # Collect all tool definitions for session properties
            all_tools = [disconnect_tool_def]
            
            # Get MCP tool definitions if available
            mcp_tools_schema = None
            if self.mcp_client:
                try:
                    logger.info("🔧 Fetching MCP tool definitions...")
                    mcp_tools_schema = await self.mcp_client.get_tools_schema()
                    
                    # Convert MCP tool schemas to OpenAI format
                    for function_schema in mcp_tools_schema.standard_tools:
                        openai_tool = {
                            "type": "function",
                            "name": function_schema.name,
                            "description": function_schema.description,
                            "parameters": {
                                "type": "object",
                                "properties": function_schema.properties,
                                "required": function_schema.required
                            }
                        }
                        all_tools.append(openai_tool)
                    
                    logger.info(f"✅ Fetched {len(mcp_tools_schema.standard_tools)} MCP tools")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to fetch MCP tool definitions: {e}")
            
            session_properties = SessionProperties(
                instructions=self.instructions,
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
                            silence_duration_ms=self.vad_silence_duration_ms
                        )
                    ),
                    output=AudioOutput(
                        format=PCMAudioFormat(),
                        voice=self.openai_realtime_voice
                    )
                ),
                tools=all_tools,
                max_output_tokens="inf"
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
                "🔧 Realtime config: model=%s voice=%s transcription=%s noise_reduction=%s vad_idle_timeout_ms=%s",
                self.openai_realtime_model,
                self.openai_realtime_voice,
                self.openai_transcription_model,
                self.openai_noise_reduction,
                self.vad_idle_timeout_ms,
            )

            logger.info(f"🔧 Creating session with {len(all_tools)} tools: {[tool.get('name', 'unknown') for tool in all_tools]}")

            # Create new service instance
            self.openai_service = PatchedOpenAIRealtimeLLMService(
                api_key=self.openai_api_key,
                model=self.openai_realtime_model,
                session_properties=session_properties,
                session_update_patch=session_update_patch,
                start_audio_paused=True
            )
            logger.info(f"✅ OpenAI Service created: {type(self.openai_service).__name__}")
            
            # Register disconnect tool handler
            disconnect_tool_handler = create_disconnect_tool_handler(self.websocket_transport)
            self.openai_service.register_function("disconnect_client", disconnect_tool_handler)
            logger.info("✅ Registered disconnect tool handler")
            
            # Register MCP tool handlers if available
            if self.mcp_client and mcp_tools_schema:
                try:
                    await self.mcp_client.register_tools_schema(mcp_tools_schema, self.openai_service)
                    logger.info(f"✅ Registered {len(mcp_tools_schema.standard_tools)} MCP tool handlers")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to register MCP tool handlers: {e}")
            
            # Register service with session manager
            if client_id:
                self.session_manager.set_current_service(client_id, self.openai_service)
            
            logger.info("✅ New OpenAI Session created")
            return self.openai_service
    
    async def run(self) -> None:
        """Run the application."""
        await self.initialize()
        
        # Create initial OpenAI service (will be replaced per connection)
        await self._ensure_openai_service()
        
        # Build pipeline - based on pipecat-examples, one pipeline handles all connections
        # The transport manages multiple connections internally
        self._build_pipeline_for_transport(self.websocket_transport, "server")
        
        # Setup WebSocket event handlers
        async def on_client_connected(client_id: str):
            """Handle new client connection."""
            if self.openai_service:
                self.openai_service.set_audio_input_paused(False)
            if self.session_manager and self.openai_service:
                self.session_manager.set_current_service(client_id, self.openai_service)
            if self.audio_recording_service:
                self.audio_recording_service.start_new_session(client_id)
        
        def on_client_disconnected(client_id: str):
            """Handle client disconnection."""
            if self.openai_service:
                self.openai_service.set_audio_input_paused(True)
            if self.session_manager:
                self.session_manager.handle_client_disconnect(client_id, self.openai_service)
                self.session_manager.clear_context("server")
            if self.audio_recording_service:
                self.audio_recording_service.stop_recording()
            if self.openai_service and hasattr(self.openai_service, "park_realtime_connection"):
                asyncio.create_task(self.openai_service.park_realtime_connection())
        
        # Function to get OpenAI service for a client
        def get_openai_service_for_client(client_id: str) -> Optional[OpenAIRealtimeLLMService]:
            """Get OpenAI service for a specific client."""
            if self.session_manager:
                return self.session_manager.get_current_service(client_id)
            return self.openai_service
        
        self.websocket_handler.setup_event_handlers(
            transport=self.websocket_transport,
            on_client_connected_callback=on_client_connected,
            on_client_disconnected_callback=on_client_disconnected,
            openai_service_getter=get_openai_service_for_client
        )
        
        try:
            # Start the pipeline runner - this will start the WebSocket server
            # Based on pipecat-examples: PipelineRunner.run() starts the transport server
            logger.info("✅ Starting WebSocket server and pipeline...")
            await self.runner.run(self.current_task)
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
        
        if self.runner:
            try:
                await self.runner.cancel()
            except Exception as e:
                logger.warning(f"⚠️ Error cancelling runner: {e}")
        
        if self.websocket_handler:
            try:
                await self.websocket_handler.cleanup()
            except Exception as e:
                logger.warning(f"⚠️ Error cleaning up WebSocket handler: {e}")
        
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
