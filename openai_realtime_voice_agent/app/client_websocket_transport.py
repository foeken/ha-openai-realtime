"""Per-client WebSocket transport for parallel voice satellite sessions."""
import asyncio
import io
import json
import logging
import time
import wave
from typing import Awaitable, Callable, Optional

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.server import WebsocketServerParams

logger = logging.getLogger(__name__)

TextMessageHandler = Callable[[str], Awaitable[None]]


class ClientWebsocketInputTransport(BaseInputTransport):
    """Input side for one already-accepted websocket connection."""

    def __init__(
        self,
        transport: BaseTransport,
        websocket,
        params: WebsocketServerParams,
        *,
        initial_messages: Optional[list[bytes | str]] = None,
        text_message_handler: Optional[TextMessageHandler] = None,
        **kwargs,
    ):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._websocket = websocket
        self._params = params
        self._initial_messages = list(initial_messages or [])
        self._text_message_handler = text_message_handler
        self._receive_task = None
        self._closed_event = asyncio.Event()
        self._initialized = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._initialized:
            return

        self._initialized = True
        if self._params.serializer:
            await self._params.serializer.setup(frame)
        if not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler())
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._cancel_receive_task()

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._cancel_receive_task()

    async def cleanup(self):
        await super().cleanup()
        await self._transport.cleanup()

    async def wait_closed(self):
        await self._closed_event.wait()

    async def _cancel_receive_task(self):
        if self._receive_task:
            await self.cancel_task(self._receive_task)
            self._receive_task = None

    async def _receive_task_handler(self):
        try:
            for message in self._initial_messages:
                await self._handle_message(message)

            async for message in self._websocket:
                await self._handle_message(message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "%s exception receiving data: %s (%s)",
                self,
                exc.__class__.__name__,
                exc,
            )
        finally:
            self._closed_event.set()

    async def _handle_message(self, message: bytes | str):
        if isinstance(message, str):
            if self._text_message_handler:
                await self._text_message_handler(message)
            return

        if not self._params.serializer:
            return

        frame = await self._params.serializer.deserialize(message)
        if not frame:
            return

        if isinstance(frame, InputAudioRawFrame):
            await self.push_audio_frame(frame)
        else:
            await self.push_frame(frame)


class ClientWebsocketOutputTransport(BaseOutputTransport):
    """Output side for one already-accepted websocket connection."""

    def __init__(self, transport: BaseTransport, websocket, params: WebsocketServerParams, **kwargs):
        super().__init__(params, **kwargs)
        self._transport = transport
        self._websocket = websocket
        self._params = params
        self._send_interval = 0
        self._next_send_time = 0
        self._initialized = False

    async def start(self, frame: StartFrame):
        await super().start(frame)
        if self._initialized:
            return

        self._initialized = True
        if self._params.serializer:
            await self._params.serializer.setup(frame)
        self._send_interval = (self.audio_chunk_size / self.sample_rate) / 2
        await self.set_transport_ready(frame)

    async def stop(self, frame: EndFrame):
        await super().stop(frame)
        await self._write_frame(frame)

    async def cancel(self, frame: CancelFrame):
        await super().cancel(frame)
        await self._write_frame(frame)

    async def cleanup(self):
        await super().cleanup()
        await self._transport.cleanup()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, InterruptionFrame):
            await self._write_frame(frame)
            self._next_send_time = 0

    async def send_message(
        self,
        frame: OutputTransportMessageFrame | OutputTransportMessageUrgentFrame,
    ):
        await self._write_frame(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        if not self._websocket:
            return False

        frame = OutputAudioRawFrame(
            audio=frame.audio,
            sample_rate=self.sample_rate,
            num_channels=self._params.audio_out_channels,
        )

        if self._params.add_wav_header:
            with io.BytesIO() as buffer:
                with wave.open(buffer, "wb") as wf:
                    wf.setsampwidth(2)
                    wf.setnchannels(frame.num_channels)
                    wf.setframerate(frame.sample_rate)
                    wf.writeframes(frame.audio)
                frame = OutputAudioRawFrame(
                    buffer.getvalue(),
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels,
                )

        await self._write_frame(frame)
        await self._write_audio_sleep()
        return True

    async def send_json(self, payload: dict):
        if not self._websocket:
            return
        await self._websocket.send(json.dumps(payload))

    async def _write_frame(self, frame: Frame):
        if not self._params.serializer:
            return
        try:
            payload = await self._params.serializer.serialize(frame)
            if payload and self._websocket:
                await self._websocket.send(payload)
        except Exception as exc:
            logger.error(
                "%s exception sending data: %s (%s)",
                self,
                exc.__class__.__name__,
                exc,
            )

    async def _write_audio_sleep(self):
        current_time = time.monotonic()
        sleep_duration = max(0, self._next_send_time - current_time)
        await asyncio.sleep(sleep_duration)
        if sleep_duration == 0:
            self._next_send_time = time.monotonic() + self._send_interval
        else:
            self._next_send_time += self._send_interval


class ClientWebsocketTransport(BaseTransport):
    """Pipecat transport bound to a single websocket connection."""

    def __init__(
        self,
        websocket,
        params: WebsocketServerParams,
        *,
        initial_messages: Optional[list[bytes | str]] = None,
        text_message_handler: Optional[TextMessageHandler] = None,
        input_name: Optional[str] = None,
        output_name: Optional[str] = None,
    ):
        super().__init__(input_name=input_name, output_name=output_name)
        self._websocket = websocket
        self._params = params
        self._initial_messages = list(initial_messages or [])
        self._text_message_handler = text_message_handler
        self._input: Optional[ClientWebsocketInputTransport] = None
        self._output: Optional[ClientWebsocketOutputTransport] = None

    def input(self) -> ClientWebsocketInputTransport:
        if not self._input:
            self._input = ClientWebsocketInputTransport(
                self,
                self._websocket,
                self._params,
                initial_messages=self._initial_messages,
                text_message_handler=self._text_message_handler,
                name=self._input_name,
            )
        return self._input

    def output(self) -> ClientWebsocketOutputTransport:
        if not self._output:
            self._output = ClientWebsocketOutputTransport(
                self,
                self._websocket,
                self._params,
                name=self._output_name,
            )
        return self._output

    async def wait_closed(self):
        await self.input().wait_closed()

    async def disconnect_client(self, reason: str = "user_requested_stop"):
        try:
            await self.output().send_json(
                {
                    "type": "disconnect",
                    "message": "User requested disconnect",
                    "reason": reason,
                }
            )
            await asyncio.sleep(0.1)
        except Exception as exc:
            logger.debug("Could not send disconnect message: %s", exc)

        try:
            await self._websocket.close()
        except Exception as exc:
            logger.debug("Could not close client websocket: %s", exc)
