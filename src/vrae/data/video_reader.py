"""The single video decoding boundary used by V-RAE.

The automatic backend policy is deliberately strict: torchcodec is preferred and
decord is considered only when importing torchcodec is impossible.  A torchcodec
open or decode error is never hidden by retrying the same video with another
backend.  Keeping that distinction visible prevents corrupt media and deployment
problems from silently changing the data population.
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache
from os import PathLike, fspath
from pathlib import Path
from typing import Any, Literal, TypeAlias

import torch
from torch import Tensor

VideoBackend: TypeAlias = Literal["auto", "torchcodec"]
VideoSeekMode: TypeAlias = Literal["exact", "approximate"]


class VideoReaderError(RuntimeError):
    """Base class for video reader failures."""


class VideoBackendUnavailableError(VideoReaderError):
    """Raised when a requested video backend cannot be imported."""


class VideoDecodeError(VideoReaderError):
    """Raised when an available backend cannot open or decode a video."""


@dataclass(frozen=True)
class VideoMetadata:
    """Readable metadata associated with an opened video stream."""

    source: str
    backend: Literal["torchcodec", "decord"]
    num_frames: int
    fps: float | None
    duration_seconds: float | None
    height: int | None
    width: int | None
    device: str = "cpu"
    dimension_order: str = "NCHW"
    dtype: str = "uint8"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _import_torchcodec_decoder() -> type[Any]:
    from torchcodec.decoders import VideoDecoder

    return VideoDecoder


def _import_decord() -> Any:
    import decord

    return decord


@cache
def _torchcodec_accepts_output_dtype(decoder_class: type[Any]) -> bool:
    """Return whether a TorchCodec decoder supports the pre-0.11 dtype option."""
    try:
        parameters = inspect.signature(decoder_class).parameters
    except (TypeError, ValueError):
        return True
    return "output_dtype" in parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )


def _source_name(source: object) -> str:
    if isinstance(source, (str, Path, PathLike)):
        return str(source)
    return f"<{type(source).__name__}>"


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result if result >= 0 else None


def _frame_data(frame_batch: object) -> Tensor:
    data = frame_batch if isinstance(frame_batch, Tensor) else getattr(frame_batch, "data", None)
    if data is None:
        raise TypeError("decoder result has neither tensor data nor a .data tensor")
    return data if isinstance(data, Tensor) else torch.as_tensor(data)


def _validate_nchw_uint8(frames: Tensor, *, backend: str) -> Tensor:
    if frames.ndim == 3:
        frames = frames.unsqueeze(0)
    if frames.ndim != 4:
        raise VideoDecodeError(
            f"{backend} returned a {frames.ndim}D frame tensor; expected [T,C,H,W]"
        )
    if frames.shape[1] != 3:
        raise VideoDecodeError(
            f"{backend} returned {frames.shape[1]} channels; expected RGB [T,3,H,W]"
        )
    if frames.dtype != torch.uint8:
        raise VideoDecodeError(f"{backend} returned dtype {frames.dtype}; expected torch.uint8")
    return frames.detach().to(device="cpu").contiguous()


def _duration_from_metadata(metadata: object) -> float | None:
    direct = _optional_float(getattr(metadata, "duration_seconds", None))
    if direct is not None:
        return direct
    begin = _optional_float(getattr(metadata, "begin_stream_seconds", None))
    end = _optional_float(getattr(metadata, "end_stream_seconds", None))
    if begin is not None and end is not None and end >= begin:
        return end - begin
    return None


class VideoReader:
    """Open and index a video through the project's sole decoding boundary.

    Frames returned by every method are CPU ``torch.uint8`` tensors in NCHW
    order.  ``backend="auto"`` falls back to decord only for a torchcodec import
    failure, never for an error while constructing or using ``VideoDecoder``.
    """

    def __init__(
        self,
        source: str | Path | bytes | Tensor | Any,
        *,
        backend: VideoBackend = "auto",
        num_threads: int = 1,
        seek_mode: VideoSeekMode | None = None,
    ) -> None:
        if backend not in {"auto", "torchcodec"}:
            raise ValueError("backend must be auto or torchcodec")
        if isinstance(num_threads, bool) or not isinstance(num_threads, int) or num_threads <= 0:
            raise ValueError("num_threads must be a positive integer")
        if seek_mode not in {None, "exact", "approximate"}:
            raise ValueError("seek_mode must be exact or approximate")

        self.source = source
        self.requested_backend = backend
        self.num_threads = int(num_threads)
        self.seek_mode = seek_mode
        self._decoder: Any
        self._decord_module: Any | None = None

        try:
            decoder_class = _import_torchcodec_decoder()
        except ImportError as error:
            if backend == "torchcodec":
                raise VideoBackendUnavailableError(
                    "torchcodec was explicitly requested but torchcodec.decoders.VideoDecoder "
                    "could not be imported"
                ) from error
        else:
            self.backend: Literal["torchcodec", "decord"] = "torchcodec"
            self._decoder = self._open_torchcodec(decoder_class)
            self._metadata = self._torchcodec_metadata()
            return

        self.backend = "decord"
        try:
            self._decord_module = _import_decord()
        except ImportError as error:
            raise VideoBackendUnavailableError(
                "torchcodec and its decord fallback could not be imported"
            ) from error
        self._decoder = self._open_decord(self._decord_module)
        self._metadata = self._decord_metadata()

    def _open_torchcodec(self, decoder_class: type[Any]) -> Any:
        try:
            if _torchcodec_accepts_output_dtype(decoder_class):
                if self.seek_mode is not None:
                    return decoder_class(
                        self.source,
                        device="cpu",
                        dimension_order="NCHW",
                        num_ffmpeg_threads=self.num_threads,
                        output_dtype=torch.uint8,
                        seek_mode=self.seek_mode,
                    )
                return decoder_class(
                    self.source,
                    device="cpu",
                    dimension_order="NCHW",
                    num_ffmpeg_threads=self.num_threads,
                    output_dtype=torch.uint8,
                )
            if self.seek_mode is not None:
                return decoder_class(
                    self.source,
                    device="cpu",
                    dimension_order="NCHW",
                    num_ffmpeg_threads=self.num_threads,
                    seek_mode=self.seek_mode,
                )
            return decoder_class(
                self.source,
                device="cpu",
                dimension_order="NCHW",
                num_ffmpeg_threads=self.num_threads,
            )
        except Exception as error:
            raise VideoDecodeError(
                f"torchcodec failed to open video {_source_name(self.source)!r}; "
                "automatic decord fallback is disabled after torchcodec imports successfully"
            ) from error

    def _open_decord(self, decord_module: Any) -> Any:
        try:
            source = fspath(self.source) if isinstance(self.source, PathLike) else self.source
            return decord_module.VideoReader(
                source,
                ctx=decord_module.cpu(0),
                num_threads=self.num_threads,
            )
        except Exception as error:
            raise VideoDecodeError(
                f"decord failed to open video {_source_name(self.source)!r}"
            ) from error

    def _torchcodec_metadata(self) -> VideoMetadata:
        native = getattr(self._decoder, "metadata", None)
        num_frames = _optional_int(getattr(native, "num_frames", None))
        if num_frames is None:
            try:
                num_frames = int(len(self._decoder))
            except Exception as error:
                raise VideoDecodeError("torchcodec failed to determine the video length") from error
        return VideoMetadata(
            source=_source_name(self.source),
            backend="torchcodec",
            num_frames=num_frames,
            fps=_optional_float(getattr(native, "average_fps", None)),
            duration_seconds=_duration_from_metadata(native),
            height=_optional_int(getattr(native, "height", None)),
            width=_optional_int(getattr(native, "width", None)),
        )

    def _decord_metadata(self) -> VideoMetadata:
        try:
            num_frames = int(len(self._decoder))
            fps = _optional_float(self._decoder.get_avg_fps())
        except Exception as error:
            raise VideoDecodeError("decord failed to determine video metadata") from error

        height: int | None = None
        width: int | None = None
        if num_frames:
            try:
                first = self._decord_batch([0])
                height, width = int(first.shape[-2]), int(first.shape[-1])
            except Exception as error:
                raise VideoDecodeError(
                    "decord failed to read the first frame for metadata"
                ) from error
        duration = num_frames / fps if fps not in {None, 0.0} else None
        return VideoMetadata(
            source=_source_name(self.source),
            backend="decord",
            num_frames=num_frames,
            fps=fps,
            duration_seconds=duration,
            height=height,
            width=width,
        )

    @property
    def metadata(self) -> VideoMetadata:
        return self._metadata

    def __len__(self) -> int:
        return self.metadata.num_frames

    def _validate_indices(self, indices: Sequence[int] | Tensor) -> list[int]:
        if isinstance(indices, Tensor):
            if indices.ndim != 1:
                raise ValueError("frame indices must be one-dimensional")
            values = indices.detach().to(device="cpu", dtype=torch.long).tolist()
        else:
            values = [int(index) for index in indices]
        invalid = next((index for index in values if index < 0 or index >= len(self)), None)
        if invalid is not None:
            raise IndexError(f"frame index {invalid} is outside [0, {len(self)})")
        return values

    def _empty_frames(self) -> Tensor:
        height = self.metadata.height or 0
        width = self.metadata.width or 0
        return torch.empty((0, 3, height, width), dtype=torch.uint8, device="cpu")

    def _decord_batch(self, indices: Sequence[int]) -> Tensor:
        batch = self._decoder.get_batch(list(indices))
        if hasattr(batch, "asnumpy"):
            batch = batch.asnumpy()
        tensor = batch if isinstance(batch, Tensor) else torch.as_tensor(batch)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[-1] != 3:
            raise VideoDecodeError(
                f"decord returned shape {tuple(tensor.shape)}; expected [T,H,W,3]"
            )
        return tensor.permute(0, 3, 1, 2)

    def get_frames(self, indices: Sequence[int] | Tensor) -> Tensor:
        """Decode arbitrary frame indices, preserving order and duplicates."""

        values = self._validate_indices(indices)
        if not values:
            return self._empty_frames()
        try:
            if self.backend == "torchcodec":
                native_range = getattr(self._decoder, "get_frames_in_range", None)
                if native_range is None:
                    raise VideoDecodeError("torchcodec decoder does not provide range decoding")
                step = values[1] - values[0] if len(values) > 1 else 1
                is_positive_range = step > 0 and all(
                    right - left == step for left, right in zip(values, values[1:], strict=False)
                )
                if is_positive_range:
                    stop = min(values[-1] + step, len(self))
                    frames = _frame_data(native_range(values[0], stop, step))
                else:
                    frames = torch.cat(
                        [_frame_data(native_range(index, index + 1, 1)) for index in values],
                        dim=0,
                    )
            else:
                frames = self._decord_batch(values)
        except VideoDecodeError:
            raise
        except Exception as error:
            note = (
                "; automatic decord fallback is disabled after torchcodec imports successfully"
                if self.backend == "torchcodec"
                else ""
            )
            raise VideoDecodeError(
                f"{self.backend} failed to decode indices from {_source_name(self.source)!r}{note}"
            ) from error
        return _validate_nchw_uint8(frames, backend=self.backend)

    def get_range(
        self,
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
    ) -> Tensor:
        """Decode a range with standard Python slice boundary semantics."""

        if step == 0:
            raise ValueError("range step cannot be zero")
        normalized = slice(start, stop, step).indices(len(self))
        values = list(range(*normalized))
        if not values:
            return self._empty_frames()
        native_range = getattr(self._decoder, "get_frames_in_range", None)
        if self.backend != "torchcodec" or normalized[2] <= 0 or native_range is None:
            return self.get_frames(values)
        try:
            frames = _frame_data(native_range(*normalized))
        except Exception as error:
            raise VideoDecodeError(
                "torchcodec failed to decode a frame range from "
                f"{_source_name(self.source)!r}; automatic decord fallback is disabled"
            ) from error
        return _validate_nchw_uint8(frames, backend=self.backend)

    def get_all_frames(self) -> Tensor:
        return self.get_range()

    def __getitem__(self, item: int | slice | Sequence[int] | Tensor) -> Tensor:
        if isinstance(item, int):
            index = item if item >= 0 else len(self) + item
            return self.get_frames([index])[0]
        if isinstance(item, slice):
            return self.get_range(item.start, item.stop, item.step or 1)
        return self.get_frames(item)

    def __enter__(self) -> VideoReader:
        return self

    def __exit__(self, *_: object) -> None:
        return None


def open_video(
    source: str | Path | bytes | Tensor | Any,
    *,
    backend: VideoBackend = "auto",
    num_threads: int = 1,
    seek_mode: VideoSeekMode | None = None,
) -> VideoReader:
    """Open a video through the unified reader."""

    return VideoReader(
        source,
        backend=backend,
        num_threads=num_threads,
        seek_mode=seek_mode,
    )


def probe_video(
    source: str | Path | bytes | Tensor | Any,
    *,
    backend: VideoBackend = "auto",
    num_threads: int = 1,
    seek_mode: VideoSeekMode | None = None,
) -> VideoMetadata:
    """Return stream metadata without exposing a backend-specific decoder."""

    return open_video(
        source,
        backend=backend,
        num_threads=num_threads,
        seek_mode=seek_mode,
    ).metadata


def read_video(
    source: str | Path | bytes | Tensor | Any,
    *,
    indices: Sequence[int] | Tensor | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int = 1,
    backend: VideoBackend = "auto",
    num_threads: int = 1,
    seek_mode: VideoSeekMode | None = None,
) -> Tensor:
    """Decode selected frames as CPU NCHW uint8.

    ``indices`` is mutually exclusive with ``start``, ``stop``, and a non-default
    ``step``.  Without either form the entire stream is returned.
    """

    if indices is not None and (start is not None or stop is not None or step != 1):
        raise ValueError("indices cannot be combined with start, stop, or step")
    reader = open_video(
        source,
        backend=backend,
        num_threads=num_threads,
        seek_mode=seek_mode,
    )
    if indices is not None:
        return reader.get_frames(indices)
    return reader.get_range(start, stop, step)


__all__ = [
    "VideoBackend",
    "VideoBackendUnavailableError",
    "VideoDecodeError",
    "VideoMetadata",
    "VideoReader",
    "VideoReaderError",
    "VideoSeekMode",
    "open_video",
    "probe_video",
    "read_video",
]
