# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import contextlib
import logging

from vllm.v1.request import Request, RequestStatus

logger = logging.getLogger(__name__)


def remove_all(lst: list, items_to_remove: set) -> list:
    """Remove all items from a list that are in the items_to_remove set.

    This method optimizes for the common case of removing a single item,
    falling back to list comprehension for multiple items.

    Args:
        lst: The list to remove items from
        items_to_remove: Set of items to remove

    Returns:
        Either the modified original list (for single item removal) or
        a new list (for multiple item removal). Callers should use the
        returned value.

    Note:
        For single item removal, this modifies the original list in-place
        and returns it. For multiple items, it creates and returns a new list.
    """
    if not items_to_remove:
        return lst

    if len(items_to_remove) == 1:
        # Fast path for single item removal (most common case)
        item = next(iter(items_to_remove))
        with contextlib.suppress(ValueError):
            lst.remove(item)
        return lst
    # For multiple items, use list comprehension
    return [item for item in lst if item not in items_to_remove]


def _is_opuslm_tts_request(
    request: Request,
    opuslm_tts_task_ids: set[int] | None,
) -> bool:
    tts_task_ids = opuslm_tts_task_ids or {81, 82, 88, 89}

    prompt_token_ids = request.prompt_token_ids or []
    if len(prompt_token_ids) >= 2 and int(prompt_token_ids[1]) in tts_task_ids:
        return True

    sampling_params = request.sampling_params
    extra_args = sampling_params.extra_args if sampling_params else None
    if not isinstance(extra_args, dict):
        return False

    # Text-only modes (ASR, text dialogue) don't produce audio flush pads,
    # so EOS deferral must be skipped — otherwise the request never stops.
    mode = extra_args.get("mode", "")
    if isinstance(mode, str) and mode in ("audio_text", "text_text"):
        return False

    task = extra_args.get("task")
    if not isinstance(task, str):
        return False
    task_norm = task.strip().lower()
    return task_norm in (
        "tts",
        "plain_tts",
        "codec_ssl_tts",
        "codec_ssl_tts_task",
        "codec_ssl_plain_tts",
        "codec_ssl_plain_tts_task",
        "audio_dialogue",
        "audio_dialogue_task",
        "text_dialogue",
        "text_dialogue_task",
    )


def _should_defer_opuslm_eos_stop(
    request: Request,
    opuslm_delay_steps: int,
    opuslm_tts_task_ids: set[int] | None,
) -> bool:
    if opuslm_delay_steps <= 0:
        return False
    if not _is_opuslm_tts_request(request, opuslm_tts_task_ids):
        return False
    if request.num_output_tokens <= opuslm_delay_steps:
        return True

    # ARDelay flush completion criterion on stream-0:
    # final EOS should be preceded by exactly `nq - 1` pad(0) tokens.
    window = request.output_token_ids[-(opuslm_delay_steps + 1):-1]
    if len(window) < opuslm_delay_steps:
        return True
    return any(tok != 0 for tok in window)


def check_stop(
    request: Request,
    max_model_len: int,
    *,
    opuslm_delay_steps: int | None = None,
    opuslm_tts_task_ids: set[int] | None = None,
) -> bool:
    assert not request.pooling_params

    sampling_params = request.sampling_params
    assert sampling_params is not None

    if request.num_output_tokens < sampling_params.min_tokens:
        return False

    last_token_id = request.output_token_ids[-1]

    defer_opuslm_eos = False
    if last_token_id == request.eos_token_id:
        defer_opuslm_eos = _should_defer_opuslm_eos_stop(
            request,
            opuslm_delay_steps=opuslm_delay_steps if opuslm_delay_steps is not None else 8,
            opuslm_tts_task_ids=opuslm_tts_task_ids,
        )

    if not sampling_params.ignore_eos and last_token_id == request.eos_token_id:
        if defer_opuslm_eos:
            return False
        request.status = RequestStatus.FINISHED_STOPPED
        return True

    if last_token_id in (sampling_params.stop_token_ids or ()):
        if last_token_id == request.eos_token_id and defer_opuslm_eos:
            return False
        request.status = RequestStatus.FINISHED_STOPPED
        return True
    if (
        request.num_tokens >= max_model_len
        or request.num_output_tokens >= request.max_tokens
    ):
        request.status = RequestStatus.FINISHED_LENGTH_CAPPED
        return True
    return False
