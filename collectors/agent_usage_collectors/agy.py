"""Antigravity (AGY) usage collector.

Combines local session/token stats read from Antigravity conversation SQLite
databases with quota allowances from `agy -p /usage --output-format json`.
Reads only: never writes to conversation databases.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
from datetime import datetime, timedelta
from typing import Any, Iterable, Iterator

from .common import (
    MAX_RESPONSE_BYTES,
    auth_missing,
    base_record,
    endpoint_problem,
    print_record,
)

AGENT_ID = "agy"
AGENT_NAME = "Antigravity"
AUTH_HELP = "Run `agy` to sign in. Local stats are still shown."
STATUS_QUOTA_UNAVAILABLE = f"{AGENT_NAME} quota unavailable"
STATUS_DATABASE_ERROR = f"{AGENT_NAME} database error"
STATUS_DB_ERROR = STATUS_DATABASE_ERROR

# ---------------------------------------------------------------------------
# Lightweight Protobuf Wire-Format Decoder
#
# Antigravity stores generation token metadata and step timestamps as binary
# protobuf blobs in SQLite (gen_metadata.data and steps.metadata). To keep this
# collector dependency-free Python (3.10+) as described in collectors/README.md
# (avoiding third-party dependencies such as google.protobuf), this module
# implements a hand-written, stream-based Tag-Length-Value (TLV) wire format
# parser.
# ---------------------------------------------------------------------------

class ProtobufParser:
    """Lightweight hand-written Protobuf wire-format parser.

    Decodes Tag-Length-Value (TLV) fields from a protobuf binary payload.
    Implements Iterable, yielding (field_number, wire_type, value) tuples.
    """

    def __init__(self, source: bytes | bytearray) -> None:
        MAX_PAYLOAD_BYTES = 64 * 1024
        self._data: bytes | bytearray = source
        self._valid: bool = bool(source and len(source) <= MAX_PAYLOAD_BYTES)

    @staticmethod
    def read_varint(stream: io.BytesIO) -> int | None:
        """Decodes an unsigned variable-length integer (varint) from a byte stream.

        In protobuf, each byte of a varint contributes 7 bits of value; the high
        bit (0x80) indicates continuation. Returns None on premature EOF or overflow.
        """
        VARINT_DATA_MASK = 0x7F
        VARINT_CONTINUATION_BIT = 0x80
        VARINT_SHIFT_STEP = 7
        MAX_VARINT_SHIFT = 70

        val = shift = 0
        while True:
            b = stream.read(1)
            if not b:
                return None
            byte = b[0]
            val |= (byte & VARINT_DATA_MASK) << shift
            if not (byte & VARINT_CONTINUATION_BIT):
                return val
            shift += VARINT_SHIFT_STEP
            if shift > MAX_VARINT_SHIFT:
                return None

    def __iter__(self) -> Iterator[tuple[int, int, bytes | int]]:
        """Iterates through Tag-Length-Value (TLV) fields in the protobuf payload.

        Yields (field_number, wire_type, value) where value is:
          - int for WIRE_VARINT
          - bytes for WIRE_LENGTH_DELIMITED, WIRE_FIXED64, and WIRE_FIXED32

        Fields are decoded purely by their wire tag ((field_number << 3) | wire_type),
        so unknown fields or field order variations are handled safely and skipped.
        If the payload is invalid, yields nothing.
        """
        if not self._valid:
            return

        WIRE_VARINT = 0             # int32, int64, uint32, uint64, bool, enum
        WIRE_FIXED64 = 1            # fixed64, sfixed64, double (8 bytes)
        WIRE_LENGTH_DELIMITED = 2   # string, bytes, embedded submessage, packed array
        WIRE_FIXED32 = 5            # fixed32, sfixed32, float (4 bytes)

        FIXED64_BYTE_SIZE = 8
        FIXED32_BYTE_SIZE = 4

        WIRE_TYPE_MASK = 0x07
        WIRE_TYPE_BIT_SHIFT = 3

        MIN_PROTO_FIELD_NUMBER = 1
        MAX_PROTO_FIELD_NUMBER = (1 << 29) - 1

        MAX_PAYLOAD_BYTES = 64 * 1024

        stream = io.BytesIO(self._data)
        while True:
            key = self.read_varint(stream)
            if key is None:
                break
            field_num = key >> WIRE_TYPE_BIT_SHIFT
            wire_type = key & WIRE_TYPE_MASK

            if field_num < MIN_PROTO_FIELD_NUMBER or field_num > MAX_PROTO_FIELD_NUMBER:
                # Illegal protobuf field number; stop parsing defensively
                break

            if wire_type == WIRE_VARINT:
                val = self.read_varint(stream)
                if val is None:
                    break
                yield field_num, wire_type, val
            elif wire_type == WIRE_LENGTH_DELIMITED:
                length = self.read_varint(stream)
                if length is None or length < 0 or length > MAX_PAYLOAD_BYTES:
                    break
                payload = stream.read(length)
                if len(payload) < length:
                    break
                yield field_num, wire_type, payload
            elif wire_type == WIRE_FIXED64:
                payload = stream.read(FIXED64_BYTE_SIZE)
                if len(payload) < FIXED64_BYTE_SIZE:
                    break
                yield field_num, wire_type, payload
            elif wire_type == WIRE_FIXED32:
                payload = stream.read(FIXED32_BYTE_SIZE)
                if len(payload) < FIXED32_BYTE_SIZE:
                    break
                yield field_num, wire_type, payload
            else:
                # Unsupported wire type; stop parsing defensively
                break


def parse_gen_metadata(g_data: bytes) -> tuple[str, int, int, int, str]:
    """Extracts model name, token counts, and response ID from gen_metadata.data.

    Antigravity stores generation records in a nested Protobuf envelope:

        message CortexGeneratorMetadata {
            // FIELD_GEN_INFO (wire type 2, tag 0x0A): Generation payload
            GenerationInfo gen_info = 1;
            // (wire type 2, tag 0x12): Generator status/completeness flags
            bytes is_full = 2;
            // (wire type 2, tag 0x22): Session trajectory UUID
            string trajectory_id = 4;
            // (wire type 2, tag 0x2A): Error string if stream failed
            string error_message = 5;
            // (wire type 2, tag 0x42): Execution hash/identifier
            bytes execution_id = 8;
        }

        message GenerationInfo {
            // (wire type 0, tag 0x18): Internal model enum identifier
            int64 model_enum = 3;
            // FIELD_USAGE_METADATA (wire type 2, tag 0x22): Token usage stats
            ModelUsageStats usage = 4;
            // (wire type 2, tag 0x4A): Prompt breakdown metadata
            bytes prompt_section_metadata = 9;
            // (wire type 2, tag 0x5A): Retry / attempt counters
            bytes retry_info = 11;
            // (wire type 2, tag 0x62): Generation latency timings
            bytes latency_breakdown = 12;
            // (wire type 2, tag 0x8A 0x01): Internal billing cost metrics
            bytes model_cost = 17;
            // FIELD_MODEL_NAME (wire type 2, tag 0x9A 0x01): e.g. "gemini-3.8-flash"
            string model_name = 19;
            // (wire type 2, tag 0xA2 0x01): Trajectory metadata map (e.g. request_id)
            map<string, string> metadata_map = 20;
        }

        message ModelUsageStats {
            // Internal model enum (1298=gemini-3.7-flash, 1318=gemini-3.8-flash)
            enum Model model = 1;
            // FIELD_INPUT_TOKENS (wire type 0, tag 0x10): Fresh un-cached prompt tokens
            int64 input_tokens = 2;
            // FIELD_OUTPUT_TOKENS (wire type 0, tag 0x18): Total output tokens
            // Note: output_tokens == thinking_output_tokens + response_output_tokens
            int64 output_tokens = 3;
            // FIELD_CACHE_READ_TOKENS (wire type 0, tag 0x28): Prompt tokens from cache
            int64 cache_read_tokens = 5;
            // (wire type 0, tag 0x30): Service tier / routing flag (constant 24)
            int64 service_tier = 6;
            // (wire type 2, tag 0x3A): Bot instance UUID (e.g. "bot-<uuid>")
            string agent_id = 7;
            // (wire type 2, tag 0x42): Session key-value pair (e.g. sessionID)
            map<string, string> session_metadata = 8;
            // (wire type 0, tag 0x48): Reasoning tokens (already in output_tokens)
            int64 thinking_output_tokens = 9;
            // (wire type 0, tag 0x50): Candidate text tokens (already in output_tokens)
            int64 response_output_tokens = 10;
            // FIELD_RESPONSE_ID (wire type 2, tag 0x5A): Upstream Gemini response ID
            string response_id = 11;
        }

    Returns:
        (model_name, input_tokens, output_tokens, cache_read_tokens, response_id).
        Defaults to ("", 0, 0, 0, "") if payload is empty, corrupted, or has no counts.
    """
    FIELD_GEN_INFO = 1
    FIELD_MODEL_NAME = 19
    FIELD_USAGE_METADATA = 4
    FIELD_INPUT_TOKENS = 2
    FIELD_OUTPUT_TOKENS = 3
    FIELD_CACHE_READ_TOKENS = 5
    FIELD_THINKING_OUTPUT_TOKENS = 9
    FIELD_RESPONSE_OUTPUT_TOKENS = 10
    FIELD_RESPONSE_ID = 11

    model = ""
    inp = 0
    out = 0
    cache_read = 0
    thinking_out = 0
    response_out = 0
    response_id = ""

    for fn, _, val in ProtobufParser(g_data):
        if fn == FIELD_GEN_INFO and isinstance(val, (bytes, bytearray)):
            for sub_fn, _, sub_val in ProtobufParser(val):
                if sub_fn == FIELD_MODEL_NAME and isinstance(sub_val, (bytes, bytearray)):
                    try:
                        decoded = sub_val.decode("utf-8", errors="replace").strip()
                        if decoded:
                            model = decoded
                    except Exception:
                        pass
                elif sub_fn == FIELD_USAGE_METADATA and isinstance(sub_val, (bytes, bytearray)):
                    for u_fn, _, u_val in ProtobufParser(sub_val):
                        if u_fn == FIELD_INPUT_TOKENS and isinstance(u_val, int):
                            inp = u_val
                        elif u_fn == FIELD_OUTPUT_TOKENS and isinstance(u_val, int):
                            out = u_val
                        elif u_fn == FIELD_CACHE_READ_TOKENS and isinstance(u_val, int):
                            cache_read = u_val
                        elif u_fn == FIELD_THINKING_OUTPUT_TOKENS and isinstance(u_val, int):
                            thinking_out = u_val
                        elif u_fn == FIELD_RESPONSE_OUTPUT_TOKENS and isinstance(u_val, int):
                            response_out = u_val
                        elif u_fn == FIELD_RESPONSE_ID and isinstance(u_val, (bytes, bytearray)):
                            try:
                                decoded = (
                                    u_val.decode("utf-8", errors="replace").strip()
                                )
                                if decoded:
                                    response_id = decoded
                            except Exception:
                                pass

    # If explicit output_tokens (tag 3) is absent or only accounts for candidate text
    # without reasoning tokens, combine thinking and response tokens so output is never undercounted.
    combined_out = thinking_out + response_out
    if combined_out > out:
        out = combined_out

    return (model, inp, out, cache_read, response_id)


def parse_step_timestamp(s_meta: bytes) -> int | None:
    """Extracts unix timestamp seconds from steps.metadata.

    Antigravity stores step execution headers in a nested Protobuf envelope:

        message CortexStepMetadata {
            // FIELD_STEP_HEADER (wire type 2, tag 0x0A): Step creation timestamp
            google.protobuf.Timestamp created_at = 1;
            // (wire type 0, tag 0x18): Step source / author (user, model, tool)
            int64 source = 3;
            // (wire type 2, tag 0x22): Invoked tool name, call ID, JSON arguments
            ToolCallMetadata tool_call = 4;
            // (wire type 2, tag 0x32): Tool execution start timestamp
            google.protobuf.Timestamp started_at = 6;
            // (wire type 2, tag 0x3A): Tool execution completion timestamp
            google.protobuf.Timestamp completed_at = 7;
            // (wire type 2, tag 0x42): UI display availability timestamp
            google.protobuf.Timestamp viewable_at = 8;
            // (wire type 2, tag 0x4A): Step model usage (matches gen_metadata 1.4)
            ModelUsageStats model_usage = 9;
            // (wire type 0, tag 0x58): Generator model enum (e.g. 1298, 1318)
            int64 generator_model = 11;
            // (wire type 2, tag 0x62): Step execution identifier UUID
            string execution_id = 12;
            // (wire type 2, tag 0xA2 0x01): Step session and trajectory identifiers
            StepSessionInfo step_session = 20;
            // (wire type 2, tag 0xD2 0x01): Detailed timing and latency breakdown
            bytes timing_events = 26;
            // (wire type 2, tag 0xE2 0x01): Credit/cost usage for step
            bytes model_cost = 28;
            // (wire type 2, tag 0x82 0x02): Model streaming finish timestamp
            google.protobuf.Timestamp finished_generating_at = 32;
            // (wire type 0, tag 0x88 0x02): Whether user or hook interrupted step
            bool is_interrupting_step = 33;
            // (wire type 2, tag 0x92 0x02): Tool approval status (e.g. "allow")
            PermissionDecision permissions = 34;
        }

        message Timestamp {
            // FIELD_TIMESTAMP_SECONDS (wire type 0, tag 0x08): Unix epoch seconds
            int64 seconds = 1;
            // (wire type 0, tag 0x10): Fractional second nanoseconds
            int32 nanos = 2;
        }

        message ToolCallMetadata {
            // Tool call ID (e.g. "call_12345")
            string call_id = 1;
            // Tool function name (e.g. "run_command", "grep_search")
            string tool_name = 2;
            // JSON-encoded tool invocation parameters
            string arguments_json = 3;
            // Tool execution context and signature
            bytes context = 7;
        }

        message StepSessionInfo {
            // Client session UUID
            string session_id = 1;
            // 0-indexed step number (idx)
            int64 step_index = 2;
            // Sub-step attempt / retry index
            int64 attempt_index = 3;
            // Root conversation UUID
            string conversation_id = 4;
        }

    Returns:
        Unix timestamp in seconds (e.g. 1757088000), or None if missing or non-positive.
    """
    FIELD_STEP_HEADER = 1
    FIELD_TIMESTAMP_SECONDS = 1
    MIN_VALID_TIMESTAMP_SECONDS = 1_000_000_000  # Sep 2001
    MAX_VALID_TIMESTAMP_SECONDS = 2_500_000_000  # Mar 2049

    for fn, _, val in ProtobufParser(s_meta):
        if fn == FIELD_STEP_HEADER and isinstance(val, (bytes, bytearray)):
            for sub_fn, _, sub_val in ProtobufParser(val):
                if (
                    sub_fn == FIELD_TIMESTAMP_SECONDS
                    and isinstance(sub_val, int)
                    and MIN_VALID_TIMESTAMP_SECONDS <= sub_val <= MAX_VALID_TIMESTAMP_SECONDS
                ):
                    return sub_val
    return None


def default_conversations_dirs() -> list[Path]:
    override = os.environ.get("AGY_CONVERSATIONS_DIR")
    if override:
        return [Path(p.strip()) for p in override.split(os.pathsep) if p.strip()]
    agy_home = os.environ.get("AGY_HOME")
    if agy_home:
        return [Path(agy_home) / "conversations"]
    home = Path.home()
    return [
        home / ".gemini" / "antigravity" / "conversations",
        home / ".gemini" / "antigravity-cli" / "conversations",
        home / ".gemini" / "antigravity-ide" / "conversations",
    ]


def stats_from_rows(
    rows: Iterable[tuple[str, bytes | None, bytes | None, float | None]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """Aggregates local token and session metrics from conversation generation rows.

    Each row is: (session_id, gen_metadata_blob, step_metadata_blob, fallback_mtime).
    """
    current_now = now or datetime.now()
    today = current_now.strftime("%Y-%m-%d")
    recent_dates = [(current_now - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(6, -1, -1)]
    recent = {day: {"date": day, "messageCount": 0} for day in recent_dates}
    today_tokens_by_model: dict[str, dict[str, int]] = {}
    model_usage: dict[str, dict[str, int]] = {}
    today_sessions: set[str] = set()
    today_prompts = 0
    today_total_tokens = 0
    total_prompts = 0
    total_sessions: set[str] = set()
    active_days: set[str] = set()
    seen_response_ids: set[str] = set()

    for session_id, g_data, s_meta, fallback_mtime in rows:
        total_sessions.add(session_id)
        if not isinstance(g_data, (bytes, bytearray)):
            continue
        model, inp, out, cache_read, response_id = parse_gen_metadata(g_data)
        if response_id:
            if response_id in seen_response_ids:
                continue
            seen_response_ids.add(response_id)
        total = inp + out + cache_read
        if total <= 0:
            continue

        day = None
        if isinstance(s_meta, (bytes, bytearray)):
            sec = parse_step_timestamp(s_meta)
            if sec:
                try:
                    day = datetime.fromtimestamp(sec).strftime("%Y-%m-%d")
                except Exception:
                    day = None
        if not day and fallback_mtime:
            try:
                day = datetime.fromtimestamp(fallback_mtime).strftime("%Y-%m-%d")
            except Exception:
                day = None
        if not day:
            day = today

        total_prompts += 1
        active_days.add(day)

        model_name = model or "gemini"
        bucket = model_usage.setdefault(
            model_name,
            {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
        )
        bucket["inputTokens"] += inp
        bucket["outputTokens"] += out
        bucket["cacheReadInputTokens"] += cache_read

        if day in recent:
            recent[day]["messageCount"] += total

        if day == today:
            today_prompts += 1
            today_sessions.add(session_id)
            today_total_tokens += total
            t_bucket = today_tokens_by_model.setdefault(
                model_name,
                {"inputTokens": 0, "outputTokens": 0, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0},
            )
            t_bucket["inputTokens"] += inp
            t_bucket["outputTokens"] += out
            t_bucket["cacheReadInputTokens"] += cache_read

    return {
        "todayPrompts": today_prompts,
        "todaySessions": len(today_sessions),
        "todayTotalTokens": today_total_tokens,
        "todayTokensByModel": today_tokens_by_model,
        "recentDays": [recent[d] for d in recent_dates],
        "modelUsage": model_usage,
        "totalPrompts": total_prompts,
        "totalSessions": len(total_sessions),
        "activeDays": len(active_days),
        "activeDates": sorted(active_days),
    }


def fetch_local_stats(
    record: dict[str, Any],
    conversations_dirs: list[Path] | None = None,
) -> bool:
    """Reads Antigravity SQLite conversation databases and updates record with local usage metrics.

    Returns True if local statistics are present, False otherwise.
    """
    def has_stats(stats: dict[str, Any]) -> bool:
        return bool(
            stats.get("totalPrompts", 0) > 0
            or stats.get("todayPrompts", 0) > 0
            or stats.get("todayTotalTokens", 0) > 0
            or stats.get("totalSessions", 0) > 0
        )

    dirs = conversations_dirs if conversations_dirs is not None else default_conversations_dirs()
    db_paths: list[Path] = []
    db_errors: list[str] = []
    for d in dirs:
        try:
            if d.is_dir():
                db_paths.extend(sorted(d.glob("*.db")))
        except OSError as exc:
            db_errors.append(f"{d.name}: {exc}")

    if db_paths:
        def iter_rows() -> Iterator[tuple[str, bytes, bytes | None, float | None]]:
            for db_path in db_paths:
                # Use full path as session_id so CLI and IDE databases sharing a stem are not merged.
                session_id = str(db_path)
                try:
                    st = db_path.stat()
                    if st.st_size == 0:
                        continue
                    mtime = st.st_mtime
                except OSError:
                    mtime = None
                try:
                    conn = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro&immutable=1", uri=True, timeout=2)
                    try:
                        c = conn.cursor()
                        # A scalar subquery on steps guarantees exactly 1 row per generation record
                        # regardless of schema evolution, preventing 1:N join row inflation.
                        c.execute(
                            "SELECT g.data, ("
                            "    SELECT s.metadata FROM steps s "
                            "    WHERE s.idx = g.idx AND s.metadata IS NOT NULL LIMIT 1"
                            ") FROM gen_metadata g WHERE g.data IS NOT NULL"
                        )
                        for g_data, s_meta in c.fetchall():
                            if not isinstance(g_data, (bytes, bytearray)):
                                continue
                            clean_s_meta = s_meta if isinstance(s_meta, (bytes, bytearray)) else None
                            yield (session_id, bytes(g_data), bytes(clean_s_meta) if clean_s_meta is not None else None, mtime)
                    finally:
                        conn.close()
                except (sqlite3.Error, OSError) as exc:
                    db_errors.append(f"{db_path.name}: {exc}")
                    continue

        stats = stats_from_rows(iter_rows())
    else:
        stats = stats_from_rows([])

    has_local_stats = has_stats(stats)
    record.update(stats)
    record["scope"] = "device"
    record["hasLocalStats"] = has_local_stats
    record["hasPromptStats"] = has_local_stats

    if db_errors:
        first_err = db_errors[0]
        suffix = f" (and {len(db_errors) - 1} other files)" if len(db_errors) > 1 else ""
        record["usageStatusText"] = STATUS_DATABASE_ERROR
        record["authHelpText"] = f"Failed to read conversation database: {first_err}{suffix}"[:300]

    return has_local_stats


def parse_quota_groups(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    WINDOW_SPECS: dict[str, tuple[str, str, int]] = {
        # window: (base_title, label_suffix, order)
        "5h": ("Session", "(5-hour)", 0),
        "weekly": ("Weekly", "(7-day)", 1),
    }

    limits: list[dict[str, Any]] = []
    for group in groups:
        if not isinstance(group, dict): continue

        # The first group represents the primary quota (Session & Weekly).
        # Subsequent groups represent auxiliary model quotas (e.g. Claude & GPT).
        group_name = str(group.get("name") or "").strip()
        model_family = "" if not limits else re.sub(r"\s+models?$", "", group_name, flags=re.IGNORECASE).strip()

        buckets = group.get("buckets")
        if not isinstance(buckets, list): continue

        group_limits: list[tuple[int, dict[str, Any]]] = []
        for bucket in buckets:
            if not isinstance(bucket, dict): continue
            try:
                remaining = float(bucket.get("remaining_fraction"))
            except (ValueError, TypeError):
                continue

            window = bucket.get("window")
            if window in WINDOW_SPECS:
                base_title, suffix, order = WINDOW_SPECS[window]
            else:
                base_title = str(bucket.get("name") or window or "")
                suffix = ""
                order = 2

            title = f"{model_family} {base_title}".strip()
            label = f"{title} {suffix}".strip()

            group_limits.append((order, {
                "label": label,
                "title": title,
                "percent": max(0.0, min(1.0, round(1.0 - remaining, 4))),
                "resetsAt": str(bucket.get("reset_time") or ""),
            }))

        # Order session/short windows (5h) before weekly windows, matching Codex
        group_limits.sort(key=lambda item: item[0])
        limits.extend(item[1] for item in group_limits)

    return limits


def fetch_quota(
    record: dict[str, Any],
    command_override: list[str] | None = None,
    timeout_seconds: float = 8.0,
) -> bool:
    """Runs `agy -p /usage --output-format json` and updates record with limits or problem status.

    Returns True if quota information was successfully retrieved, False otherwise.
    """
    had_db_error = record.get("usageStatusText") == STATUS_DATABASE_ERROR
    previous_status = record.get("usageStatusText")
    previous_help = record.get("authHelpText")

    try:
        if command_override:
            cmd = command_override
        else:
            agy_bin = os.environ.get("AGY_CLI_PATH") or shutil.which("agy")
            if not agy_bin:
                auth_missing(record, status="Waiting for agy", help_text="agy not found in PATH")
                return False
            cmd = [agy_bin, "-p", "/usage", "--output-format", "json"]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text="Usage probe timed out", retry=True)
            return False
        except OSError as exc:
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text=str(exc))
            return False

        if proc.returncode != 0:
            err = proc.stderr.strip() or f"Command failed with exit code {proc.returncode}"
            if "not found" in err.lower() or "sign in" in err.lower():
                auth_missing(record, status="Waiting for agy", help_text=err or AUTH_HELP)
            else:
                endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text=err)
            return False

        stdout = proc.stdout
        if len(stdout) > MAX_RESPONSE_BYTES:
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text="Response payload too large")
            return False

        try:
            payload = json.loads(stdout)
        except (ValueError, TypeError) as exc:
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text=f"Invalid JSON output: {exc}")
            return False

        if not isinstance(payload, dict):
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text="Unexpected JSON shape")
            return False

        if payload.get("status") != "SUCCESS":
            resp = str(payload.get("response") or "Usage query failed")
            if "not found" in resp.lower() or "sign in" in resp.lower():
                auth_missing(record, status="Waiting for agy", help_text=resp)
            else:
                endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text=resp)
            return False

        command = payload.get("command")
        if not isinstance(command, dict):
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text="Unexpected command payload")
            return False

        data = command.get("data")
        if not isinstance(data, dict):
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text="Unexpected data payload")
            return False

        groups = data.get("groups", [])
        if not isinstance(groups, list):
            endpoint_problem(record, status=STATUS_QUOTA_UNAVAILABLE, help_text="No quota groups returned")
            return False

        record["limits"] = parse_quota_groups(groups)
        return True
    finally:
        if had_db_error:
            record["usageStatusText"] = previous_status
            record["authHelpText"] = previous_help


def collect() -> dict[str, Any]:
    record = base_record(AGENT_ID, AGENT_NAME, "Antigravity")
    local_ok = fetch_local_stats(record)
    quota_ok = fetch_quota(record)
    if local_ok or quota_ok:
        record["ready"] = True
    return record


def main() -> None:
    print_record(collect())


if __name__ == "__main__":
    main()
