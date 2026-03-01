"""
ScriptGuard vs VirusTotal — Comparative Benchmark
==================================================
Runs the 180-sample benchmark against ScriptGuard's /classify endpoint
AND submits the same samples to VirusTotal for engine-based analysis.
Produces a side-by-side comparison with full metrics for both systems.

VirusTotal workflow:
  1. Upload file → get scan ID
  2. Poll /analyses/{id} until status == "completed"
  3. Parse results: malicious if vt_positive_engines >= --vt-threshold (default 1)

Usage:
    python compare_vt_benchmark.py \\
        --vt-key YOUR_VT_API_KEY \\
        --url http://localhost:8000 \\
        [--threshold 0.5] \\
        [--vt-threshold 1] \\
        [--vt-poll-interval 15] \\
        [--vt-max-wait 300] \\
        [--workers 4] \\
        [--json-out results.json] \\
        [--html-out report.html] \\
        [--verbose]

Notes:
  - VirusTotal free API: 4 requests/min, 500/day.
    The script rate-limits automatically (--vt-ratelimit, default 15s).
  - Each code sample is submitted as a .py file for proper engine coverage.
  - VirusTotal may not have seen synthetic test code before — expect many
    "undetected" results on obfuscated samples. That is valuable data.
  - Results JSON is compatible with the original benchmark format plus
    extra "vt_*" keys on each row.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests library required:  pip install requests")

# Import test cases from existing benchmark module
sys.path.insert(0, str(Path(__file__).parent))
try:
    from test_benchmark_codebert import (
        ALL_SAMPLES, BENIGN_SAMPLES, MALICIOUS_SAMPLES,
        OBFUSCATED_MALICIOUS_SAMPLES, TestCase,
    )
except ImportError as e:
    sys.exit(
        f"Cannot import test_benchmark_codebert: {e}\n"
        f"Ensure test_benchmark_codebert.py is in the same directory."
    )

# ============================================================================
# Dataclasses
# ============================================================================

@dataclass
class SGResult:
    """ScriptGuard classification result."""
    label:          str
    confidence:     float
    malicious_prob: float
    latency_ms:     float
    error:          Optional[str] = None


@dataclass
class VTEngineResult:
    """Per-engine VirusTotal result."""
    engine:   str
    category: str   # "malicious" | "undetected" | "timeout" | "type-unsupported"
    result:   Optional[str] = None


@dataclass
class VTResult:
    """VirusTotal scan result for one sample."""
    scan_id:          str = ""
    status:           str = ""             # "queued"|"in-progress"|"completed"|"error"
    total_engines:    int = 0
    positive_engines: int = 0
    undetected:       int = 0
    suspicious:       int = 0
    timeout_engines:  int = 0
    unsupported:      int = 0
    label:            str = ""             # "malicious" | "benign" (after threshold)
    engine_details:   list[VTEngineResult] = field(default_factory=list)
    latency_ms:       float = 0.0
    error:            Optional[str] = None
    file_sha256:      str = ""


@dataclass
class ComparisonRow:
    """Full comparison row for one test sample."""
    description: str
    expected:    str
    sg:          Optional[SGResult]
    vt:          Optional[VTResult]


# ============================================================================
# ScriptGuard client
# ============================================================================

def sg_classify(base_url: str, code: str,
                threshold: Optional[float]) -> SGResult:
    payload: dict = {"code": code}
    if threshold is not None:
        payload["threshold"] = threshold
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/classify", json=payload, timeout=60
        )
        elapsed = (time.perf_counter() - t0) * 1000
        if resp.status_code != 200:
            return SGResult("", 0.0, 0.0, elapsed,
                            error=f"HTTP {resp.status_code}: {resp.text[:200]}")
        d = resp.json()
        return SGResult(
            label=d["label"],
            confidence=d["confidence"],
            malicious_prob=d["malicious_prob"],
            latency_ms=elapsed,
        )
    except requests.exceptions.ConnectionError:
        elapsed = (time.perf_counter() - t0) * 1000
        return SGResult("", 0.0, 0.0, elapsed,
                        error="Connection refused — is the API running?")
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        return SGResult("", 0.0, 0.0, elapsed, error=str(exc))


# ============================================================================
# VirusTotal client
# ============================================================================

VT_BASE = "https://www.virustotal.com/api/v3"

# Rate-limiter: tracks timestamps of recent VT requests.
_vt_lock = threading.Lock()
_vt_timestamps: list[float] = []


def _vt_rate_limit(min_interval_s: float) -> None:
    """Block until enough time has passed since the last VT request."""
    with _vt_lock:
        now = time.monotonic()
        # Purge timestamps older than 60 seconds
        while _vt_timestamps and now - _vt_timestamps[0] > 60:
            _vt_timestamps.pop(0)
        if _vt_timestamps:
            wait = min_interval_s - (now - _vt_timestamps[-1])
            if wait > 0:
                time.sleep(wait)
        _vt_timestamps.append(time.monotonic())


def _vt_headers(api_key: str) -> dict:
    return {"x-apikey": api_key}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def vt_upload(api_key: str, code: str,
              rate_limit_s: float) -> tuple[str, str]:
    """
    Upload a Python source file to VirusTotal.
    Returns (scan_id, sha256) or raises RuntimeError.
    """
    _vt_rate_limit(rate_limit_s)
    content = code.encode("utf-8", errors="replace")
    sha = _sha256(content)
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as fh:
            resp = requests.post(
                f"{VT_BASE}/files",
                headers=_vt_headers(api_key),
                files={"file": (f"sample_{sha[:8]}.py", fh, "text/plain")},
                timeout=60,
            )
        resp.raise_for_status()
        data = resp.json()
        scan_id = data["data"]["id"]
        return scan_id, sha
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def vt_get_analysis(api_key: str, scan_id: str,
                    rate_limit_s: float) -> dict:
    """Fetch analysis result from VirusTotal."""
    _vt_rate_limit(rate_limit_s)
    resp = requests.get(
        f"{VT_BASE}/analyses/{scan_id}",
        headers=_vt_headers(api_key),
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def vt_check_existing(api_key: str, sha256: str,
                      rate_limit_s: float) -> Optional[dict]:
    """
    Check if VirusTotal already has a result for this sha256.
    Returns the file report dict or None.
    """
    _vt_rate_limit(rate_limit_s)
    try:
        resp = requests.get(
            f"{VT_BASE}/files/{sha256}",
            headers=_vt_headers(api_key),
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def _parse_vt_report(data: dict, vt_threshold: int) -> VTResult:
    """
    Parse a VirusTotal file report (from /files/{sha256}) into VTResult.
    """
    try:
        stats = data["data"]["attributes"]["last_analysis_stats"]
        engines_raw = data["data"]["attributes"].get("last_analysis_results", {})

        positives   = stats.get("malicious", 0)
        suspicious  = stats.get("suspicious", 0)
        undetected  = stats.get("undetected", 0)
        timeout_e   = stats.get("timeout", 0)
        unsupported = stats.get("type-unsupported", 0)
        total       = positives + suspicious + undetected + timeout_e + unsupported

        engine_details = [
            VTEngineResult(
                engine=eng,
                category=info.get("category", ""),
                result=info.get("result"),
            )
            for eng, info in engines_raw.items()
        ]

        label = "malicious" if positives >= vt_threshold else "benign"
        return VTResult(
            status="completed",
            total_engines=total,
            positive_engines=positives,
            undetected=undetected,
            suspicious=suspicious,
            timeout_engines=timeout_e,
            unsupported=unsupported,
            label=label,
            engine_details=engine_details,
            file_sha256=data["data"].get("id", ""),
        )
    except Exception as exc:
        return VTResult(status="error", error=f"Parse error: {exc}")


def _parse_vt_analysis(data: dict, vt_threshold: int,
                        sha256: str) -> VTResult:
    """
    Parse a VirusTotal analysis result (from /analyses/{id}) into VTResult.
    """
    try:
        attrs = data["data"]["attributes"]
        status = attrs.get("status", "unknown")
        if status != "completed":
            return VTResult(status=status,
                            file_sha256=sha256)
        stats = attrs.get("stats", {})
        results_raw = attrs.get("results", {})

        positives   = stats.get("malicious", 0)
        suspicious  = stats.get("suspicious", 0)
        undetected  = stats.get("undetected", 0)
        timeout_e   = stats.get("timeout", 0)
        unsupported = stats.get("type-unsupported", 0)
        total       = positives + suspicious + undetected + timeout_e + unsupported

        engine_details = [
            VTEngineResult(
                engine=eng,
                category=info.get("category", ""),
                result=info.get("result"),
            )
            for eng, info in results_raw.items()
        ]

        label = "malicious" if positives >= vt_threshold else "benign"
        return VTResult(
            status="completed",
            total_engines=total,
            positive_engines=positives,
            undetected=undetected,
            suspicious=suspicious,
            timeout_engines=timeout_e,
            unsupported=unsupported,
            label=label,
            engine_details=engine_details,
            file_sha256=sha256,
        )
    except Exception as exc:
        return VTResult(status="error", error=f"Parse error: {exc}",
                        file_sha256=sha256)


def vt_scan_sample(api_key: str, code: str, vt_threshold: int,
                   rate_limit_s: float, poll_interval_s: float,
                   max_wait_s: float) -> VTResult:
    """
    Full VirusTotal scan pipeline for one code sample.
    1. Compute SHA256 → check if already cached on VT.
    2. Upload if not cached.
    3. Poll until completed or timeout.
    """
    t0 = time.perf_counter()
    content = code.encode("utf-8", errors="replace")
    sha = _sha256(content)

    # Step 1: check cache
    cached = vt_check_existing(api_key, sha, rate_limit_s)
    if cached:
        result = _parse_vt_report(cached, vt_threshold)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        result.file_sha256 = sha
        return result

    # Step 2: upload
    try:
        scan_id, sha = vt_upload(api_key, code, rate_limit_s)
    except Exception as exc:
        return VTResult(
            status="error",
            error=f"Upload failed: {exc}",
            latency_ms=(time.perf_counter() - t0) * 1000,
            file_sha256=sha,
        )

    # Step 3: poll
    deadline = time.monotonic() + max_wait_s
    while time.monotonic() < deadline:
        time.sleep(poll_interval_s)
        try:
            analysis = vt_get_analysis(api_key, scan_id, rate_limit_s)
            status = analysis["data"]["attributes"].get("status", "")
            if status == "completed":
                result = _parse_vt_analysis(analysis, vt_threshold, sha)
                result.latency_ms = (time.perf_counter() - t0) * 1000
                result.scan_id = scan_id
                return result
            # still in-progress or queued — keep polling
        except Exception as exc:
            return VTResult(
                status="error",
                error=f"Poll failed: {exc}",
                scan_id=scan_id,
                latency_ms=(time.perf_counter() - t0) * 1000,
                file_sha256=sha,
            )

    return VTResult(
        status="timeout",
        error=f"Analysis not completed within {max_wait_s}s",
        scan_id=scan_id,
        latency_ms=(time.perf_counter() - t0) * 1000,
        file_sha256=sha,
    )


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    total       = tp + tn + fp + fn
    accuracy    = (tp + tn) / total if total > 0 else 0.0
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr         = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    mcc_denom   = (
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    ) ** 0.5
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0
    return dict(
        accuracy=accuracy, precision=precision, recall=recall, f1=f1,
        specificity=specificity, fpr=fpr, fnr=fnr, mcc=mcc,
        tp=tp, tn=tn, fp=fp, fn=fn, total=total,
    )


def metrics_from_rows(rows: list[ComparisonRow],
                      system: str) -> dict:
    """Compute TP/TN/FP/FN for 'sg' or 'vt'."""
    tp = tn = fp = fn = 0
    for r in rows:
        result = r.sg if system == "sg" else r.vt
        if result is None or result.error or result.label == "":
            continue
        expected = r.expected
        got = result.label
        if   expected == "malicious" and got == "malicious": tp += 1
        elif expected == "benign"    and got == "benign":    tn += 1
        elif expected == "benign"    and got == "malicious": fp += 1
        elif expected == "malicious" and got == "benign":    fn += 1
    return compute_metrics(tp, tn, fp, fn)


# ============================================================================
# Color helpers (ANSI)
# ============================================================================

def _green(s: str) -> str: return f"\033[92m{s}\033[0m"
def _red(s:   str) -> str: return f"\033[91m{s}\033[0m"
def _yellow(s: str) -> str: return f"\033[93m{s}\033[0m"
def _bold(s:  str) -> str: return f"\033[1m{s}\033[0m"
def _col(s: str, ok: bool) -> str: return _green(s) if ok else _red(s)


# ============================================================================
# Core benchmark runner
# ============================================================================

def run_comparison(
    base_url:       str,
    sg_threshold:   Optional[float],
    vt_api_key:     Optional[str],
    vt_threshold:   int,
    vt_rate_limit:  float,
    vt_poll:        float,
    vt_max_wait:    float,
    workers:        int,
    verbose:        bool,
) -> list[ComparisonRow]:
    """
    Run full comparison benchmark.
    Returns list of ComparisonRow — one per test sample.
    """

    print(f"\n{'='*72}")
    print(_bold("  ScriptGuard vs VirusTotal — Comparative Benchmark"))
    print(f"  ScriptGuard API : {base_url}")
    print(f"  SG threshold    : {sg_threshold if sg_threshold is not None else 'model default'}")
    if vt_api_key:
        print(f"  VT API key      : {'*'*8}{vt_api_key[-4:]}")
        print(f"  VT threshold    : >= {vt_threshold} engine(s) positive")
        print(f"  VT rate limit   : {vt_rate_limit}s between requests")
        print(f"  VT poll interval: {vt_poll}s")
        print(f"  VT max wait     : {vt_max_wait}s per sample")
    else:
        print("  VT API key      : NOT PROVIDED — skipping VirusTotal")
    print(
        f"  Samples         : {len(BENIGN_SAMPLES)} benign + "
        f"{len(MALICIOUS_SAMPLES)} malicious + "
        f"{len(OBFUSCATED_MALICIOUS_SAMPLES)} obfuscated = "
        f"{len(ALL_SAMPLES)} total"
    )
    print(f"{'='*72}\n")

    # ── Pre-flight: ScriptGuard readiness ──────────────────────────────────
    try:
        r = requests.get(f"{base_url}/ready", timeout=5)
        if r.status_code != 200:
            print(_yellow(f"WARNING: /ready returned {r.status_code} — {r.text}"))
    except requests.exceptions.ConnectionError:
        print(_red(f"ERROR: Cannot connect to {base_url}"))
        sys.exit(1)

    # ── Step 1: Run ScriptGuard classifications in parallel ────────────────
    print(_bold("  [1/2] Running ScriptGuard classifications..."))
    sg_results: dict[int, SGResult] = {}

    def _sg_worker(idx_tc: tuple[int, TestCase]) -> tuple[int, SGResult]:
        idx, tc = idx_tc
        return idx, sg_classify(base_url, tc.code, sg_threshold)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_sg_worker, (i, tc)): i
            for i, tc in enumerate(ALL_SAMPLES)
        }
        done = 0
        for fut in as_completed(futures):
            idx, res = fut.result()
            sg_results[idx] = res
            done += 1
            pct = 100 * done // len(ALL_SAMPLES)
            print(f"\r  SG: {done}/{len(ALL_SAMPLES)} ({pct}%)  ", end="", flush=True)
    print()

    # ── Step 2: Run VirusTotal scans (sequential due to rate limits) ───────
    vt_results: dict[int, Optional[VTResult]] = {
        i: None for i in range(len(ALL_SAMPLES))
    }

    if vt_api_key:
        print(_bold(f"\n  [2/2] Submitting to VirusTotal "
                    f"({len(ALL_SAMPLES)} samples, "
                    f"{vt_rate_limit}s between requests)..."))
        print(_yellow(
            f"  ETA: ~{len(ALL_SAMPLES) * (vt_rate_limit + vt_poll):.0f}s "
            f"if all are new submissions"
        ))

        for i, tc in enumerate(ALL_SAMPLES):
            print(
                f"\r  VT: {i+1}/{len(ALL_SAMPLES)} — "
                f"{tc.description[:50]:<50}",
                end="", flush=True,
            )
            vt_res = vt_scan_sample(
                api_key=vt_api_key,
                code=tc.code,
                vt_threshold=vt_threshold,
                rate_limit_s=vt_rate_limit,
                poll_interval_s=vt_poll,
                max_wait_s=vt_max_wait,
            )
            vt_results[i] = vt_res
        print()
    else:
        print("  [2/2] VirusTotal: SKIPPED (no API key)")

    # ── Assemble ComparisonRows ────────────────────────────────────────────
    rows = [
        ComparisonRow(
            description=ALL_SAMPLES[i].description,
            expected=ALL_SAMPLES[i].expected,
            sg=sg_results.get(i),
            vt=vt_results.get(i),
        )
        for i in range(len(ALL_SAMPLES))
    ]

    return rows


# ============================================================================
# Report printer
# ============================================================================

def print_comparison_report(
    rows:         list[ComparisonRow],
    vt_available: bool,
    vt_threshold: int,
) -> None:

    has_vt = vt_available and any(r.vt is not None for r in rows)
    W = 46

    # ── Per-sample table ───────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(_bold("  SAMPLE-BY-SAMPLE RESULTS"))
    print(f"{'='*72}")

    if has_vt:
        hdr = (f"  {'DESCRIPTION':<{W}}  {'EXPECTED':<10}  "
               f"{'SG':<12}  {'VT':<12}  {'AGREE':5}")
    else:
        hdr = (f"  {'DESCRIPTION':<{W}}  {'EXPECTED':<10}  "
               f"{'SG LABEL':<12}  {'SG PROB':>8}  {'SG CONF':>8}  {'MS':>6}  PASS")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    prev_group = ""
    for r in rows:
        group = ("obfuscated" if r.description.startswith("obf:")
                 else r.expected)
        if group != prev_group:
            label = {
                "benign":     "── BENIGN ──",
                "malicious":  "── MALICIOUS (plain) ──",
                "obfuscated": "── MALICIOUS (obfuscated) ──",
            }.get(group, group)
            print(f"\n  {_bold(label)}")
            prev_group = group

        sg = r.sg
        vt = r.vt
        expected = r.expected
        desc = (r.description[:W-3] + "...") if len(r.description) > W else r.description

        if has_vt:
            sg_label  = sg.label  if sg  and not sg.error  else "ERROR"
            vt_label  = vt.label  if vt  and not vt.error and vt.status == "completed" \
                        else (vt.status if vt else "N/A")
            sg_ok = sg_label == expected
            vt_ok = vt_label == expected
            agree = sg_label == vt_label

            sg_col = _col(sg_label, sg_ok)
            vt_col = _col(vt_label, vt_ok)
            agree_col = _green("YES") if agree else _yellow("NO")

            vt_detail = ""
            if vt and vt.status == "completed":
                vt_detail = f"[{vt.positive_engines}/{vt.total_engines}]"

            print(
                f"  {desc:<{W}}  {expected:<10}  "
                f"{sg_col:<22}  "
                f"{vt_col:<22}{vt_detail:<10}  "
                f"{agree_col}"
            )

            if sg and vt and vt.status == "completed":
                sg_correct = sg.label == expected
                vt_correct = vt.label == expected
                if not sg_correct and vt_correct:
                    print(_yellow(f"    ^ SG missed, VT caught it"))
                elif sg_correct and not vt_correct:
                    print(_yellow(f"    ^ SG correct, VT missed it"))
                elif not sg_correct and not vt_correct:
                    print(_red(f"    ^ Both missed"))
        else:
            if sg is None:
                print(f"  {desc:<{W}}  {expected:<10}  {'N/A':<12}  {'':>8}  {'':>8}  {'':>6}  N/A")
                continue
            sg_ok = sg.label == expected
            if sg.error:
                print(
                    f"  {desc:<{W}}  {expected:<10}  "
                    f"{_red('ERROR'):<22}  {sg.latency_ms:>6.0f}ms"
                )
            else:
                print(
                    f"  {desc:<{W}}  {expected:<10}  "
                    f"{_col(sg.label, sg_ok):<22}  "
                    f"{sg.malicious_prob:>8.3f}  "
                    f"{sg.confidence:>8.3f}  "
                    f"{sg.latency_ms:>6.0f}ms  "
                    f"{_col('PASS', sg_ok) if sg_ok else _col('FAIL', False)}"
                )

    # ── Confusion matrices ─────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(_bold("  CONFUSION MATRICES"))
    print(f"{'='*72}")

    sg_m = metrics_from_rows(rows, "sg")
    _print_confusion(sg_m, "ScriptGuard")

    if has_vt:
        vt_m = metrics_from_rows(rows, "vt")
        _print_confusion(vt_m, f"VirusTotal  (>={vt_threshold} engine)")

    # ── Metrics comparison table ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print(_bold("  METRICS COMPARISON"))
    print(f"{'='*72}")

    if has_vt:
        vt_m = metrics_from_rows(rows, "vt")
        _print_metrics_comparison(sg_m, vt_m, vt_threshold)
    else:
        _print_metrics_single(sg_m)

    # ── Per-group pass rates ───────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(_bold("  PASS RATES BY GROUP"))
    print(f"{'='*72}")

    benign_rows = [r for r in rows if r.expected == "benign"]
    plain_rows  = [r for r in rows
                   if r.expected == "malicious"
                   and not r.description.startswith("obf:")]
    obf_rows    = [r for r in rows if r.description.startswith("obf:")]

    groups = [
        ("Benign              ", benign_rows),
        ("Malicious (plain)   ", plain_rows),
        ("Malicious (obfusc.) ", obf_rows),
    ]

    for gname, grows in groups:
        sg_pass  = _pass_rate(grows, "sg")
        vt_pass  = _pass_rate(grows, "vt") if has_vt else "N/A"
        if has_vt:
            print(f"  {gname}  SG: {sg_pass:<20}  VT: {vt_pass}")
        else:
            print(f"  {gname}  SG: {sg_pass}")

    # ── Disagreements summary ──────────────────────────────────────────────
    if has_vt:
        print(f"\n{'='*72}")
        print(_bold("  DISAGREEMENTS (SG vs VT)"))
        print(f"{'='*72}")

        disagree = [
            r for r in rows
            if r.sg and r.vt
            and r.sg.label and r.vt.label
            and not r.sg.error
            and r.vt.status == "completed"
            and r.sg.label != r.vt.label
        ]

        if not disagree:
            print(_green("  None — both systems agreed on all samples."))
        else:
            print(f"  {len(disagree)} disagreement(s):\n")
            for r in disagree:
                sg_ok = r.sg.label == r.expected
                vt_ok = r.vt.label == r.expected
                winner = ("SG correct" if sg_ok and not vt_ok
                          else "VT correct" if vt_ok and not sg_ok
                          else "both wrong")
                print(
                    f"  {'['+winner+']':<14}  {r.description[:50]:<52}"
                    f"  expected={r.expected}  "
                    f"SG={r.sg.label}  "
                    f"VT={r.vt.label}({r.vt.positive_engines}/{r.vt.total_engines})"
                )

    # ── VT engine leaderboard ──────────────────────────────────────────────
    if has_vt:
        print(f"\n{'='*72}")
        print(_bold("  VIRUSTOTAL ENGINE LEADERBOARD"))
        print(_bold("  (engines that correctly flagged malicious samples)"))
        print(f"{'='*72}")

        malicious_rows = [
            r for r in rows
            if r.expected == "malicious"
            and r.vt
            and r.vt.status == "completed"
        ]
        engine_hits: dict[str, int] = {}
        engine_total: dict[str, int] = {}

        for r in malicious_rows:
            for eng in r.vt.engine_details:
                engine_total[eng.engine] = engine_total.get(eng.engine, 0) + 1
                if eng.category == "malicious":
                    engine_hits[eng.engine] = engine_hits.get(eng.engine, 0) + 1

        if engine_hits:
            sorted_engines = sorted(
                engine_hits.items(), key=lambda x: x[1], reverse=True
            )
            print(f"  {'ENGINE':<30}  {'HITS':>5}  {'TOTAL':>6}  {'RATE':>6}")
            print("  " + "-" * 52)
            for eng, hits in sorted_engines[:25]:
                total_e = engine_total.get(eng, 1)
                rate = 100 * hits / total_e
                print(
                    f"  {eng:<30}  {hits:>5}  {total_e:>6}  {rate:>5.0f}%"
                )
        else:
            print(_yellow("  No engine positives recorded."))

    # ── SG latency ────────────────────────────────────────────────────────
    lats = [r.sg.latency_ms for r in rows
            if r.sg and not r.sg.error]
    if lats:
        print(f"\n{'='*72}")
        print(_bold("  SCRIPTGUARD LATENCY  (ms per request)"))
        print(f"{'='*72}")
        lats_s = sorted(lats)
        p95 = lats_s[int(len(lats_s) * 0.95)]
        p99 = lats_s[int(len(lats_s) * 0.99)]
        print(
            f"  min={min(lats):.0f}  "
            f"median={statistics.median(lats):.0f}  "
            f"mean={statistics.mean(lats):.0f}  "
            f"p95={p95:.0f}  "
            f"p99={p99:.0f}  "
            f"max={max(lats):.0f}"
        )

    print(f"\n{'='*72}\n")


def _print_confusion(m: dict, name: str) -> None:
    tp, tn, fp, fn = m["tp"], m["tn"], m["fp"], m["fn"]
    print(f"\n  {_bold(name)}")
    print(f"                         Pred MALICIOUS   Pred BENIGN")
    print(f"  Actual MALICIOUS           TP={tp:<6}       FN={fn}")
    print(f"  Actual BENIGN              FP={fp:<6}       TN={tn}")


def _print_metrics_comparison(sg: dict, vt: dict, vt_threshold: int) -> None:
    W = 12

    def _diff(sg_v: float, vt_v: float) -> str:
        d = sg_v - vt_v
        if abs(d) < 0.0005:
            return "  ─"
        return f"{_green(f'+{d:.4f}') if d > 0 else _red(f'{d:.4f}')}"

    header = (f"  {'METRIC':<18}  {'ScriptGuard':>{W}}  "
              f"{'VirusTotal':>{W}}  {'SG - VT':>10}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    metrics_order = [
        ("Accuracy",    "accuracy"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
        ("F1",          "f1"),
        ("Specificity", "specificity"),
        ("FPR",         "fpr"),
        ("FNR",         "fnr"),
        ("MCC",         "mcc"),
    ]
    for label, key in metrics_order:
        sg_v = sg[key]
        vt_v = vt[key]
        diff_str = _diff(sg_v, vt_v)
        print(
            f"  {label:<18}  {sg_v:>{W}.4f}  {vt_v:>{W}.4f}  {diff_str:>10}"
        )
    print(
        f"\n  {'Samples':<18}  {sg['total']:>{W}}  {vt['total']:>{W}}"
    )


def _print_metrics_single(m: dict) -> None:
    print(f"  {'METRIC':<18}  {'ScriptGuard':>12}")
    print("  " + "-" * 34)
    for label, key in [
        ("Accuracy",    "accuracy"),
        ("Precision",   "precision"),
        ("Recall",      "recall"),
        ("F1",          "f1"),
        ("Specificity", "specificity"),
        ("FPR",         "fpr"),
        ("FNR",         "fnr"),
        ("MCC",         "mcc"),
    ]:
        print(f"  {label:<18}  {m[key]:>12.4f}")
    print(f"\n  {'Samples':<18}  {m['total']:>12}")


def _pass_rate(rows: list[ComparisonRow], system: str) -> str:
    results_list = []
    for r in rows:
        res = r.sg if system == "sg" else r.vt
        if res and not res.error and res.label:
            ok = res.label == r.expected
            results_list.append(ok)
    if not results_list:
        return "N/A"
    n = len(results_list)
    ok = sum(results_list)
    pct = 100 * ok // n
    return f"{ok}/{n} ({pct}%)"


# ============================================================================
# HTML report generator
# ============================================================================

def generate_html_report(
    rows: list[ComparisonRow],
    vt_threshold: int,
    output_path: str,
) -> None:
    """Generate a self-contained HTML report with a comparison table."""

    has_vt = any(r.vt is not None for r in rows)
    sg_m = metrics_from_rows(rows, "sg")
    vt_m = metrics_from_rows(rows, "vt") if has_vt else None

    def _cell_class(label: str, expected: str) -> str:
        if label == expected:
            return "pass"
        elif label in ("ERROR", "timeout", "error", "N/A", ""):
            return "error"
        return "fail"

    def _esc(s) -> str:
        return html.escape(str(s) if s is not None else "")

    rows_html = ""
    prev_group = ""
    for r in rows:
        group = ("obfuscated" if r.description.startswith("obf:")
                 else r.expected)
        if group != prev_group:
            label_map = {
                "benign":     "BENIGN",
                "malicious":  "MALICIOUS (plain)",
                "obfuscated": "MALICIOUS (obfuscated)",
            }
            rows_html += (
                f'<tr class="group-header">'
                f'<td colspan="9">{label_map.get(group, group)}</td>'
                f'</tr>\n'
            )
            prev_group = group

        sg = r.sg
        vt = r.vt

        sg_label = sg.label if sg and not sg.error else ("ERROR" if sg else "N/A")
        sg_prob  = f"{sg.malicious_prob:.3f}" if sg and not sg.error else ""
        sg_conf  = f"{sg.confidence:.3f}"     if sg and not sg.error else ""
        sg_ms    = f"{sg.latency_ms:.0f}"     if sg else ""
        sg_cls   = _cell_class(sg_label, r.expected)

        vt_label = (
            vt.label if vt and vt.status == "completed" and not vt.error
            else (vt.status if vt else "N/A")
        )
        vt_pos   = (f"{vt.positive_engines}/{vt.total_engines}"
                    if vt and vt.status == "completed" else "")
        vt_cls   = _cell_class(vt_label, r.expected)

        agree_cls = ""
        agree_txt = ""
        if sg and vt and sg.label and vt.label == "benign" or vt_label == "malicious":
            agree = sg_label == vt_label
            agree_cls = "agree" if agree else "disagree"
            agree_txt = "✓" if agree else "✗"

        rows_html += (
            f'<tr>'
            f'<td class="desc">{_esc(r.description)}</td>'
            f'<td class="expected">{_esc(r.expected)}</td>'
            f'<td class="{sg_cls}">{_esc(sg_label)}</td>'
            f'<td class="num">{_esc(sg_prob)}</td>'
            f'<td class="num">{_esc(sg_conf)}</td>'
            f'<td class="num">{_esc(sg_ms)}</td>'
            f'<td class="{vt_cls}">{_esc(vt_label)}</td>'
            f'<td class="num">{_esc(vt_pos)}</td>'
            f'<td class="{agree_cls}">{agree_txt}</td>'
            f'</tr>\n'
        )

    def _m_row(name: str, key: str) -> str:
        sg_v = f"{sg_m[key]:.4f}"
        vt_v = f"{vt_m[key]:.4f}" if vt_m else "—"
        diff = ""
        diff_cls = ""
        if vt_m:
            d = sg_m[key] - vt_m[key]
            if abs(d) >= 0.0005:
                diff = f"{'+' if d > 0 else ''}{d:.4f}"
                diff_cls = "diff-pos" if d > 0 else "diff-neg"
        return (
            f'<tr>'
            f'<td>{_esc(name)}</td>'
            f'<td class="num">{_esc(sg_v)}</td>'
            f'<td class="num">{_esc(vt_v)}</td>'
            f'<td class="num {diff_cls}">{_esc(diff)}</td>'
            f'</tr>\n'
        )

    metrics_html = "".join([
        _m_row("Accuracy",    "accuracy"),
        _m_row("Precision",   "precision"),
        _m_row("Recall",      "recall"),
        _m_row("F1",          "f1"),
        _m_row("Specificity", "specificity"),
        _m_row("FPR",         "fpr"),
        _m_row("FNR",         "fnr"),
        _m_row("MCC",         "mcc"),
    ])

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime())

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScriptGuard vs VirusTotal — Benchmark Report</title>
<style>
  :root {{
    --bg: #0f1117; --surface: #1a1d27; --border: #2a2d3e;
    --text: #e2e8f0; --muted: #94a3b8;
    --pass: #22c55e; --fail: #ef4444; --warn: #f59e0b;
    --blue: #60a5fa; --purple: #a78bfa;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font: 14px/1.6 'Segoe UI', system-ui, sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.6rem; color: var(--blue); margin-bottom: 4px; }}
  h2 {{ font-size: 1.1rem; color: var(--purple); margin: 28px 0 12px; }}
  .meta {{ color: var(--muted); font-size: 0.85rem; margin-bottom: 24px; }}
  .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px;
           margin-bottom: 24px; }}
  .card {{ background: var(--surface); border: 1px solid var(--border);
           border-radius: 8px; padding: 16px; }}
  .card-title {{ color: var(--muted); font-size: 0.8rem; text-transform: uppercase;
                 letter-spacing: 0.08em; margin-bottom: 8px; }}
  .big-num {{ font-size: 2rem; font-weight: 700; }}
  .big-num.good {{ color: var(--pass); }}
  .big-num.bad  {{ color: var(--fail); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  th {{ background: var(--surface); color: var(--muted); text-align: left;
        padding: 8px 10px; border-bottom: 1px solid var(--border);
        position: sticky; top: 0; z-index: 1; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
  tr:hover td {{ background: rgba(255,255,255,0.02); }}
  .group-header td {{ background: var(--surface); color: var(--purple);
                      font-weight: 600; font-size: 0.8rem;
                      letter-spacing: 0.05em; padding: 10px 10px; }}
  .pass   {{ color: var(--pass); }}
  .fail   {{ color: var(--fail); }}
  .error  {{ color: var(--warn); }}
  .agree  {{ color: var(--pass); text-align: center; }}
  .disagree {{ color: var(--fail); text-align: center; }}
  .diff-pos {{ color: var(--pass); }}
  .diff-neg {{ color: var(--fail); }}
  .num  {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .desc {{ max-width: 360px; word-break: break-word; }}
  .expected {{ color: var(--muted); }}
  .scroll {{ overflow-x: auto; }}
</style>
</head>
<body>
<h1>ScriptGuard vs VirusTotal — Benchmark Report</h1>
<div class="meta">Generated {_esc(timestamp)} · {len(rows)} samples total</div>

<h2>Metrics Summary</h2>
<div class="scroll">
<table>
<thead>
<tr>
  <th>Metric</th>
  <th class="num">ScriptGuard</th>
  <th class="num">VirusTotal</th>
  <th class="num">SG − VT</th>
</tr>
</thead>
<tbody>
{metrics_html}
</tbody>
</table>
</div>

<h2>Sample Results</h2>
<div class="scroll">
<table>
<thead>
<tr>
  <th>Description</th>
  <th>Expected</th>
  <th>SG Label</th>
  <th class="num">SG Prob</th>
  <th class="num">SG Conf</th>
  <th class="num">SG ms</th>
  <th>VT Label</th>
  <th class="num">VT Pos/Tot</th>
  <th>Agree</th>
</tr>
</thead>
<tbody>
{rows_html}
</tbody>
</table>
</div>
</body>
</html>"""

    Path(output_path).write_text(html_doc, encoding="utf-8")
    print(f"  HTML report saved to {output_path}")


# ============================================================================
# JSON export
# ============================================================================

def export_json(rows: list[ComparisonRow], path: str) -> None:
    out = []
    for r in rows:
        sg = r.sg
        vt = r.vt
        row_dict: dict = {
            "description": r.description,
            "expected":    r.expected,
        }
        if sg:
            row_dict.update({
                "sg_label":          sg.label,
                "sg_confidence":     sg.confidence,
                "sg_malicious_prob": sg.malicious_prob,
                "sg_latency_ms":     sg.latency_ms,
                "sg_error":          sg.error,
            })
        if vt:
            row_dict.update({
                "vt_label":            vt.label,
                "vt_status":           vt.status,
                "vt_positive_engines": vt.positive_engines,
                "vt_total_engines":    vt.total_engines,
                "vt_file_sha256":      vt.file_sha256,
                "vt_latency_ms":       vt.latency_ms,
                "vt_error":            vt.error,
                "vt_engine_details":   [
                    {
                        "engine":   e.engine,
                        "category": e.category,
                        "result":   e.result,
                    }
                    for e in (vt.engine_details or [])
                ],
            })
        out.append(row_dict)

    Path(path).write_text(json.dumps(out, indent=2, default=str),
                          encoding="utf-8")
    print(f"  JSON results saved to {path}")


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    p = argparse.ArgumentParser(
        description="ScriptGuard vs VirusTotal comparative benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--url", default="http://localhost:8000",
        help="ScriptGuard API base URL",
    )
    p.add_argument(
        "--threshold", type=float, default=None,
        help="ScriptGuard decision threshold override (0–1)",
    )
    p.add_argument(
        "--vt-key", metavar="KEY", default=None,
        help="VirusTotal API key (free or premium). Omit to skip VT.",
    )
    p.add_argument(
        "--vt-threshold", type=int, default=1,
        help="Number of VT engine positives required to call sample malicious",
    )
    p.add_argument(
        "--vt-ratelimit", type=float, default=16.0,
        help=(
            "Seconds to wait between VT requests (free tier: 4 req/min → 15s). "
            "Reduce if you have a premium key."
        ),
    )
    p.add_argument(
        "--vt-poll", type=float, default=20.0,
        help="Seconds between polling VT for analysis completion",
    )
    p.add_argument(
        "--vt-max-wait", type=float, default=300.0,
        help="Max seconds to wait for a single VT analysis to complete",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Parallel workers for ScriptGuard requests",
    )
    p.add_argument(
        "--json-out", metavar="FILE", default=None,
        help="Save full results to JSON file",
    )
    p.add_argument(
        "--html-out", metavar="FILE", default=None,
        help="Save HTML comparison report to file",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print extra info on misclassifications",
    )
    args = p.parse_args()

    # ETA warning for VT
    if args.vt_key:
        n = len(ALL_SAMPLES)
        eta_min = n * args.vt_ratelimit / 60
        eta_max = n * (args.vt_ratelimit + args.vt_poll) / 60
        print(
            f"\n[INFO] VirusTotal enabled. "
            f"Estimated time: {eta_min:.0f}–{eta_max:.0f} minutes "
            f"for {n} samples (free-tier rate limit)."
        )
        print(
            "[INFO] New submissions take longer (waiting for analysis). "
            "Already-scanned hashes are returned instantly.\n"
        )

    rows = run_comparison(
        base_url=args.url,
        sg_threshold=args.threshold,
        vt_api_key=args.vt_key,
        vt_threshold=args.vt_threshold,
        vt_rate_limit=args.vt_ratelimit,
        vt_poll=args.vt_poll,
        vt_max_wait=args.vt_max_wait,
        workers=args.workers,
        verbose=args.verbose,
    )

    has_vt = args.vt_key is not None and any(r.vt is not None for r in rows)
    print_comparison_report(rows, has_vt, args.vt_threshold)

    if args.json_out:
        export_json(rows, args.json_out)

    if args.html_out:
        generate_html_report(rows, args.vt_threshold, args.html_out)


if __name__ == "__main__":
    main()