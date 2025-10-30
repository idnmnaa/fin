from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from openai import OpenAI, BadRequestError
import os, json, traceback, re, threading, asyncio
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta

app = FastAPI()

CODE_VERSION = "v1.13."
print(f"🔁 Starting GPT signal evaluation server — code version: {CODE_VERSION}")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY is not set. The /evaluate route will fail until it is provided.")
client = OpenAI(api_key=OPENAI_API_KEY)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini").strip()
STRICT_FAIL_ON_UNPARSABLE = os.getenv("STRICT_FAIL_ON_UNPARSABLE", "0").strip() == "1"

# ===================== SYSTEM PROMPT (single source) =====================
# You can override via env var SYSTEM_PROMPT. The default below mirrors the original inline prompt.
DEFAULT_SYSTEM_PROMPT = (
    """
You are a trade evaluation assistant using a 3-rule decision model.
This trade setup already passed all filters 
(RSI, ATR, SMA50 cross, structural filters, obv.). 
Do NOT re-evaluate them.
For each trade input, determine whether the trade should be TAKE=1 or SKIP=0.
Rules:
1.	SMA50 Slope must support the trade direction.
•	BUY: SMA50 should be clearly sloping upward.
•	SELL: SMA50 should be clearly sloping downward.
2.	Price Structure must support the trade direction.
•	BUY: Look for higher highs and higher lows.
•	SELL: Look for lower highs and lower lows.
3.	HTF Bias must support the trade direction.
•	BUY: HTF trend or momentum should be bullish.
•	SELL: HTF trend or momentum should be bearish.
Decision Logic:
•	If at least 2 out of the 3 rules align with the trade direction, return TAKE=1.
•	Otherwise, return SKIP=0.
Return a single value 1 or 0 based on these factors. No text, only the value.
"""
)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT).strip()

# Show the effective SYSTEM_PROMPT once at startup
if os.getenv("SHOW_PROMPT_ON_START", "1").strip() == "1":
    print("\n===== SYSTEM_PROMPT (effective) =====\n" + SYSTEM_PROMPT + "\n===== END SYSTEM_PROMPT =====\n")

# ===================== CACHE (GPTarr) =====================
# Structure: {sym, tf, time (iso), time_dt (UTC), answer, key, ts_added}
GPTarr: List[Dict[str, Any]] = []
_GPTARR_LOCK = threading.Lock()
_MAX_AGE = timedelta(hours=1)

def _now_utc() -> datetime:
    return datetime.now(timezone.utc)

# ----------------- Generic helpers -----------------
def is_nano_or_mini(model_name: str) -> bool:
    m = model_name.lower()
    return any(k in m for k in ["nano", "mini", "small"])

def parse_json_strict_but_safe(body_bytes: bytes) -> dict:
    tail = body_bytes[-16:] if len(body_bytes) >= 16 else body_bytes
    print(f"📦 Incoming bytes: len={len(body_bytes)} tail={repr(tail)}")
    cleaned = body_bytes.replace(b"\x00", b"")
    start = cleaned.find(b"{")
    end = cleaned.rfind(b"}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No valid JSON object delimiters found in body")
    s = cleaned[start:end + 1].decode("utf-8", errors="ignore").strip()
    return json.loads(s)

def extract_probability(text: str) -> Optional[float]:
    if not text:
        return None
    text = text.strip()
    try:
        val = float(text)
        if 0.0 <= val <= 1.0:
            return val
    except Exception:
        pass
    m = re.search(r"(?<!\d)(?:0(?:\.\d+)?|1(?:\.0+)?)", text)
    if m:
        try:
            val = float(m.group(0))
            if 0.0 <= val <= 1.0:
                return val
        except Exception:
            return None
    return None

_TF_RE = re.compile(r"^\s*([mMhH])\s*([0-9]+)\s*$")

def timeframe_to_seconds(tf: Optional[str]) -> int:
    if not tf or not isinstance(tf, str):
        return 60
    m = _TF_RE.match(tf)
    if not m:
        m2 = re.search(r"(M|H)(\d+)", tf, re.IGNORECASE)
        if not m2:
            return 60
        unit, num = m2.group(1).upper(), int(m2.group(2))
    else:
        unit, num = m.group(1).upper(), int(m.group(2))
    return (num * 60) if unit == "M" else (num * 3600)

def parse_first_bar_time(payload: Dict[str, Any]) -> Optional[datetime]:
    try:
        bars = payload.get("bars") or []
        if not bars:
            return None
        t = bars[0].get("t")
        if not t or not isinstance(t, str):
            return None
        if t.endswith("Z"):
            t = t[:-1] + "+00:00"
        dt = datetime.fromisoformat(t)
        dt = dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc)
        return dt
    except Exception as e:
        print("⚠️ parse_first_bar_time error:", e)
        return None

def floor_to_bar(dt: datetime, bar_sec: int) -> datetime:
    dt = dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    epoch = int(dt.timestamp())
    floored = epoch - (epoch % max(1, bar_sec))
    return datetime.fromtimestamp(floored, tz=timezone.utc)

def canonical_key(sym: str, tf: str, first_bar_dt: datetime, bar_sec: int) -> str:
    base_dt = floor_to_bar(first_bar_dt, bar_sec)
    return f"{sym}|{tf}|{base_dt.isoformat()}"

def compute_tolerance_seconds(bar_sec: int) -> int:
    return max(30, min(180, bar_sec // 2 if bar_sec > 0 else 60))

def _extract_sym_tf(payload: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    meta = payload.get("meta") or {}
    sym = payload.get("sym") or payload.get("symbol") or meta.get("sym") or meta.get("symbol")
    tf  = payload.get("tf")  or payload.get("timeframe") or payload.get("TF") \
          or meta.get("tf")  or meta.get("timeframe")    or meta.get("TF")
    if isinstance(sym, str): sym = sym.strip().upper()
    if isinstance(tf, str):  tf  = tf.strip().upper()
    return sym, tf

# ----------------- Cache lookup/store/clean -----------------
def find_cached_answer(payload: Dict[str, Any]) -> Optional[float]:
    sym, tf = _extract_sym_tf(payload)
    if not sym or not tf:
        return None
    first_bar_dt = parse_first_bar_time(payload)
    if not first_bar_dt:
        return None

    bar_sec = timeframe_to_seconds(tf)
    tol = compute_tolerance_seconds(bar_sec)
    key_exact = canonical_key(sym, tf, first_bar_dt, bar_sec)

    with _GPTARR_LOCK:
        # 1) Exact key match
        for row in reversed(GPTarr):
            if row.get("key") == key_exact and isinstance(row.get("answer"), (int, float)):
                prob = float(row["answer"])
                print(f"💾 Cache HIT (exact) — key={key_exact} prob={prob:.4f}")
                return prob

        # 2) Proximity by sym/tf and |dt diff| ≤ tol
        fb_epoch = int(first_bar_dt.timestamp())
        for row in reversed(GPTarr):
            if row.get("sym") != sym or row.get("tf") != tf:
                continue
            row_dt = row.get("time_dt")
            if not isinstance(row_dt, datetime):
                continue
            if abs(int(row_dt.timestamp()) - fb_epoch) <= tol:
                ans = row.get("answer")
                if isinstance(ans, (int, float)):
                    prob = float(ans)
                    print(f"💾 Cache HIT (prox ±{tol}s) — sym={sym} tf={tf} "
                          f"req={first_bar_dt.isoformat()} cached={row_dt.isoformat()} prob={prob:.4f}")
                    return prob
    return None

def add_cache_record(payload: Dict[str, Any], answer: float) -> str:
    sym, tf = _extract_sym_tf(payload)
    first_bar_dt = parse_first_bar_time(payload)
    if not sym or not tf or not first_bar_dt:
        return "N/A"
    bar_sec = timeframe_to_seconds(tf)
    key = canonical_key(sym, tf, first_bar_dt, bar_sec)
    row = {
        "sym": sym,
        "tf": tf,
        "time": first_bar_dt.isoformat(),
        "time_dt": first_bar_dt,
        "answer": float(answer),
        "key": key,
        "ts_added": _now_utc().isoformat(),
    }
    with _GPTARR_LOCK:
        GPTarr.append(row)
    print(f"🆕 Unique result stored — key={key} prob={answer:.4f}")
    return key

def clean_cache() -> None:
    cutoff = _now_utc() - _MAX_AGE
    with _GPTARR_LOCK:
        before = len(GPTarr)
        GPTarr[:] = [r for r in GPTarr if isinstance(r.get("time_dt"), datetime) and r["time_dt"] >= cutoff]
        after = len(GPTarr)
    if before != after:
        print(f"🧹 Cache cleaned: {before} -> {after} (drop if first_bar_dt < {cutoff.isoformat()})")

# ===================== IN-FLIGHT DEDUPE =====================
# Followers wait briefly for the leader’s result to avoid duplicate GPT calls.
_INFLIGHT: Dict[str, asyncio.Future] = {}
_INFLIGHT_LOCK = asyncio.Lock()

def inflight_bucket_key(sym: str, tf: str, first_bar_dt: datetime, bar_sec: int) -> str:
    """
    Proximity bucket (coarser than cache key) so tiny time differences coalesce:
    bucket by round(epoch / tol), where tol = half a bar (clamped 30..180).
    """
    tol = compute_tolerance_seconds(bar_sec)
    buck = int(round(first_bar_dt.timestamp() / tol))
    return f"{sym}|{tf}|prox|{tol}|{buck}"

async def inflight_acquire(key: str) -> Tuple[bool, asyncio.Future]:
    async with _INFLIGHT_LOCK:
        fut = _INFLIGHT.get(key)
        if fut and not fut.done():
            return False, fut  # follower
        fut = asyncio.get_event_loop().create_future()
        _INFLIGHT[key] = fut
        return True, fut       # leader

async def inflight_finish(key: str, result: float = None, err: Exception = None):
    async with _INFLIGHT_LOCK:
        fut = _INFLIGHT.pop(key, None)
    if fut and not fut.done():
        if err is not None:
            fut.set_exception(err)
        else:
            fut.set_result(result)

# ===================== MODEL CALLERS =====================
def build_args(system_prompt: str, compact_json: str, *, max_tok_primary=3, max_tok_retry=6, retry=False):
    args = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": compact_json},
        ],
        "n": 1,
    }
    if is_nano_or_mini(MODEL_NAME):
        args["max_completion_tokens"] = max_tok_retry if retry else max_tok_primary
    else:
        args["max_tokens"] = max_tok_retry if retry else max_tok_primary
        args["temperature"] = 0
        args["stop"] = ["\n"]
    return args

def auto_heal_and_call(args):
    try:
        return client.chat.completions.create(**args)
    except BadRequestError as e:
        msg = str(e)
        print("⚠️ BadRequestError, attempting auto-fix:", msg)
        for p in re.findall(r"Unsupported parameter: '([^']+)'", msg):
            args.pop(p, None)
        if "max_tokens" in msg and "Unsupported" in msg:
            args.pop("max_tokens", None)
            args["max_completion_tokens"] = args.get("max_completion_tokens", 3)
        if "max_completion_tokens" in msg and "Unsupported" in msg:
            args.pop("max_completion_tokens", None)
            args["max_tokens"] = args.get("max_tokens", 3)
        if "temperature" in msg and "Unsupported" in msg:
            args.pop("temperature", None)
        if "stop" in msg and "Unsupported" in msg:
            args.pop("stop", None)
        return client.chat.completions.create(**args)

# ===================== ROUTES =====================
@app.get("/health")
async def health():
    return {"status": "ok", "version": CODE_VERSION, "model": MODEL_NAME, "cache_size": len(GPTarr)}

@app.post("/evaluate")
async def evaluate(request: Request):
    print(f"\n📥 New request — version: {CODE_VERSION}")
    try:
        body_bytes = await request.body()
        preview = body_bytes[:400]
        print("Raw request (first 400 bytes):", preview.decode("utf-8", errors="ignore"))

        try:
            payload = parse_json_strict_but_safe(body_bytes)
        except Exception as pe:
            print("❌ JSON parse error:", str(pe))
            traceback.print_exc()
            return JSONResponse(status_code=400, content={"error": "Invalid JSON", "details": str(pe), "version": CODE_VERSION})

        # Extract + log key parts
        sym, tf = _extract_sym_tf(payload)
        first_bar_dt = parse_first_bar_time(payload)
        if not sym or not tf or not first_bar_dt:
            return JSONResponse(status_code=400, content={"error": "Missing sym/tf/bars[0].t", "version": CODE_VERSION})
        bar_sec = timeframe_to_seconds(tf)
        tol = compute_tolerance_seconds(bar_sec)
        key = canonical_key(sym, tf, first_bar_dt, bar_sec)
        now = _now_utc()
        print(f"🔑 Key={key} | tol_sec={tol} | first_bar_dt={first_bar_dt.isoformat()} | now_utc={now.isoformat()} | Δ={(now-first_bar_dt).total_seconds():.0f}s")

        # Housekeeping
        clean_cache()

        # Try cache
        cached = find_cached_answer(payload)
        if isinstance(cached, (int, float)):
            print(f"🔁 Already have result for response — key={key} prob={cached:.4f}")
            return {"probability": float(cached), "version": CODE_VERSION, "model": MODEL_NAME, "cache": "hit"}

        # In-flight dedupe (proximity bucket)
        prox_key = inflight_bucket_key(sym, tf, first_bar_dt, bar_sec)
        leader, fut = await inflight_acquire(prox_key)
        if not leader:
            print(f"⏳ In-flight pending — waiting for leader (prox_key={prox_key})")
            try:
                # Wait up to 5s; afterwards, re-check cache and possibly compute
                prob = await asyncio.wait_for(fut, timeout=5.0)
                print(f"🔁 Already have result for response (in-flight reuse) — key={key} prob={prob:.4f}")
                return {"probability": float(prob), "version": CODE_VERSION, "model": MODEL_NAME, "cache": "hit_inflight"}
            except asyncio.TimeoutError:
                print(f"⏱️ In-flight wait timed out (prox_key={prox_key}); rechecking cache...")
                cached2 = find_cached_answer(payload)
                if isinstance(cached2, (int, float)):
                    print(f"🔁 Already have result for response (post-timeout cache) — key={key} prob={cached2:.4f}")
                    return {"probability": float(cached2), "version": CODE_VERSION, "model": MODEL_NAME, "cache": "hit"}
                # proceed as ad-hoc leader

        # Leader path — compute with GPT (offload blocking call to a thread)
        compact_json = json.dumps(payload, separators=(",", ":"))

        system_prompt_primary = SYSTEM_PROMPT
        args = build_args(system_prompt_primary, compact_json, max_tok_primary=3, max_tok_retry=6, retry=False)
        try:
            resp = await asyncio.to_thread(auto_heal_and_call, args)
            reply = (resp.choices[0].message.content or "").strip()
            prob = extract_probability(reply)

            if prob is None:
                args_retry = build_args(SYSTEM_PROMPT, compact_json, max_tok_primary=3, max_tok_retry=6, retry=True)
                resp2 = await asyncio.to_thread(auto_heal_and_call, args_retry)
                reply2 = (resp2.choices[0].message.content or "").strip()
                prob = extract_probability(reply2)

            if prob is None:
                msg = "Model did not return a numeric probability"
                print(f"⚠️ {msg}. Using fallback.")
                if STRICT_FAIL_ON_UNPARSABLE:
                    await inflight_finish(prox_key, err=RuntimeError(msg))
                    return JSONResponse(status_code=502, content={"error": msg, "version": CODE_VERSION, "model": MODEL_NAME, "cache": "miss"})
                prob = 0.5

            prob = min(1.0, max(0.0, float(prob)))
            stored_key = add_cache_record(payload, prob)
            await inflight_finish(prox_key, result=prob)

            print(f"✅ Final probability (NEW) — key={stored_key} prob={prob:.4f}")
            return {"probability": prob, "version": CODE_VERSION, "model": MODEL_NAME, "cache": "miss"}

        except Exception as e:
            await inflight_finish(prox_key, err=e)
            print("❌ Unhandled ERROR during GPT call:", str(e))
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

    except BadRequestError as e:
        print("❌ OpenAI BadRequestError:", str(e))
        return JSONResponse(status_code=400, content={"error": str(e), "version": CODE_VERSION})
    except Exception as e:
        print("❌ Unhandled ERROR:", str(e))
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "version": CODE_VERSION})

@app.get("/", response_class=PlainTextResponse)
async def root():
    return f"OK: {CODE_VERSION}\n"

