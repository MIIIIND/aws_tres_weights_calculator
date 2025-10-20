#!/usr/bin/env python3
import argparse
import re
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

# ---- pricing (per HOUR) ----
P_CPU = 0.036377      # per vCPU-hour
P_MEM_GB = 0.004561   # per GB-hour
P_GPU = 0.848508      # per GPU-hour (for a10 in this run)

MEM_RE = re.compile(r"^(?P<val>[\d\.]+)\s*(?P<unit>[KMGTP])?B?$", re.IGNORECASE)

def setup_logging(verbosity:int, logfile:Path|None):
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    fmt = "%(asctime)s | %(levelname)-8s | %(message)s"
    logging.basicConfig(level=level, format=fmt, handlers=[logging.StreamHandler()])
    if logfile:
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)

def to_gb(mem_str: str) -> float:
    if pd.isna(mem_str) or mem_str == '':
        return 0.0
    m = MEM_RE.match(str(mem_str))
    if not m:
        try:
            return float(str(mem_str).rstrip('Gg'))
        except Exception:
            logging.debug(f"MEM parse fallback -> 0 for value: {mem_str!r}")
            return 0.0
    val = float(m.group('val'))
    unit = (m.group('unit') or 'G').upper()
    factor = {'K': 1/(1024*1024), 'M': 1/1024, 'G': 1, 'T': 1024, 'P': 1024*1024}.get(unit, 1)
    return val * factor

def parse_alloctres(s: str):
    """
    Robust parser for strings like: 'cpu=9,mem=65G,gres/gpu=1,gres/gpu:a10=1'
    Returns dict: cpu, mem_gb, gpu_count, gpu_type
    """
    result = {'cpu': 0, 'mem_gb': 0.0, 'gpu_count': 0, 'gpu_type': None}
    if pd.isna(s) or s == '':
        return result

    parts = [p.strip() for p in str(s).split(',') if p.strip() and '=' in p]
    kv = {}
    for p in parts:
        k, v = p.split('=', 1)
        kv[k.strip()] = v.strip()

    # cpu
    if 'cpu' in kv:
        try:
            result['cpu'] = int(float(kv['cpu']))
        except Exception:
            logging.debug(f"CPU parse failed for {kv.get('cpu')!r}")

    # mem
    if 'mem' in kv:
        result['mem_gb'] = to_gb(kv['mem'])

    # GPUs (generic and typed)
    gpu_type_candidates = []
    for k, v in kv.items():
        if k.startswith('gres/gpu'):
            parts = k.split(':', 1)
            gtype = parts[1] if len(parts) > 1 else None
            try:
                count = int(float(v))
            except Exception:
                count = 0
            if count > 0:
                gpu_type_candidates.append((gtype, count))

    if gpu_type_candidates:
        # prefer a known type, else generic (None)
        gtype, gcount = gpu_type_candidates[0]
        result['gpu_type'] = gtype
        result['gpu_count'] = gcount
    # else keep defaults (0, None)
    return result

def summarize(df: pd.DataFrame, stage: str):
    logging.info(f"[{stage}] rows={len(df)}")
    if len(df) == 0:
        return
    # Peek at AllocTRES
    at = df.get("AllocTRES")
    if at is not None:
        logging.debug(f"[{stage}] sample AllocTRES:\n" + at.astype(str).head(5).to_string(index=False))
    # GPU stats
    for col in ("gpu_type", "gpu_count"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False).head(10)
            logging.debug(f"[{stage}] {col} value_counts:\n{vc}")

def main():
    ap = argparse.ArgumentParser(description="Compute job prices from sacct CSV with logging.")
    ap.add_argument("input_csv", help="Path to sacct CSV (must contain ElapsedRaw, AllocTRES)")
    ap.add_argument("--out", default=None, help="Output CSV path (default: <input>_priced.csv)")
    ap.add_argument("--gpu-type", default="a10", help="Target GPU type (default: a10)")
    ap.add_argument("--include-no-gpu", action="store_true", help="Also keep jobs with no GPU.")
    ap.add_argument("--no-filter", action="store_true", help="Do not filter by GPU type at all.")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="Increase log verbosity (-v, -vv).")
    ap.add_argument("--logfile", default=None, help="Optional log file path.")
    ap.add_argument("--diag-csv", default=None, help="Write a diagnostics CSV explaining filter masks.")
    args = ap.parse_args()

    inp = Path(args.input_csv)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) if args.out else inp.with_name(inp.stem + "_priced.csv")
    logfile = Path(args.logfile) if args.logfile else inp.with_name(inp.stem + f"_{timestamp}.log")
    setup_logging(args.verbose, logfile)

    logging.info(f"Reading: {inp}")
    df = pd.read_csv(inp)
    summarize(df, "read_csv")

    # Check required columns
    required_cols = {"ElapsedRaw", "AllocTRES"}
    missing = required_cols - set(df.columns)
    if missing:
        logging.error(f"Missing required columns: {missing}")
        raise SystemExit(2)

    # Parse AllocTRES -> new cols
    logging.info("Parsing AllocTRES...")
    parsed = df["AllocTRES"].apply(parse_alloctres).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    summarize(df, "parsed_alloc")

    # Normalize ElapsedRaw minutes -> hours (float)
    df["elapsed_minutes"] = pd.to_numeric(df["ElapsedRaw"], errors="coerce")
    if df["elapsed_minutes"].isna().any():
        n = df["elapsed_minutes"].isna().sum()
        logging.warning(f"{n} rows had non-numeric ElapsedRaw; treating as 0 minutes.")
    df["elapsed_minutes"] = df["elapsed_minutes"].fillna(0.0)
    df["elapsed_hours"] = df["elapsed_minutes"] / 60.0

    # Prepare masks for diagnostics
    target = str(args.gpu_type).lower()
    df["gpu_type"] = df["gpu_type"].astype("string")
    df["gpu_type_norm"] = df["gpu_type"].str.lower()
    m_has_gpu = df["gpu_count"] > 0
    m_type_match = df["gpu_type_norm"] == target
    m_no_gpu = df["gpu_count"] == 0

    # Filter logic
    if args.no_filter:
        df_filt = df.copy()
        reason = "no_filter"
        logging.info("Bypassing GPU-type filtering (--no-filter).")
    else:
        if args.include_no_gpu:
            df_filt = df[(m_type_match & m_has_gpu) | m_no_gpu]
            reason = "type_or_no_gpu"
            logging.info(f"Filtering: keep (gpu_type=={target} & gpu_count>0) OR (gpu_count==0).")
        else:
            df_filt = df[m_type_match & m_has_gpu]
            reason = "type_only"
            logging.info(f"Filtering: keep (gpu_type=={target} & gpu_count>0).")

    # Diagnostics
    if args.diag_csv:
        diag = df.copy()
        diag["mask_has_gpu"] = m_has_gpu
        diag["mask_type_match"] = m_type_match
        diag["mask_kept"] = diag.index.isin(df_filt.index)
        Path(args.diag_csv).parent.mkdir(parents=True, exist_ok=True)
        diag.to_csv(args.diag_csv, index=False)
        logging.info(f"Wrote diagnostics CSV: {args.diag_csv}")

    summarize(df_filt, f"after_filter ({reason})")
    if len(df_filt) == 0:
        logging.warning("No rows after filtering. Common causes:")
        logging.warning(f"- No rows have gpu_count>0, or none have gpu_type=='{target}'.")
        logging.warning("- GPU type labels may differ (e.g., 'nvidia-a10' vs 'a10'). Check value_counts above.")
        logging.warning("- AllocTRES parsing issue. Review sample AllocTRES in DEBUG logs.")
        logging.warning("- You filtered too strictly; try --no-filter or --include-no-gpu.")
        # We still continue to write an empty priced CSV for traceability.

    # Price per job
    df_filt["unit_rate_sum"] = (
        (df_filt["cpu"].fillna(0) * P_CPU) +
        (df_filt["mem_gb"].fillna(0) * P_MEM_GB) +
        (df_filt["gpu_count"].fillna(0) * P_GPU)
    )
    df_filt["price_per_job"] = (df_filt["elapsed_hours"].fillna(0) * df_filt["unit_rate_sum"]).round(6)

    # Save
    df_filt.drop(columns=["gpu_type_norm"], errors="ignore", inplace=True)
    df_filt.to_csv(out, index=False)
    logging.info(f"Wrote priced CSV: {out}")
    logging.info(f"Log file: {logfile}")

if __name__ == "__main__":
    main()
