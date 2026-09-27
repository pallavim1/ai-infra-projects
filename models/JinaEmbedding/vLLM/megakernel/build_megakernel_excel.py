#!/usr/bin/env python3
"""Builds the Excel (.xlsx), CSV (.csv), and consolidated JSON benchmark assets
strictly from the single-shot end-to-end empirical run on cb460828:
  - scratch/opus_v6e_tests/full_e2e_results.json (08:52:54Z -> 09:49:12Z)
  - scratch/opus_v6e_tests/mid_1k_490_results.json (09:50:12Z -> 09:51:17Z)
  - scratch/opus_v6e_tests/mid_2k_330_results.json (09:51:33Z -> 09:52:39Z)
"""

import csv
import json
import os
import shutil
import zipfile
from xml.sax.saxutils import escape

BASE_DIR = "/usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff"
SCRATCH_DIR = os.path.join(BASE_DIR, "scratch/opus_v6e_tests")
REPO_MEGAKERNEL_DIR = "/usr/local/google/home/pallaviam/panw-tpu-inference/models/JinaEmbedding/vLLM/megakernel"

ZHEMIN_V5E_BATCH = {
    (1, 1): ("85.8/s", 11.5, 12.7),
    (1, 4): ("187.4/s", 21.2, 22.8),
    (1, 8): ("187.6/s", 42.5, 44.5),
    (1, 16): ("186.8/s", 85.4, 89.9),
    (2, 1): ("59.5/s", 16.5, 17.7),
    (2, 4): ("97.7/s", 40.7, 42.8),
    (2, 8): ("96.6/s", 82.6, 86.0),
    (2, 16): ("96.5/s", 165.9, 171.2),
}

ZHEMIN_L4_BATCH = {
    (1, 1): ("43.2/s", 20.7, 23.5),
    (1, 4): ("103.4/s", 38.1, 42.9),
    (1, 8): ("135.1/s", 58.7, 66.3),
    (1, 16): ("165.2/s", 96.5, 107.6),
    (2, 1): ("39.0/s", 24.3, 28.2),
    (2, 4): ("75.7/s", 52.1, 58.2),
    (2, 8): ("90.5/s", 87.4, 97.3),
    (2, 16): ("101.1/s", 156.9, 175.5),
}

ZHEMIN_V5E_1K_RPS = {
    100: ("100", "11.5 ms", "14.0 ms", "PASS"),
    120: ("120", "11.8 ms", "15.5 ms", "PASS"),
    140: ("140", "11.6 ms", "18.5 ms", "PASS"),
    160: ("160", "16.8 ms", "24.2 ms", "PASS"),
    180: ("180.1", "19.9 ms", "29.6 ms", "PASS"),
    190: ("187.8", "523.7 ms", "762.7 ms", "SATURATED"),
    200: ("189", "1709 ms", "2818 ms", "SATURATED"),
    220: ("187.9", "3738 ms", "6182 ms", "SATURATED"),
}

ZHEMIN_V5E_2K_RPS = {
    90: ("90", "17.2 ms", "26.5 ms", "PASS"),
    95: ("94.75", "218.6 ms", "346.5 ms", "SATURATED"),
    100: ("96.32", "1031 ms", "2185 ms", "SATURATED"),
    110: ("96.74", "3208 ms", "5170 ms", "SATURATED"),
}


def load_empirical_data():
    with open(os.path.join(SCRATCH_DIR, "full_e2e_results.json")) as f:
        e2e = json.load(f)
    with open(os.path.join(SCRATCH_DIR, "mid_1k_490_results.json")) as f:
        m490 = json.load(f)["rows"]
    with open(os.path.join(SCRATCH_DIR, "mid_2k_330_results.json")) as f:
        m330 = json.load(f)["rows"]

    all_rps_rows = e2e["saturation_rows"] + m490 + m330

    rps_1k = {}
    rps_2k = {}
    for r in all_rps_rows:
        if r["phase"] == "warmup":
            continue
        if r["tokens"] == 1024:
            assert r["server_tokens_per_request"] == 1024.0, f"1K token mismatch: {r}"
            rps_1k[r["target_rps"]] = r
        elif r["tokens"] == 2048:
            assert r["server_tokens_per_request"] == 2048.0, f"2K token mismatch: {r}"
            rps_2k[r["target_rps"]] = r

    k6_conc = {}
    for r in e2e["k6_concurrency"]:
        expected = 1024.0 if r["payload_kb"] == 1 else 2048.0
        assert r["server_tokens_per_request"] == expected, f"k6 conc token mismatch: {r}"
        k6_conc[(r["payload_kb"], r["concurrency"])] = r

    single_batch = {}
    for r in e2e["single_http_batch"]:
        expected = 1024.0 if r["payload_kb"] == 1 else 2048.0
        assert r["server_tokens_per_prompt"] == expected, f"single batch token mismatch: {r}"
        single_batch[(r["payload_kb"], r["concurrency"])] = r

    # Save merged full_e2e_results.json into both scratch and repo
    e2e["saturation_rows_merged_sorted"] = {
        "1k": [rps_1k[k] for k in sorted(rps_1k.keys())],
        "2k": [rps_2k[k] for k in sorted(rps_2k.keys())],
    }
    with open(os.path.join(SCRATCH_DIR, "full_e2e_results.json"), "w") as f:
        json.dump(e2e, f, indent=2)
    with open(os.path.join(REPO_MEGAKERNEL_DIR, "cb460828_full_e2e_results.json"), "w") as f:
        json.dump(e2e, f, indent=2)

    return (
        [rps_1k[k] for k in sorted(rps_1k.keys())],
        [rps_2k[k] for k in sorted(rps_2k.keys())],
        k6_conc,
        single_batch,
    )


def col_letter(idx):
    res = ""
    idx += 1
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        res = chr(65 + rem) + res
    return res


def build_sheet_xml(rows):
    lines = [
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">',
        "  <cols>",
        '    <col min="1" max="2" width="28" customWidth="1"/>',
        '    <col min="3" max="12" width="24" customWidth="1"/>',
        "  </cols>",
        "  <sheetData>",
    ]
    for r_idx, row in enumerate(rows, start=1):
        lines.append(f'    <row r="{r_idx}">')
        for c_idx, val in enumerate(row):
            if val is None or val == "":
                continue
            ref = f"{col_letter(c_idx)}{r_idx}"
            if isinstance(val, (int, float)):
                lines.append(f'      <c r="{ref}" s="0"><v>{val}</v></c>')
            else:
                s_val = escape(str(val))
                lines.append(f'      <c r="{ref}" t="inlineStr"><is><t>{s_val}</t></is></c>')
        lines.append("    </row>")
    lines.append("  </sheetData>")
    lines.append("</worksheet>")
    return "\n".join(lines)


def create_xlsx(filename, sheets):
    with zipfile.ZipFile(filename, "w", zipfile.ZIP_DEFLATED) as zf:
        ct = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">',
            '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>',
            '  <Default Extension="xml" ContentType="application/xml"/>',
            '  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>',
        ]
        for i in range(1, len(sheets) + 1):
            ct.append(
                f'  <Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            )
        ct.append("</Types>")
        zf.writestr("[Content_Types].xml", "\n".join(ct))

        rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"""
        zf.writestr("_rels/.rels", rels)

        wb = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">',
            "  <sheets>",
        ]
        for i, (name, _) in enumerate(sheets, start=1):
            wb.append(f'    <sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>')
        wb.append("  </sheets>")
        wb.append("</workbook>")
        zf.writestr("xl/workbook.xml", "\n".join(wb))

        wb_rels = [
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>',
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">',
        ]
        for i in range(1, len(sheets) + 1):
            wb_rels.append(
                f'  <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            )
        wb_rels.append("</Relationships>")
        zf.writestr("xl/_rels/workbook.xml.rels", "\n".join(wb_rels))

        for i, (_, rows) in enumerate(sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{i}.xml", build_sheet_xml(rows))


def build_tab1(rps_1k, rps_2k, k6_conc, single_batch):
    rows = []
    rows.append([
        "TPU v6e FP32 Opus Megakernel v2 + Device Pooling (jina-v2-opus-megakernel @ cb460828 | Single-Shot E2E Run 2026-09-27T08:52:54Z–09:52:39Z | FP32, max_model_len=2048, Exact 1,024 & 2,048 Tokens) — Matching Zhemin gid=1161755388 AS-IS"
    ])
    rows.append([])

    # Section 1A: Batch Request Testing (Single HTTP request contains N prompts)
    rows.append([
        "Batch Request Testing: A single HTTP request contains multiple prompts (Single HTTP Request with N=1,4,8,16 prompts; Exact 1,024 & 2,048 tokens)"
    ])
    rows.append([
        "Payload Size",
        "Concurrency",
        "Throughput (TPU v6e cb460828)",
        "p50 (TPU v6e cb460828)",
        "p99 (TPU v6e cb460828)",
        "Throughput (TPU v5e Zhemin)",
        "p50 (TPU v5e Zhemin)",
        "p99 (TPU v5e Zhemin)",
        "Throughput (L4 GPU Zhemin)",
        "p50 (L4 GPU Zhemin)",
        "p99 (L4 GPU Zhemin)",
    ])
    for kb in [1, 2]:
        for c in [1, 4, 8, 16]:
            r = single_batch[(kb, c)]
            v5_t, v5_50, v5_99 = ZHEMIN_V5E_BATCH[(kb, c)]
            l4_t, l4_50, l4_99 = ZHEMIN_L4_BATCH[(kb, c)]
            rows.append([
                r["payload_label"],
                c,
                f"{r['throughput_rps']:.1f}/s",
                f"{r['p50_ms']:.1f}ms",
                f"{r['p99_ms']:.1f}ms",
                v5_t,
                f"{v5_50:.1f}ms",
                f"{v5_99:.1f}ms",
                l4_t,
                f"{l4_50:.1f}ms",
                f"{l4_99:.1f}ms",
            ])

    rows.append([])
    # Section 1B: Concurrent HTTP Request Testing (k6 constant-vus = 1, 4, 8, 16)
    rows.append([
        "Concurrent Request Testing (k6 constant-vus, VUS = 1, 4, 8, 16 concurrent HTTP requests; Exact 1,024 & 2,048 tokens)"
    ])
    rows.append([
        "Payload Size",
        "Concurrency",
        "Throughput (TPU v6e cb460828)",
        "p50 (TPU v6e cb460828)",
        "p99 (TPU v6e cb460828)",
        "Throughput (TPU v5e Zhemin)",
        "p50 (TPU v5e Zhemin)",
        "p99 (TPU v5e Zhemin)",
        "Throughput (L4 GPU Zhemin)",
        "p50 (L4 GPU Zhemin)",
        "p99 (L4 GPU Zhemin)",
    ])
    for kb in [1, 2]:
        for c in [1, 4, 8, 16]:
            r = k6_conc[(kb, c)]
            v5_t, v5_50, v5_99 = ZHEMIN_V5E_BATCH[(kb, c)]
            l4_t, l4_50, l4_99 = ZHEMIN_L4_BATCH[(kb, c)]
            rows.append([
                r["payload_label"],
                c,
                f"{r['throughput_rps']:.1f}/s",
                f"{r['p50_ms']:.1f}ms",
                f"{r['p99_ms']:.1f}ms",
                v5_t,
                f"{v5_50:.1f}ms",
                f"{v5_99:.1f}ms",
                l4_t,
                f"{l4_50:.1f}ms",
                f"{l4_99:.1f}ms",
            ])

    rows.append([])
    # Section 2: 1KB Dedicated Saturation
    rows.append(["1KB Dedicated Saturation (Exact 1,024 Tokens per Request, FP32, p99 < 50 ms SLA)"])
    rows.append([
        "RPS",
        "Achieved (TPU v6e cb460828)",
        "P50 (TPU v6e cb460828)",
        "P99 (TPU v6e cb460828)",
        "SLA (TPU v6e cb460828)",
        "Achieved (TPU v5e Zhemin)",
        "P50 (TPU v5e Zhemin)",
        "P99 (TPU v5e Zhemin)",
        "SLA (TPU v5e Zhemin)",
    ])
    for r in rps_1k:
        t_rps = r["target_rps"]
        v5 = ZHEMIN_V5E_1K_RPS.get(t_rps, ("—", "—", "—", "SATURATED (>180 RPS)"))
        rows.append([
            t_rps,
            f"{r['achieved_rps']:.2f}",
            f"{r['p50']:.1f} ms",
            f"{r['p99']:.1f} ms",
            r["status"],
            v5[0],
            v5[1],
            v5[2],
            v5[3],
        ])

    rows.append([])
    # Section 3: 2KB Dedicated Saturation
    rows.append(["2KB Dedicated Saturation (Exact 2,048 Tokens per Request, FP32, p99 < 50 ms SLA)"])
    rows.append([
        "RPS",
        "Achieved (TPU v6e cb460828)",
        "P50 (TPU v6e cb460828)",
        "P99 (TPU v6e cb460828)",
        "SLA (TPU v6e cb460828)",
        "Achieved (TPU v5e Zhemin)",
        "P50 (TPU v5e Zhemin)",
        "P99 (TPU v5e Zhemin)",
        "SLA (TPU v5e Zhemin)",
    ])
    for r in rps_2k:
        t_rps = r["target_rps"]
        v5 = ZHEMIN_V5E_2K_RPS.get(t_rps, ("—", "—", "—", "SATURATED (>90 RPS)"))
        rows.append([
            t_rps,
            f"{r['achieved_rps']:.2f}",
            f"{r['p50']:.1f} ms",
            f"{r['p99']:.1f} ms",
            r["status"],
            v5[0],
            v5[1],
            v5[2],
            v5[3],
        ])

    return rows


def build_tab2(rps_1k, rps_2k, k6_conc, single_batch):
    rows = []
    rows.append([
        "TPU v6e Opus Megakernel cb460828 ($2.70/hr) vs TPU v5e ($1.20 / $1.22/hr) & NVIDIA L4 ($0.70/hr) — Matching Zhemin gid=1972899730 AS-IS"
    ])
    rows.append([])

    # 1A. Concurrent Request Comparison (Single-HTTP Multi-Prompt Batch N=1,4,8,16)
    rows.append(["Concurrent Request Comparison — Single HTTP Request with N Prompts (P50)"])
    rows.append([
        "Payload Size",
        "Concurrency",
        "TPU v6e cb460828 (ms)",
        "TPU v5e (ms)",
        "L4 GPU (ms)",
        "Absolute Delta (v6e vs v5e)",
        "% Reduction (v6e vs v5e)",
        "Absolute Delta (v6e vs L4)",
        "% Reduction (v6e vs L4)",
        "Faster Setup",
    ])
    for kb in [1, 2]:
        for c in [1, 4, 8, 16]:
            r = single_batch[(kb, c)]
            v6 = round(r["p50_ms"], 1)
            _, v5, _ = ZHEMIN_V5E_BATCH[(kb, c)]
            _, l4, _ = ZHEMIN_L4_BATCH[(kb, c)]
            d_v5 = v6 - v5
            pct_v5 = (1.0 - v6 / v5) * 100.0
            d_l4 = v6 - l4
            pct_l4 = (1.0 - v6 / l4) * 100.0
            rows.append([
                r["payload_label"],
                c,
                f"{v6:.1f}ms",
                f"{v5:.1f}ms",
                f"{l4:.1f}ms",
                f"{d_v5:+.1f}ms",
                f"{pct_v5:.1f}%",
                f"{d_l4:+.1f}ms",
                f"{pct_l4:.1f}%",
                "TPU v6e",
            ])

    rows.append([])
    rows.append(["Concurrent Request Comparison — Single HTTP Request with N Prompts (p99)"])
    rows.append([
        "Payload Size",
        "Concurrency",
        "TPU v6e cb460828 (ms)",
        "TPU v5e (ms)",
        "L4 GPU (ms)",
        "Absolute Delta (v6e vs v5e)",
        "% Reduction (v6e vs v5e)",
        "Absolute Delta (v6e vs L4)",
        "% Reduction (v6e vs L4)",
        "Faster Setup",
    ])
    for kb in [1, 2]:
        for c in [1, 4, 8, 16]:
            r = single_batch[(kb, c)]
            v6 = round(r["p99_ms"], 1)
            _, _, v5 = ZHEMIN_V5E_BATCH[(kb, c)]
            _, _, l4 = ZHEMIN_L4_BATCH[(kb, c)]
            d_v5 = v6 - v5
            pct_v5 = (1.0 - v6 / v5) * 100.0
            d_l4 = v6 - l4
            pct_l4 = (1.0 - v6 / l4) * 100.0
            rows.append([
                r["payload_label"],
                c,
                f"{v6:.1f}ms",
                f"{v5:.1f}ms",
                f"{l4:.1f}ms",
                f"{d_v5:+.1f}ms",
                f"{pct_v5:.1f}%",
                f"{d_l4:+.1f}ms",
                f"{pct_l4:.1f}%",
                "TPU v6e",
            ])

    rows.append([])
    # 1B. Concurrent Request Comparison (k6 Closed-Loop VUS=1,4,8,16)
    rows.append(["Concurrent Request Comparison — k6 Concurrent HTTP Requests VUS=1,4,8,16 (P50)"])
    rows.append([
        "Payload Size",
        "Concurrency",
        "TPU v6e cb460828 (ms)",
        "TPU v5e (ms)",
        "L4 GPU (ms)",
        "Absolute Delta (v6e vs v5e)",
        "% Reduction (v6e vs v5e)",
        "Absolute Delta (v6e vs L4)",
        "% Reduction (v6e vs L4)",
        "Faster Setup",
    ])
    for kb in [1, 2]:
        for c in [1, 4, 8, 16]:
            r = k6_conc[(kb, c)]
            v6 = round(r["p50_ms"], 1)
            _, v5, _ = ZHEMIN_V5E_BATCH[(kb, c)]
            _, l4, _ = ZHEMIN_L4_BATCH[(kb, c)]
            d_v5 = v6 - v5
            pct_v5 = (1.0 - v6 / v5) * 100.0
            d_l4 = v6 - l4
            pct_l4 = (1.0 - v6 / l4) * 100.0
            rows.append([
                r["payload_label"],
                c,
                f"{v6:.1f}ms",
                f"{v5:.1f}ms",
                f"{l4:.1f}ms",
                f"{d_v5:+.1f}ms",
                f"{pct_v5:.1f}%",
                f"{d_l4:+.1f}ms",
                f"{pct_l4:.1f}%",
                "TPU v6e",
            ])

    rows.append([])
    rows.append(["Concurrent Request Comparison — k6 Concurrent HTTP Requests VUS=1,4,8,16 (p99)"])
    rows.append([
        "Payload Size",
        "Concurrency",
        "TPU v6e cb460828 (ms)",
        "TPU v5e (ms)",
        "L4 GPU (ms)",
        "Absolute Delta (v6e vs v5e)",
        "% Reduction (v6e vs v5e)",
        "Absolute Delta (v6e vs L4)",
        "% Reduction (v6e vs L4)",
        "Faster Setup",
    ])
    for kb in [1, 2]:
        for c in [1, 4, 8, 16]:
            r = k6_conc[(kb, c)]
            v6 = round(r["p99_ms"], 1)
            _, _, v5 = ZHEMIN_V5E_BATCH[(kb, c)]
            _, _, l4 = ZHEMIN_L4_BATCH[(kb, c)]
            d_v5 = v6 - v5
            pct_v5 = (1.0 - v6 / v5) * 100.0
            d_l4 = v6 - l4
            pct_l4 = (1.0 - v6 / l4) * 100.0
            rows.append([
                r["payload_label"],
                c,
                f"{v6:.1f}ms",
                f"{v5:.1f}ms",
                f"{l4:.1f}ms",
                f"{d_v5:+.1f}ms",
                f"{pct_v5:.1f}%",
                f"{d_l4:+.1f}ms",
                f"{pct_l4:.1f}%",
                "TPU v6e",
            ])

    rows.append([])
    # 2. RPS Saturation Result
    r1k_480 = [x for x in rps_1k if x["target_rps"] == 480][0]
    r1k_490 = [x for x in rps_1k if x["target_rps"] == 490][0]
    r2k_300 = [x for x in rps_2k if x["target_rps"] == 300][0]
    r2k_320 = [x for x in rps_2k if x["target_rps"] == 320][0]

    rows.append(["RPS Saturation Result (p99 < 50 ms SLA)"])
    rows.append([
        "Payload",
        "Setup",
        "RPS",
        "p50",
        "p99",
        "RPS Improvement (vs L4 GPU)",
        "RPS Improvement (vs TPU v5e)",
    ])
    rows.append(["1K", "L4 GPU", "70/s", "—", "—", "Baseline", "—"])
    rows.append(["1K", "TPU v5e", "180/s (187/s max achieved)", "19.9ms", "29.6ms", "+157.1%", "Baseline"])
    rows.append([
        "1K",
        "TPU v6e (Opus Megakernel cb460828, 10-RPS grid)",
        "490/s",
        f"{r1k_490['p50']:.1f}ms",
        f"{r1k_490['p99']:.1f}ms",
        f"+{(490.0/70.0 - 1.0)*100:.1f}% (7.00x)",
        f"+{(490.0/180.0 - 1.0)*100:.1f}% vs 180/s (+{(490.0/187.0 - 1.0)*100:.1f}% vs 187/s)",
    ])
    rows.append([
        "1K",
        "TPU v6e (Opus Megakernel cb460828, 20-RPS grid)",
        "480/s",
        f"{r1k_480['p50']:.1f}ms",
        f"{r1k_480['p99']:.1f}ms",
        f"+{(480.0/70.0 - 1.0)*100:.1f}% (6.86x)",
        f"+{(480.0/180.0 - 1.0)*100:.1f}% vs 180/s (+{(480.0/187.0 - 1.0)*100:.1f}% vs 187/s)",
    ])
    rows.append(["2K", "L4 GPU", "40/s", "—", "—", "Baseline", "—"])
    rows.append(["2K", "TPU v5e", "90/s", "17.2ms", "26.5ms", "+125.0%", "Baseline"])
    rows.append([
        "2K",
        "TPU v6e (Opus Megakernel cb460828, max PASS)",
        "320/s",
        f"{r2k_320['p50']:.1f}ms",
        f"{r2k_320['p99']:.1f}ms",
        f"+{(320.0/40.0 - 1.0)*100:.1f}% (8.00x)",
        f"+{(320.0/90.0 - 1.0)*100:.1f}% (3.56x)",
    ])
    rows.append([
        "2K",
        "TPU v6e (Opus Megakernel cb460828, 300 RPS target)",
        "300/s",
        f"{r2k_300['p50']:.1f}ms",
        f"{r2k_300['p99']:.1f}ms",
        f"+{(300.0/40.0 - 1.0)*100:.1f}% (7.50x)",
        f"+{(300.0/90.0 - 1.0)*100:.1f}% (3.33x)",
    ])

    rows.append([])
    # 3. Cost Improvement (Matching Zhemin's Exact Formula: Hourly_Cost / (0.40 * RPS * 3600) * 1e6)
    def cost_per_1m(hourly_cost, rps):
        return hourly_cost / (0.40 * rps * 3600.0) * 1e6

    c_l4_1k = cost_per_1m(0.70, 70)
    c_l4_2k = cost_per_1m(0.70, 40)
    c_v5_120_1k_187 = cost_per_1m(1.20, 187)
    c_v5_120_1k_180 = cost_per_1m(1.20, 180)
    c_v5_120_2k = cost_per_1m(1.20, 90)
    c_v5_122_1k_187 = cost_per_1m(1.22, 187)
    c_v5_122_1k_180 = cost_per_1m(1.22, 180)
    c_v5_122_2k = cost_per_1m(1.22, 90)
    c_v6_1k_490 = cost_per_1m(2.70, 490)
    c_v6_1k_480 = cost_per_1m(2.70, 480)
    c_v6_2k_320 = cost_per_1m(2.70, 320)
    c_v6_2k_300 = cost_per_1m(2.70, 300)

    rows.append(["Cost Improvement (Matching Zhemin's Formula: Hourly Cost / (0.40 * Max RPS * 3600) * 1,000,000)"])
    rows.append([
        "Machine Type",
        "Machine Config",
        "Hourly Cost",
        "Cost per 1M request (1K Payload)",
        "Cost per 1M request (2K Payload)",
    ])
    rows.append([
        "g2-standard-4",
        "L4 GPUs: 1 | vCPUs: 4 | Memory: 16GiB | GPU Memory: 24GiB",
        0.7,
        round(c_l4_1k, 9),
        round(c_l4_2k, 8),
    ])
    rows.append([
        "ct5lp-hightpu-1t (Zhemin sheet $1.20, 187/90 RPS)",
        "vCPUs: 24 | Memory: 48 GB | TPU HBM: 16 GB",
        1.2,
        round(c_v5_120_1k_187, 9),
        round(c_v5_120_2k, 9),
    ])
    rows.append([
        "ct5lp-hightpu-1t (OD $1.22, 187/90 RPS)",
        "vCPUs: 24 | Memory: 48 GB | TPU HBM: 16 GB",
        1.22,
        round(c_v5_122_1k_187, 9),
        round(c_v5_122_2k, 9),
    ])
    rows.append([
        "ct5lp-hightpu-1t (OD $1.22, SLA-passing 180/90 RPS)",
        "vCPUs: 24 | Memory: 48 GB | TPU HBM: 16 GB",
        1.22,
        round(c_v5_122_1k_180, 9),
        round(c_v5_122_2k, 9),
    ])
    rows.append([
        "ct6e-standard-1t (Opus Megakernel cb460828, 490/320 RPS max PASS)",
        "TPU v6e: 1 chip | vCPUs: 44 | Memory: 176 GB | TPU HBM: 32 GB",
        2.7,
        round(c_v6_1k_490, 9),
        round(c_v6_2k_320, 9),
    ])
    rows.append([
        "ct6e-standard-1t (Opus Megakernel cb460828, 480/320 RPS 20-RPS grid)",
        "TPU v6e: 1 chip | vCPUs: 44 | Memory: 176 GB | TPU HBM: 32 GB",
        2.7,
        round(c_v6_1k_480, 9),
        round(c_v6_2k_320, 9),
    ])
    rows.append([
        "Cost Improvement (v5e $1.20 vs L4 $0.70 — Zhemin Baseline)",
        "",
        "",
        f"{(1.0 - c_v5_120_1k_187 / c_l4_1k)*100:.2f}% reduction",
        f"{(1.0 - c_v5_120_2k / c_l4_2k)*100:.2f}% reduction",
    ])
    rows.append([
        "Cost Improvement (v6e $2.70 vs L4 $0.70 — 490 / 320 RPS)",
        "",
        "",
        f"{(1.0 - c_v6_1k_490 / c_l4_1k)*100:.2f}% reduction",
        f"{(1.0 - c_v6_2k_320 / c_l4_2k)*100:.2f}% reduction",
    ])
    rows.append([
        "Cost Improvement (v6e $2.70 vs L4 $0.70 — 480 / 320 RPS)",
        "",
        "",
        f"{(1.0 - c_v6_1k_480 / c_l4_1k)*100:.2f}% reduction",
        f"{(1.0 - c_v6_2k_320 / c_l4_2k)*100:.2f}% reduction",
    ])
    rows.append([
        "Cost Improvement (v6e $2.70 vs v5e $1.20 @ 187/90 RPS — 490/320 RPS)",
        "",
        "",
        f"{(1.0 - c_v6_1k_490 / c_v5_120_1k_187)*100:.2f}% reduction",
        f"{(1.0 - c_v6_2k_320 / c_v5_120_2k)*100:.2f}% reduction",
    ])
    rows.append([
        "Cost Improvement (v6e $2.70 vs v5e $1.22 @ 187/90 RPS — 490/320 RPS)",
        "",
        "",
        f"{(1.0 - c_v6_1k_490 / c_v5_122_1k_187)*100:.2f}% reduction",
        f"{(1.0 - c_v6_2k_320 / c_v5_122_2k)*100:.2f}% reduction",
    ])
    rows.append([
        "Cost Improvement (v6e $2.70 vs v5e $1.22 @ 180/90 RPS — 490/320 RPS)",
        "",
        "",
        f"{(1.0 - c_v6_1k_490 / c_v5_122_1k_180)*100:.2f}% reduction",
        f"{(1.0 - c_v6_2k_320 / c_v5_122_2k)*100:.2f}% reduction",
    ])

    return rows


def main():
    rps_1k, rps_2k, k6_conc, single_batch = load_empirical_data()
    tab1 = build_tab1(rps_1k, rps_2k, k6_conc, single_batch)
    tab2 = build_tab2(rps_1k, rps_2k, k6_conc, single_batch)

    out_paths = [
        os.path.join(BASE_DIR, "ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx"),
        os.path.join(BASE_DIR, "ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx"),
        os.path.join(REPO_MEGAKERNEL_DIR, "ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx"),
    ]
    for p in out_paths:
        create_xlsx(
            p,
            [
                ("v6e_Megakernel_gid1161755388", tab1),
                ("Comparison_gid1972899730", tab2),
            ],
        )
        print(f"Wrote {p}")

    csv_paths = [
        os.path.join(SCRATCH_DIR, "TPU_v6e_Opus_Megakernel_cb460828_Zhemin_Comparison.csv"),
        os.path.join(REPO_MEGAKERNEL_DIR, "TPU_v6e_FP32_Megakernel_Zhemin_Spreadsheet_Comparison.csv"),
    ]
    for cp in csv_paths:
        with open(cp, "w", newline="") as f:
            w = csv.writer(f)
            for row in tab1:
                w.writerow(row)
            w.writerow([])
            for row in tab2:
                w.writerow(row)
        print(f"Wrote {cp}")
    print(f"Verified {len(rps_1k)} 1K RPS steps and {len(rps_2k)} 2K RPS steps.")


if __name__ == "__main__":
    main()
