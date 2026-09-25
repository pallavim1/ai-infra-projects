#!/usr/bin/env python3
"""Generates the Excel (.xlsx) and CSV comparison workbooks for TPU v6e FP32 Megakernel
tested on `jina-v2-embeddings-clean` (`max_model_len = 2048` ONLY, `max_num_batched_tokens = 2048`).

Includes:
1. Tab 1 (`TPU_v6e_Megakernel_gid1161755388`):
   Matches Zhemin's `gid=1161755388` tab AS-IS:
   - Batch Request Testing (`1KB` & `2KB`, Concurrency `1, 4, 8, 16`)
   - 1KB Dedicated Saturation (`100` to `290` RPS, saturating at `270 RPS` `< 50 ms` P99)
   - 2KB Dedicated Saturation (`90` to `160` RPS, saturating at `150 RPS` `< 50 ms` P99)
2. Tab 2 (`TPU_vs_L4_Comparison_gid1972899730`):
   Matches Zhemin's `gid=1972899730` tab AS-IS (for `1KB` & `2KB` tiers):
   - Concurrent Request Latency Comparison (L4 vs TPU v5e vs TPU v6e Megakernel)
   - RPS Saturation Result (`< 50 ms` P99 SLA)
   - Cost Improvement (Performance/$ & Monthly TCO at 100% and 40% Fleet Utilization)
   - Regional Availability
"""

import csv
import json
import os
import shutil
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BATCH_TESTING_ROWS = [
    # (Payload Size, Concurrency, v6e_tput, v6e_p50, v6e_p99, v5e_tput, v5e_p50, v5e_p99, l4_tput, l4_p50, l4_p99)
    ("1KB (1024 chars)", 1, "109.5/s", "9.0ms", "11.4ms", "85.8/s", "11.5ms", "12.7ms", "43.2/s", "20.7ms", "23.5ms"),
    ("1KB (1024 chars)", 4, "196.3/s", "20.5ms", "23.9ms", "187.4/s", "21.2ms", "22.8ms", "103.4/s", "38.1ms", "42.9ms"),
    ("1KB (1024 chars)", 8, "252.4/s", "31.4ms", "37.5ms", "187.6/s", "42.5ms", "44.5ms", "135.1/s", "58.7ms", "66.3ms"),
    ("1KB (1024 chars)", 16, "280.2/s", "56.9ms", "65.1ms", "186.8/s", "85.4ms", "89.9ms", "165.2/s", "96.5ms", "107.6ms"),
    ("2KB (2048 chars)", 1, "89.0/s", "11.3ms", "13.0ms", "59.5/s", "16.5ms", "17.7ms", "39.0/s", "24.3ms", "28.2ms"),
    ("2KB (2048 chars)", 4, "146.1/s", "27.4ms", "32.2ms", "97.7/s", "40.7ms", "42.8ms", "75.7/s", "52.1ms", "58.2ms"),
    ("2KB (2048 chars)", 8, "163.5/s", "50.1ms", "54.4ms", "96.6/s", "82.6ms", "86.0ms", "90.5/s", "87.4ms", "97.3ms"),
    ("2KB (2048 chars)", 16, "175.0/s", "91.5ms", "101.4ms", "96.5/s", "165.9ms", "171.2ms", "101.1/s", "156.9ms", "175.5ms"),
]

SATURATION_1KB_ROWS = [
    # (RPS, v6e_ach, v6e_p50, v6e_p99, v6e_sla, v5e_ach, v5e_p50, v5e_p99, v5e_sla)
    (100, "99.97", "11.5 ms", "13.6 ms", "✅ PASS", "100", "11.5 ms", "14.0 ms", "✅ PASS"),
    (120, "119.96", "12.2 ms", "15.0 ms", "✅ PASS", "120", "11.8 ms", "15.5 ms", "✅ PASS"),
    (140, "139.92", "13.1 ms", "15.1 ms", "✅ PASS", "140", "11.6 ms", "18.5 ms", "✅ PASS"),
    (160, "159.91", "12.5 ms", "16.3 ms", "✅ PASS", "160", "16.8 ms", "24.2 ms", "✅ PASS"),
    (180, "179.89", "12.7 ms", "25.5 ms", "✅ PASS", "180.1", "19.9 ms", "29.6 ms", "✅ PASS"),
    (190, "189.87", "12.3 ms", "22.7 ms", "✅ PASS", "187.8", "523.7 ms", "762.7 ms", "⚠️ SATURATED"),
    (200, "199.84", "17.9 ms", "26.6 ms", "✅ PASS", "189", "1709 ms", "2818 ms", "⚠️ SATURATED"),
    (220, "219.76", "18.5 ms", "27.2 ms", "✅ PASS", "187.9", "3738 ms", "6182 ms", "⚠️ SATURATED"),
    (240, "239.71", "19.7 ms", "28.4 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (260, "259.69", "20.7 ms", "36.2 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (270, "269.73", "19.0 ms", "28.3 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (280, "275.68", "47.6 ms", "190.1 ms", "⚠️ SATURATED", "—", "—", "—", "⚠️ SATURATED"),
    (290, "271.16", "507.9 ms", "836.9 ms", "⚠️ SATURATED", "—", "—", "—", "⚠️ SATURATED"),
]

SATURATION_2KB_ROWS = [
    (90, "89.96", "17.0 ms", "18.9 ms", "✅ PASS", "90", "17.2 ms", "26.5 ms", "✅ PASS"),
    (95, "94.97", "16.6 ms", "18.3 ms", "✅ PASS", "94.75", "218.6 ms", "346.5 ms", "⚠️ SATURATED"),
    (100, "99.96", "16.3 ms", "18.0 ms", "✅ PASS", "96.32", "1031 ms", "2185 ms", "⚠️ SATURATED"),
    (110, "109.94", "15.4 ms", "16.9 ms", "✅ PASS", "96.74", "3208 ms", "5170 ms", "⚠️ SATURATED"),
    (120, "119.94", "14.5 ms", "16.4 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (130, "129.92", "14.1 ms", "17.9 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (140, "139.92", "14.1 ms", "30.3 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (150, "149.90", "13.9 ms", "31.1 ms", "✅ PASS", "—", "—", "—", "⚠️ SATURATED"),
    (155, "151.34", "122.9 ms", "292.1 ms", "⚠️ SATURATED", "—", "—", "—", "⚠️ SATURATED"),
    (160, "149.57", "456.9 ms", "836.6 ms", "⚠️ SATURATED", "—", "—", "—", "⚠️ SATURATED"),
]

CONCURRENT_COMPARISON_ROWS = [
    # (Category, L4_p50, v5e_p50, v6e_p50, v6e_vs_l4_p50, v6e_vs_v5e_p50, L4_p99, v5e_p99, v6e_p99, v6e_vs_l4_p99, v6e_vs_v5e_p99)
    ("1KB (1024 chars ≈ 1,009 tokens) @ C=1", 20.7, 11.5, 9.0, "-56.5%", "-21.7%", 23.5, 12.7, 11.4, "-51.5%", "-10.2%"),
    ("2KB (2048 chars ≈ 2,016 tokens) @ C=1", 24.3, 16.5, 11.3, "-53.5%", "-31.5%", 28.2, 17.7, 13.0, "-53.9%", "-26.6%"),
    ("1KB (1024 chars ≈ 1,009 tokens) @ C=4", 38.1, 21.2, 20.5, "-46.2%", "-3.3%", 42.9, 22.8, 23.9, "-44.3%", "+4.8%"),
    ("2KB (2048 chars ≈ 2,016 tokens) @ C=4", 52.1, 40.7, 27.4, "-47.4%", "-32.7%", 58.2, 42.8, 32.2, "-44.7%", "-24.8%"),
    ("1KB (1024 chars ≈ 1,009 tokens) @ C=8", 58.7, 42.5, 31.4, "-46.5%", "-26.1%", 66.3, 44.5, 37.5, "-43.4%", "-15.7%"),
    ("2KB (2048 chars ≈ 2,016 tokens) @ C=8", 87.4, 82.6, 50.1, "-42.7%", "-39.3%", 97.3, 86.0, 54.4, "-44.1%", "-36.7%"),
]

RPS_SATURATION_SUMMARY_ROWS = [
    # (Payload, L4_RPS, v5e_RPS, v6e_RPS, v6e_P50, v6e_P99, v6e_vs_L4, v6e_vs_v5e)
    ("1KB (1,024 chars ≈ 1,009 tokens)", "70 RPS", "180 RPS", "270 RPS", "19.0 ms", "28.3 ms", "3.86x (+285.7%)", "1.50x (+50.0%)"),
    ("2KB (2,048 chars ≈ 2,016 tokens)", "40 RPS", "90 RPS", "150 RPS", "13.9 ms", "31.1 ms", "3.75x (+275.0%)", "1.67x (+66.7%)"),
]

COST_IMPROVEMENT_ROWS = [
    # (Payload, Platform, Hourly, Max_RPS_100, Perf_Per_Dollar_100, Rel_Perf_Dollar, Fleet_RPS_40, Nodes_for_1000RPS_40, Monthly_TCO_1000RPS_40, TCO_Savings_vs_L4)
    ("1KB (1,024 chars)", "NVIDIA L4 (g2-standard-8)", "$1.0124", "70 RPS", "69.14 RPS/$", "1.00x (Baseline)", "28.0 RPS", 36, "$26,606 / mo", "Baseline"),
    ("1KB (1,024 chars)", "TPU v5e-1 (ct5lp-hightpu-1t, FP32)", "$0.7990", "180 RPS", "225.28 RPS/$", "3.26x (+225.8%)", "72.0 RPS", 14, "$8,166 / mo", "-69.3% ($18,440/mo saved)"),
    ("1KB (1,024 chars)", "TPU v6e-1 (ct6e-standard-1t, FP32 Megakernel, max_len=2048)", "$0.9990", "270 RPS", "270.27 RPS/$", "3.91x (+290.9%)", "108.0 RPS", 10, "$7,293 / mo", "-72.6% ($19,313/mo saved)"),
    ("2KB (2,048 chars)", "NVIDIA L4 (g2-standard-8)", "$1.0124", "40 RPS", "39.51 RPS/$", "1.00x (Baseline)", "16.0 RPS", 63, "$46,560 / mo", "Baseline"),
    ("2KB (2,048 chars)", "TPU v5e-1 (ct5lp-hightpu-1t, FP32)", "$0.7990", "90 RPS", "112.64 RPS/$", "2.85x (+185.1%)", "36.0 RPS", 28, "$16,332 / mo", "-64.9% ($30,228/mo saved)"),
    ("2KB (2,048 chars)", "TPU v6e-1 (ct6e-standard-1t, FP32 Megakernel, max_len=2048)", "$0.9990", "150 RPS", "150.15 RPS/$", "3.80x (+280.0%)", "60.0 RPS", 17, "$12,398 / mo", "-73.4% ($34,162/mo saved)"),
]


def style_sheet(ws):
    header_fill = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")
    sub_fill = PatternFill(start_color="D9E1F2", end_color="D9E1F2", fill_type="solid")
    pass_fill = PatternFill(start_color="E2EFDA", end_color="E2EFDA", fill_type="solid")
    sat_fill = PatternFill(start_color="FCE4D6", end_color="FCE4D6", fill_type="solid")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    sub_font = Font(name="Calibri", size=11, bold=True, color="1F4E78")
    bold_font = Font(name="Calibri", size=10, bold=True)
    reg_font = Font(name="Calibri", size=10)
    thin = Side(border_style="thin", color="BFBFBF")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    for row in ws.iter_rows():
        for cell in row:
            if cell.value is not None:
                cell.border = border
                val_str = str(cell.value)
                if cell.row in (1,):
                    cell.fill = header_fill
                    cell.font = header_font
                elif val_str.startswith("SECTION") or val_str.startswith("1.") or val_str.startswith("2.") or val_str.startswith("3.") or val_str.startswith("4."):
                    cell.fill = sub_fill
                    cell.font = sub_font
                elif val_str in ("Payload Size", "RPS", "Payload Category", "Platform"):
                    cell.fill = header_fill
                    cell.font = header_font
                elif "✅ PASS" in val_str:
                    cell.fill = pass_fill
                    cell.font = bold_font
                elif "⚠️ SATURATED" in val_str:
                    cell.fill = sat_fill
                    cell.font = bold_font
                else:
                    cell.font = reg_font
                cell.alignment = Alignment(vertical="center", wrap_text=False)

    for col in ws.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.value is not None:
                max_len = max(max_len, len(str(cell.value)))
        ws.column_dimensions[col_letter].width = min(max(max_len + 4, 14), 52)


def build_tab1_gid1161755388(ws):
    ws.append(["TPU v6e FP32 4-Layer Megakernel (Branch: jina-v2-embeddings-clean | Strictly max_model_len=2048 & max_num_batched_tokens=2048) — Matching Zhemin gid=1161755388 AS-IS"])
    ws.append([])
    ws.append(["1. Batch Request Testing: A single HTTP request contains multiple prompts (Random Chars: 1KB ≈ 1,009 tokens, 2KB ≈ 2,016 tokens)"])
    ws.append([
        "Payload Size", "Concurrency",
        "TPU v6e Megakernel Throughput (max_len=2048)", "TPU v6e Megakernel p50", "TPU v6e Megakernel p99",
        "TPU v5e Baseline Throughput (Zhemin)", "TPU v5e Baseline p50", "TPU v5e Baseline p99",
        "NVIDIA L4 Throughput (Zhemin)", "NVIDIA L4 p50", "NVIDIA L4 p99"
    ])
    for r in BATCH_TESTING_ROWS:
        ws.append(list(r))

    ws.append([])
    ws.append(["2. 1KB Dedicated Saturation (1,024 random chars ≈ 1,009 tokens, FP32, < 50 ms P99 SLA)"])
    ws.append([
        "RPS",
        "TPU v6e Megakernel Achieved", "TPU v6e Megakernel P50", "TPU v6e Megakernel P99", "TPU v6e Megakernel SLA (<50ms)",
        "TPU v5e Achieved (Zhemin)", "TPU v5e P50 (Zhemin)", "TPU v5e P99 (Zhemin)", "TPU v5e SLA (Zhemin)"
    ])
    for r in SATURATION_1KB_ROWS:
        ws.append(list(r))

    ws.append([])
    ws.append(["3. 2KB Dedicated Saturation (2,048 random chars ≈ 2,016 tokens, FP32, < 50 ms P99 SLA)"])
    ws.append([
        "RPS",
        "TPU v6e Megakernel Achieved", "TPU v6e Megakernel P50", "TPU v6e Megakernel P99", "TPU v6e Megakernel SLA (<50ms)",
        "TPU v5e Achieved (Zhemin)", "TPU v5e P50 (Zhemin)", "TPU v5e P99 (Zhemin)", "TPU v5e SLA (Zhemin)"
    ])
    for r in SATURATION_2KB_ROWS:
        ws.append(list(r))
    style_sheet(ws)


def build_tab2_gid1972899730(ws):
    ws.append(["TPU v6e FP32 Megakernel (Branch: jina-v2-embeddings-clean | Strictly max_model_len=2048) vs TPU v5e & NVIDIA L4 — Matching Zhemin gid=1972899730"])
    ws.append([])
    ws.append(["1. Concurrent Request Latency Comparison (1KB & 2KB Random Characters)"])
    ws.append([
        "Payload Category",
        "L4 P50 (ms)", "TPU v5e P50 (ms)", "TPU v6e Megakernel P50 (ms)", "v6e vs L4 P50", "v6e vs v5e P50",
        "L4 P99 (ms)", "TPU v5e P99 (ms)", "TPU v6e Megakernel P99 (ms)", "v6e vs L4 P99", "v6e vs v5e P99"
    ])
    for r in CONCURRENT_COMPARISON_ROWS:
        ws.append(list(r))

    ws.append([])
    ws.append(["2. RPS Saturation Result (< 50 ms P99 Latency SLA)"])
    ws.append([
        "Payload Size", "NVIDIA L4 Max RPS", "TPU v5e Max RPS", "TPU v6e Megakernel Max RPS (max_len=2048)",
        "TPU v6e P50 at Max RPS", "TPU v6e P99 at Max RPS", "v6e Gain vs L4", "v6e Gain vs TPU v5e"
    ])
    for r in RPS_SATURATION_SUMMARY_ROWS:
        ws.append(list(r))

    ws.append([])
    ws.append(["3. Cost Improvement & Fleet TCO Analysis (100% Max Capacity & 40% Production Fleet Utilization)"])
    ws.append([
        "Payload Size", "Platform", "On-Demand Price ($/hr)", "Max Validated RPS (<50ms P99)",
        "Throughput per Dollar (RPS/$/hr)", "Relative Perf/$ vs L4",
        "Effective RPS @ 40% Fleet Util", "Chips Needed for 1,000 RPS (@40% Util)",
        "Monthly Fleet Cost (1,000 RPS @ 40% Util)", "Monthly TCO Savings vs L4"
    ])
    for r in COST_IMPROVEMENT_ROWS:
        ws.append(list(r))

    ws.append([])
    ws.append(["4. Regional Availability (GCP Americas, Europe, Asia-Pacific)"])
    ws.append(["Accelerator", "Americas Regions", "Europe Regions", "Asia-Pacific Regions"])
    ws.append(["NVIDIA L4 (g2)", "us-central1, us-east1, us-east4, us-west1, us-west4, northamerica-northeast1, southamerica-east1", "europe-west1, europe-west2, europe-west3, europe-west4, europe-north1", "asia-east1, asia-northeast1, asia-northeast3, asia-south1, asia-southeast1, australia-southeast1"])
    ws.append(["Cloud TPU v5e", "us-central1, us-east1, us-west1, us-west4, southamerica-west1", "europe-west4", "asia-east1, asia-northeast1"])
    ws.append(["Cloud TPU v6e (Trillium)", "us-central1, us-east1, us-east5, us-south1, us-west4, southamerica-west1", "europe-west4", "asia-east1, asia-northeast1"])
    style_sheet(ws)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    xlsx_path = os.path.join(out_dir, "ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx")
    csv_path = os.path.join(out_dir, "TPU_v6e_FP32_Megakernel_Zhemin_Spreadsheet_Comparison.csv")
    json_path = os.path.join(out_dir, "v6e_fp32_megakernel_2k_results.json")

    wb = openpyxl.Workbook()
    ws1 = wb.active
    ws1.title = "v6e_Megakernel_gid1161755388"
    build_tab1_gid1161755388(ws1)

    ws2 = wb.create_sheet(title="Comparison_gid1972899730")
    build_tab2_gid1972899730(ws2)
    wb.save(xlsx_path)

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        for row in ws1.iter_rows(values_only=True):
            w.writerow(row)
        w.writerow([])
        for row in ws2.iter_rows(values_only=True):
            w.writerow(row)

    data = {
        "branch": "jina-v2-embeddings-clean",
        "commit": "a8733e93",
        "max_model_len": 2048,
        "max_num_batched_tokens": 2048,
        "truncate_prompt_tokens": 2048,
        "dtype": "float32",
        "batch_request_testing": BATCH_TESTING_ROWS,
        "saturation_1kb": SATURATION_1KB_ROWS,
        "saturation_2kb": SATURATION_2KB_ROWS,
        "concurrent_comparison": CONCURRENT_COMPARISON_ROWS,
        "rps_saturation_summary": RPS_SATURATION_SUMMARY_ROWS,
        "cost_improvement": COST_IMPROVEMENT_ROWS,
    }
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    artifact_dir = "/usr/local/google/home/pallaviam/.gemini/jetski/brain/292ebc90-dbb8-4a33-8f1a-ca6dee63d8ff"
    if os.path.isdir(artifact_dir):
        shutil.copy2(xlsx_path, os.path.join(artifact_dir, "ATP_AIC2_Benchmarks_TPU_v6e_FP32_Megakernel_vs_L4_and_Baselines.xlsx"))
        shutil.copy2(xlsx_path, os.path.join(artifact_dir, "ATP_AIC2_Benchmarks_TPU_v6e_v5e_FP32_and_BF16_vs_L4.xlsx"))

    print(f"Successfully generated:\n  - {xlsx_path}\n  - {csv_path}\n  - {json_path}")


if __name__ == "__main__":
    main()
