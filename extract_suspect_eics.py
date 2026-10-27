from pathlib import Path
from datetime import datetime
import re
import textwrap
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

SAMPLE_FILE = "data/sample_NEG_MS1.mzXML"
BLANK_FILE = None
SUSPECT_FILE = "data/Ar_CF3_suspect_list.csv"
PPM = 10
RT_RANGE = (60, 1200)
MS_SETTINGS = {
    1: {"peak_height": 3000, "min_nonzero_points": 6, "point_window_min": 2.0},
    2: {"peak_height": 1000, "min_nonzero_points": 2, "point_window_min": 2.0},
}
BLANK_RATIO = 3
MAX_PEAK_WIDTH_MIN = 3.0
MERGE_GAP_MIN = 0.05
MAX_SCAN_GAP_FACTOR = 3.0
BATCH_SIZE = 256
DPI = 300
MAX_DISPLAY_COMBINATIONS = 3
SAVE_REJECTED_EICS = True
DIAGNOSTIC_MZ = (194.9830,)

MATCH_COLUMNS = [
    "source", "scan_id", "RT_min", "target_mz", "observed_fragment_mz",
    "mass_error_ppm", "intensity", "precursor_index", "precursor_mz",
    "isolation_lower_offset", "isolation_upper_offset",
]


def get_settings(path):
    name = Path(path).stem.upper()
    modes = set(re.findall(r"(?<![A-Z])(NEG|POS)(?![A-Z])", name))
    levels = set(re.findall(r"(?<![A-Z])MS([12])(?!\d)", name))
    if len(modes) != 1 or len(levels) != 1:
        raise ValueError(f"Filename must contain NEG/POS and MS1/MS2: {path}")
    return ("negative" if "NEG" in modes else "positive"), int(levels.pop())


def enum_code(value):
    return int(getattr(value, "value", value))


def load_data(path, mode, level):
    from pyopenms import MSExperiment, MzXMLFile
    experiment = MSExperiment()
    MzXMLFile().load(str(path), experiment)
    spectra = []
    expected = 2 if mode == "negative" else 1
    for number, spectrum in enumerate(experiment, 1):
        rt = spectrum.getRT()
        if spectrum.getMSLevel() != level or not RT_RANGE[0] <= rt <= RT_RANGE[1]:
            continue
        if enum_code(spectrum.getInstrumentSettings().getPolarity()) not in (0, expected):
            raise ValueError(f"Scan polarity conflicts with filename: {path}")
        if enum_code(spectrum.getType()) != 1:
            raise ValueError(f"Centroid spectra are required: {path}")
        mz, intensity = spectrum.get_peaks()
        order = np.argsort(mz)
        mz = np.asarray(mz[order], dtype=np.float64)
        intensity = np.asarray(intensity[order], dtype=np.float64)
        if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
            raise ValueError(f"Non-finite spectrum: {path}")
        precursors = []
        if level == 2:
            precursors = [
                (p.getMZ(), p.getIsolationWindowLowerOffset(), p.getIsolationWindowUpperOffset())
                for p in spectrum.getPrecursors()
            ]
        event_key = "MS1"
        if level == 2:
            filter_text = ""
            for key in ("filter string", "filterString", "scan filter"):
                if spectrum.metaValueExists(key):
                    value = spectrum.getMetaValue(key)
                    filter_text = value.decode() if isinstance(value, bytes) else str(value)
                    filter_text = " ".join(filter_text.split())
                    if filter_text:
                        break
            precursor_details = [
                (round(p.getMZ(), 6), round(p.getIsolationWindowLowerOffset(), 6),
                 round(p.getIsolationWindowUpperOffset(), 6), round(p.getActivationEnergy(), 6))
                for p in spectrum.getPrecursors()
            ]
            if not filter_text and not any(p[0] > 0 for p in precursor_details):
                raise ValueError(f"MS2 scan {number} lacks filter/precursor metadata; cannot separate events safely: {path}")
            event_key = json.dumps({"filter": filter_text, "isolation_and_energy": precursor_details}, sort_keys=True)
        spectra.append({
            "rt": rt, "mz": mz, "intensity": intensity, "event_key": event_key,
            "scan_id": spectrum.getNativeID() or str(number),
            "precursors": precursors,
        })
    if not spectra:
        raise ValueError(f"No MS{level} scans in selected RT range: {path}")
    spectra.sort(key=lambda scan: scan["rt"])
    times, groups = np.unique([scan["rt"] for scan in spectra], return_inverse=True)
    return spectra, times, groups


def extract_batch(spectra, groups, number_times, targets):
    values = np.full((number_times, len(targets)), -np.inf)
    tolerance = targets * PPM / 1e6
    for scan, group in zip(spectra, groups):
        mz, intensity = scan["mz"], scan["intensity"]
        left = np.searchsorted(mz, targets - tolerance, side="left")
        right = np.searchsorted(mz, targets + tolerance, side="right")
        totals = np.zeros(len(targets))
        counts = right - left
        for offset in range(int(counts.max(initial=0))):
            valid = counts > offset
            totals[valid] += intensity[left[valid] + offset]
        np.maximum(values[group], totals, out=values[group])
    return values


def ms2_matches(spectra, target, source, start, end):
    tolerance = target * PPM / 1e6
    for scan in spectra:
        if not start <= scan["rt"] <= end:
            continue
        left = np.searchsorted(scan["mz"], target - tolerance, side="left")
        right = np.searchsorted(scan["mz"], target + tolerance, side="right")
        for observed, signal in zip(scan["mz"][left:right], scan["intensity"][left:right]):
            if signal <= 0:
                continue
            for index, precursor in enumerate(scan["precursors"] or [None], 1):
                yield {
                    "source": source, "scan_id": scan["scan_id"], "RT_min": scan["rt"] / 60,
                    "target_mz": target, "observed_fragment_mz": observed,
                    "mass_error_ppm": (observed - target) / target * 1e6, "intensity": signal,
                    "precursor_index": index if precursor else None,
                    "precursor_mz": precursor[0] if precursor else None,
                    "isolation_lower_offset": precursor[1] if precursor else None,
                    "isolation_upper_offset": precursor[2] if precursor else None,
                }


def screen_peaks(rt, signal, bt, blank, settings):
    active = np.flatnonzero(signal > 0)
    if not len(active):
        return []
    cadence = np.median(np.diff(rt)) if len(rt) > 1 else np.inf
    splits = np.flatnonzero(
        (np.diff(active) > 1)
        & (np.diff(rt[active]) > MERGE_GAP_MIN * 60)
    ) + 1
    records = []
    for group in np.split(active, splits):
        start, end = int(group[0]), int(group[-1])
        left, right = max(0, start - 1), min(len(rt) - 1, end + 1)
        apex = int(group[np.argmax(signal[group])])
        width = (rt[right] - rt[left]) / 60
        reasons = []
        times = rt[group]
        apex_time = rt[apex]
        window = settings["point_window_min"] * 60
        window_starts = np.r_[apex_time - window, times[
            (times >= apex_time - window) & (times <= apex_time)
        ]]
        counts = (
            np.searchsorted(times, window_starts + window, side="right")
            - np.searchsorted(times, window_starts, side="left")
        )
        points_in_window = int(counts.max())
        if points_in_window < settings["min_nonzero_points"]:
            reasons.append("too_few_nonzero_points_in_apex_window")
        if signal[apex] < settings["peak_height"]:
            reasons.append("peak_height_below_threshold")
        if start == 0 or end == len(rt) - 1:
            reasons.append("no_complete_return_to_baseline")
        if width <= 0 or width >= MAX_PEAK_WIDTH_MIN:
            reasons.append("peak_width_out_of_range")
        if right > left and np.max(np.diff(rt[left:right + 1])) > MAX_SCAN_GAP_FACTOR * cadence:
            reasons.append("scan_time_gap")
        blank_max = np.nan
        blank_check = "not_performed"
        if blank is not None:
            mask = (bt >= rt[left]) & (bt <= rt[right])
            if not len(bt) or rt[left] < bt[0] or rt[right] > bt[-1] or not mask.any():
                reasons.append("insufficient_blank_coverage")
            else:
                blank_max = float(max(
                    blank[mask].max(),
                    np.interp([rt[left], rt[right]], bt, blank).max(),
                ))
                if signal[apex] < BLANK_RATIO * blank_max:
                    reasons.append("sample_blank_ratio_below_limit")
            blank_check = "failed" if any(
                r in reasons for r in ("insufficient_blank_coverage", "sample_blank_ratio_below_limit")
            ) else "passed"
        records.append({
            "RT_start_min": rt[left] / 60,
            "RT_apex_min": rt[apex] / 60,
            "RT_end_min": rt[right] / 60,
            "peak_width_min": width,
            "nonzero_points": len(group),
            "points_in_2min": points_in_window,
            "sample_peak_height": signal[apex],
            "blank_max": blank_max,
            "blank_check": blank_check,
            "passed": not reasons,
            "reason": ";".join(reasons) if reasons else "passed",
        })
    return records


def load_suspects(path, mode, level):
    table = pd.read_csv(path)
    table.columns = table.columns.str.strip()
    required = {"mz", "polarity", "sources"}
    if not required.issubset(table.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(table.columns))}")
    table["mz"] = pd.to_numeric(table["mz"], errors="coerce")
    table["polarity"] = table["polarity"].astype(str).str.strip().str.lower()
    table = table.loc[(table["polarity"] == mode) & np.isfinite(table["mz"]) & (table["mz"] > 0)]
    rows = []
    for _, row in table.iterrows():
        if pd.isna(row["sources"]):
            continue
        formula = str(row.get("ion_formula", ""))
        if formula == "nan":
            formula = ""
        for source in str(row["sources"]).split("|"):
            fields = [field.strip() for field in source.split(";") if field.strip()]
            if not fields:
                continue
            properties = {}
            scaffold = []
            for field in fields:
                if "=" in field:
                    key, value = field.split("=", 1)
                    properties[key.strip().lower()] = value.strip()
                else:
                    scaffold.append(field)
            variant = properties.get("variant", "")
            if level == 1 and variant.lower() != "base_ion":
                continue
            try:
                count = int(properties["cf3"])
                if count not in (1, 2) or not scaffold:
                    raise ValueError()
                parts = scaffold + ["CF3"] * count
                groups = properties.get("groups", "none")
                if groups.lower() != "none":
                    parts.extend(g.strip() for g in groups.split("+") if g.strip())
                combination = " + ".join(parts)
            except (KeyError, ValueError):
                combination = source.strip()
            if level == 2:
                combination += f" [{variant or 'variant unspecified'}]"
            rows.append({"mz": float(row["mz"]), "polarity": mode,
                         "ion_formula": formula, "variant": variant,
                         "combination": combination, "source": source.strip()})
    if not rows:
        raise ValueError("No matching suspects after polarity and variant filtering.")
    metadata = pd.DataFrame(rows).drop_duplicates()
    mapping = {float(mz): group.reset_index(drop=True)
               for mz, group in metadata.groupby("mz", sort=True)}
    return np.array(sorted(mapping)), metadata, mapping


def combination_footer(metadata):
    labels = list(dict.fromkeys(
        (f"{r.ion_formula}: " if r.ion_formula else "") + r.combination
        for r in metadata.itertuples(index=False)
    ))
    lines = ["Candidate combinations (not structural identification):"]
    for label in labels[:MAX_DISPLAY_COMBINATIONS]:
        lines.extend(textwrap.wrap(label, width=100) or [label])
    if len(labels) > MAX_DISPLAY_COMBINATIONS:
        lines.append(f"+ {len(labels) - MAX_DISPLAY_COMBINATIONS} more; see combinations.csv for this ion.")
    return "\n".join(lines), len(lines)


def group_events(scans):
    events = {}
    for scan in scans:
        events.setdefault(scan["event_key"], []).append(scan)
    return events


def time_index(scans):
    return np.unique([scan["rt"] for scan in scans], return_inverse=True)


def process_event(scans, bscans, mode, level, ions, combinations, output, use_blank, event_label):
    settings = MS_SETTINGS[level]
    rt, groups = time_index(scans)
    bt, bgroups = time_index(bscans)
    audit_columns = [
        "mz", "polarity", "MS_level", "RT_start_min", "RT_apex_min", "RT_end_min",
        "peak_width_min", "nonzero_points", "points_in_2min", "sample_peak_height", "blank_max", "blank_check", "passed", "reason",
    ]
    pd.DataFrame(columns=audit_columns).to_csv(output / "peak_audit.csv", index=False)
    summaries, accepted = [], []
    saved = 0
    with tqdm(total=len(ions), desc="Extracting and screening", unit="ion") as progress:
        for first in range(0, len(ions), BATCH_SIZE):
            targets = ions[first:first + BATCH_SIZE]
            ys = extract_batch(scans, groups, len(rt), targets)
            bs = extract_batch(bscans, bgroups, len(bt), targets) if use_blank else None
            batch_audit = []
            for column, ion in enumerate(targets):
                y = ys[:, column]
                blank = bs[:, column] if use_blank else None
                records = screen_peaks(rt, y, bt, blank, settings)
                peaks = [record for record in records if record["passed"]]
                batch_audit.extend(dict(mz=ion, polarity=mode, MS_level=level, **r) for r in records)
                accepted.extend(dict(mz=ion, polarity=mode, MS_level=level, **r) for r in peaks)
                summaries.append({
                    "event": event_label,
                    "mz": ion, "candidate_regions": len(records), "accepted_peaks": len(peaks),
                    "status": "passed" if peaks else ("rejected" if records else "no_signal"),
                })
                diagnostic = any(abs(ion - target) <= target * PPM / 1e6 for target in DIAGNOSTIC_MZ)
                export_rejected = level == 2 and SAVE_REJECTED_EICS and bool(records)
                if peaks or export_rejected or diagnostic:
                    name = f"{first + column + 1:06d}_mz_{ion:.5f}"
                    destination = output if peaks else output / "rejected"
                    destination.mkdir(exist_ok=True)
                    pd.DataFrame(records).to_csv(destination / f"{name}_audit.csv", index=False)
                    eic_data = {"RT_min": rt / 60, "Sample": y}
                    if use_blank:
                        eic_data["Blank"] = np.interp(rt, bt, blank, left=np.nan, right=np.nan) if len(bt) else np.full(len(rt), np.nan)
                    pd.DataFrame(eic_data).to_csv(
                        destination / f"{name}_EIC.csv", index=False
                    )
                    if level == 2:
                        records2 = list(ms2_matches(scans, ion, "Sample", rt[0], rt[-1]))
                        if use_blank:
                            records2.extend(ms2_matches(bscans, ion, "Blank", rt[0], rt[-1]))
                        pd.DataFrame(records2, columns=MATCH_COLUMNS).to_csv(
                            destination / f"{name}_precursors.csv", index=False
                        )
                    ion_metadata = combinations[float(ion)]
                    ion_metadata.to_csv(destination / f"{name}_combinations.csv", index=False, encoding="utf-8-sig")
                    if peaks or diagnostic:
                        footer, line_count = combination_footer(ion_metadata)
                        footer += "\n" + "\n".join(textwrap.wrap(event_label, width=100))
                        line_count = len(footer.splitlines())
                        if not peaks:
                            reasons = sorted({r["reason"] for r in records}) or ["no_signal"]
                            reason_lines = textwrap.wrap("Rejected: " + " | ".join(reasons), width=100)
                            footer += "\n" + "\n".join(reason_lines)
                            line_count += len(reason_lines)
                        if not use_blank:
                            footer += "\nBlank comparison: not performed"
                            line_count += 1
                        footer_height = 0.22 * line_count + 0.2
                        figure_height = 4 + footer_height
                        fig, ax = plt.subplots(figsize=(9, figure_height))
                        ax.plot(rt / 60, y, color="black", linewidth=1, label="Sample")
                        if use_blank:
                            ax.plot(bt / 60, blank, color="gray", linestyle="--", linewidth=1, label="Blank")
                        ax.axhline(settings["peak_height"], color="steelblue", linestyle=":", label="Peak threshold")
                        for n, peak in enumerate(peaks, 1):
                            ax.axvspan(peak["RT_start_min"], peak["RT_end_min"], color="orange", alpha=.15)
                            ax.annotate(
                                f'P{n}: {peak["RT_apex_min"]:.2f} min',
                                (peak["RT_apex_min"], peak["sample_peak_height"]),
                                xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9,
                            )
                        ax.set(xlabel="Retention time (min)", ylabel="Intensity",
                               title=f"{mode.capitalize()} MS{level} | m/z {ion:.5f}")
                        ax.margins(y=.2)
                        ax.legend(frameon=False)
                        fig.text(0.08, 0.04 / figure_height, footer, ha="left", va="bottom", fontsize=9)
                        fig.tight_layout(rect=(0, footer_height / figure_height, 1, 1))
                        fig.savefig(destination / f"{name}.png", dpi=DPI)
                        plt.close(fig)
                        saved += 1
                progress.set_postfix(images=saved, peaks=len(accepted), refresh=False)
                progress.update(1)
            pd.DataFrame(batch_audit, columns=audit_columns).to_csv(
                output / "peak_audit.csv", mode="a", header=False, index=False
            )
    pd.DataFrame(summaries).to_csv(output / "screening_summary.csv", index=False)
    pd.DataFrame(accepted, columns=audit_columns).to_csv(output / "accepted_peaks.csv", index=False)
    return summaries


def main():
    mode, level = get_settings(SAMPLE_FILE)
    use_blank = BLANK_FILE is not None and bool(str(BLANK_FILE).strip())
    if use_blank and get_settings(BLANK_FILE) != (mode, level):
        raise ValueError("Sample and blank polarity or MS level do not match.")
    ions, metadata, combinations = load_suspects(SUSPECT_FILE, mode, level)
    print("Loading sample...")
    scans, _, _ = load_data(SAMPLE_FILE, mode, level)
    bscans = []
    if use_blank:
        print("Loading blank...")
        bscans, _, _ = load_data(BLANK_FILE, mode, level)
    output = Path(SAMPLE_FILE).resolve().parent / (
        f"{Path(SAMPLE_FILE).stem}_by_event_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output.mkdir(exist_ok=False)
    print(f"Output folder: {output}")
    metadata.to_csv(output / "selected_suspects.csv", index=False, encoding="utf-8-sig")
    settings = MS_SETTINGS[level]
    pd.DataFrame([dict(
        sample=SAMPLE_FILE, blank=BLANK_FILE, suspects=SUSPECT_FILE,
        polarity=mode, MS_level=level, ppm=PPM, blank_comparison_enabled=use_blank,
        blank_ratio=BLANK_RATIO if use_blank else np.nan,
        variant_filter="base_ion" if level == 1 else "all",
        RT_start_seconds=RT_RANGE[0], RT_end_seconds=RT_RANGE[1],
        max_peak_width_min=MAX_PEAK_WIDTH_MIN, merge_gap_min=MERGE_GAP_MIN,
        max_scan_gap_factor=MAX_SCAN_GAP_FACTOR, **settings,
    )]).to_csv(output / "parameters.csv", index=False)
    events = group_events(scans)
    blank_events = group_events(bscans)
    index_rows, all_summaries = [], []
    for number, (key, event_scans) in enumerate(events.items(), 1):
        event_id = f"MS2_event_{number:03d}" if level == 2 else "MS1"
        event_output = output / event_id
        event_output.mkdir()
        matched_blank = blank_events.get(key, [])
        label = event_id + (" | " + key if level == 2 else "")
        print(f"{event_id}: {len(event_scans)} sample scans; {len(matched_blank)} blank scans")
        index_rows.append(dict(event=event_id, event_key=key,
                               sample_scans=len(event_scans), blank_scans=len(matched_blank),
                               blank_status=("matched" if matched_blank else "missing_event") if use_blank else "not_performed"))
        results = process_event(event_scans, matched_blank, mode, level, ions,
                                combinations, event_output, use_blank, label)
        all_summaries.extend(results)
    pd.DataFrame(index_rows).to_csv(output / "event_index.csv", index=False)
    pd.DataFrame(all_summaries).to_csv(output / "screening_summary.csv", index=False)
    print(f"Finished. Output: {output}")


if __name__ == "__main__":
    main()
