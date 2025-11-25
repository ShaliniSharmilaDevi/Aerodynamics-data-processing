"""
Configuration objects and YAML-based overrides for CFD post-processing.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# -----------------------------------------
# Logging Configuration
# -----------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s"
)

# -----------------------------------------
# CFD Configuration using dataclass
# -----------------------------------------
@dataclass
class CFDConfig:
    """Configuration for aerodynamic post-processing.

       Attributes
       ----------
       rho : float
           Freestream air density [kg/m^3].
       velocity : float
           Freestream velocity [m/s].
       col_aoa : str
           Column name for angle of attack.
       col_drag : str
           Column name for drag force.
       col_lift : str
           Column name for lift force.
       col_moment : str
           Column name for pitching moment.
       col_area : str
           Column name for reference area.
       col_chord : Optional[str]
           Column name for chord length (for Cm), if available.
       steady_window : int
           Window size for steady-state rolling standard deviation.
       steady_threshold : float
           Threshold (fraction of mean) for steady-state detection.
       zscore_threshold : float
           Z-score threshold for outlier removal.
       output_dir : Optional[Path]
           Optional custom output directory. If None, folder/post_processed is used.
       """

    # Flow conditions
    rho: float = 1.225          # Air density [kg/m^3]
    velocity: float = 50.0      # Freestream velocity [m/s]

    # Column names in STAR-CCM+ CSV
    col_aoa: str = "Angle_of_Attack"
    col_drag: str = "Drag_Force"
    col_lift: str = "Lift_Force"
    col_moment: str = "Pitching_Moment"
    col_area: str = "Reference_Area"
    col_chord: Optional[str] = None   # For Cm normalization if needed

    # Steady-state detection
    steady_window: int = 50
    steady_threshold: float = 0.01  # as fraction of mean

    # Outlier filtering
    zscore_threshold: float = 3.0

    # Optional: path where outputs go (plots, exports)
    output_dir: Optional[Path] = None

    @property
    def q_inf(self) -> float:
        """Return freestream dynamic pressure."""
        return 0.5 * self.rho * (self.velocity ** 2)


# -----------------------------------------
# Optional: Load YAML config if available
# -----------------------------------------
def load_config_from_yaml(yaml_path: Path) -> CFDConfig:
    """Load CFDConfig overrides from config.yml if present.

        Parameters
        ----------
        folder : Path
            Folder that may contain a config.yml file.

        Returns
        -------
        CFDConfig
            Configuration instance with YAML overrides applied if available.
        """
    cfg = CFDConfig()

    if not yaml_path.exists():
        logging.info("No config.yml found; using default CFDConfig.")
        return cfg

    try:
        import yaml
    except ImportError:
        logging.warning("PyYAML not installed; ignoring config.yml.")
        return cfg

    with yaml_path.open("r") as f:
        data = yaml.safe_load(f) or {}

    # Map YAML keys to dataclass attributes if present
    for field in cfg.__dataclass_fields__.keys():
        if field in data:
            setattr(cfg, field, data[field])

    logging.info(f"Loaded configuration from {yaml_path}")
    return cfg


# -----------------------------------------
# Utility: Detect steady state region
# -----------------------------------------
def detect_steady_state(series: pd.Series, window: int, threshold: float) -> int:
    """
    Returns the index from which the signal becomes steady-state.

    Parameters
    ----------
    series : pd.Series
        Time/iteration series of a quantity (e.g. drag force).
    window : int
        Rolling window size for standard deviation.
    threshold : float
        Threshold as a fraction of the mean value.

    Returns
    -------
    int
        First index of the steady-state region.
    """
    rolling_std = series.rolling(window).std()
    mean_val = series.mean()

    if np.isclose(mean_val, 0):
        # If mean is ~0, fall back to no steady-state detection
        return 0

    steady_mask = rolling_std < (threshold * abs(mean_val))
    steady_idx = steady_mask[steady_mask].index.min() if steady_mask.any() else None

    return int(steady_idx) if steady_idx is not None else 0


# -----------------------------------------
# Load and Validate CSV
# -----------------------------------------
def load_starccm_csv(path: Path, config: CFDConfig) -> Optional[pd.DataFrame]:
    """Load a STAR-CCM+ CSV file and validate required columns.

       Parameters
       ----------
       path : Path
           CSV file path.
       config : CFDConfig
           Configuration specifying column names.

       Returns
       -------
       Optional[pd.DataFrame]
           Loaded DataFrame if successful, otherwise None.
       """
    try:
        df = pd.read_csv(path, comment="#", skip_blank_lines=True)
    except Exception as e:
        logging.error(f"Failed to read {path}: {e}")
        return None

    required = [config.col_aoa, config.col_drag, config.col_area]
    missing = [c for c in required if c not in df.columns]
    if missing:
        logging.error(f"{path.name}: missing required columns: {missing}")
        return None

    # Lift and moment are optional, but log if missing
    opt_missing = []
    for c in [config.col_lift, config.col_moment]:
        if c is not None and c not in df.columns:
            opt_missing.append(c)
    if opt_missing:
        logging.warning(f"{path.name}: optional columns missing: {opt_missing}")

    return df


# -----------------------------------------
# Compute Aerodynamic Coefficients
# -----------------------------------------
def compute_coefficients(df: pd.DataFrame, config: CFDConfig) -> pd.DataFrame:
    df = df.copy()
    q_inf = config.q_inf

    df["Cd"] = df[config.col_drag] / (q_inf * df[config.col_area])

    if config.col_lift in df.columns:
        df["Cl"] = df[config.col_lift] / (q_inf * df[config.col_area])

    if config.col_moment in df.columns and config.col_chord and config.col_chord in df.columns:
        # Classic definition: Cm = M / (q_inf * A_ref * c)
        df["Cm"] = df[config.col_moment] / (q_inf * df[config.col_area] * df[config.col_chord])
    elif config.col_moment in df.columns:
        # Fallback: normalize by A_ref only (not ideal but better than nothing)
        df["Cm"] = df[config.col_moment] / (q_inf * df[config.col_area])
        logging.warning("Chord column not provided; Cm normalized without chord length.")

    # L/D where possible
    if "Cl" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            df["L_over_D"] = df["Cl"] / df["Cd"]

    return df


# -----------------------------------------
# Clean Data + Remove Outliers
# -----------------------------------------
def clean_data(df: pd.DataFrame, config: CFDConfig) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Outlier removal on Cd; extend to others if needed
    if "Cd" in df.columns and df["Cd"].std() > 0:
        z = np.abs((df["Cd"] - df["Cd"].mean()) / df["Cd"].std())
        df = df[z < config.zscore_threshold]

    return df


# -----------------------------------------
# Case Label from File / Folder
# -----------------------------------------
def infer_case_label(path: Path) -> str:
    """Infer a human-readable case label from file path.

        If the parent folder name is generic (e.g. 'data'), only the file stem is used.
        Otherwise 'parent | stem' is returned.
        """
    stem = path.stem
    parent = path.parent.name

    # If parent folder is generic like "data" or "exports", ignore it
    if parent.lower() in {"data", "exports", "csv", "results"}:
        return stem
    return f"{parent} | {stem}"


# -----------------------------------------
# Aggregate by AoA
# -----------------------------------------
def aggregate_by_aoa(df: pd.DataFrame, config: CFDConfig) -> pd.DataFrame:
    agg_cols = {"Cd": "mean"}
    if "Cl" in df.columns:
        agg_cols["Cl"] = "mean"
    if "Cm" in df.columns:
        agg_cols["Cm"] = "mean"
    if "L_over_D" in df.columns:
        agg_cols["L_over_D"] = "mean"

    grouped = df.groupby(config.col_aoa).agg(agg_cols).reset_index()
    return grouped.sort_values(by=config.col_aoa)


# -----------------------------------------
# Plotting Utilities
# -----------------------------------------
def plot_single_case(df: pd.DataFrame, case_label: str, config: CFDConfig, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cd vs AoA
    plt.figure(figsize=(8, 5))
    plt.plot(df[config.col_aoa], df["Cd"], marker="o", linewidth=2)
    plt.title(f"Cd vs AoA — {case_label}")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Cd")
    plt.grid(True, alpha=0.4)
    plt.minorticks_on()
    cd_path = out_dir / f"{case_label}_Cd_vs_AoA.png"
    plt.savefig(cd_path, dpi=300, bbox_inches="tight")
    plt.close()

    if "Cl" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df[config.col_aoa], df["Cl"], marker="o", linewidth=2)
        plt.title(f"Cl vs AoA — {case_label}")
        plt.xlabel("Angle of Attack [deg]")
        plt.ylabel("Cl")
        plt.grid(True, alpha=0.4)
        plt.minorticks_on()
        cl_path = out_dir / f"{case_label}_Cl_vs_AoA.png"
        plt.savefig(cl_path, dpi=300, bbox_inches="tight")
        plt.close()

    if "L_over_D" in df.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(df[config.col_aoa], df["L_over_D"], marker="o", linewidth=2)
        plt.title(f"L/D vs AoA — {case_label}")
        plt.xlabel("Angle of Attack [deg]")
        plt.ylabel("L/D")
        plt.grid(True, alpha=0.4)
        plt.minorticks_on()
        ld_path = out_dir / f"{case_label}_LoverD_vs_AoA.png"
        plt.savefig(ld_path, dpi=300, bbox_inches="tight")
        plt.close()



def plot_multi_case_comparison(all_results: Dict[str, pd.DataFrame],
                               config: CFDConfig,
                               out_dir: Path):
    """
    Overlay multiple cases on:
    - Cd vs AoA
    - Cl vs AoA (if available)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cd vs AoA (multi-case)
    plt.figure(figsize=(9, 6))
    for case_label, df in all_results.items():
        if "Cd" not in df.columns:
            continue
        plt.plot(df[config.col_aoa], df["Cd"], marker="o", linewidth=2, label=case_label)
    plt.title("Comparison: Cd vs AoA")
    plt.xlabel("Angle of Attack [deg]")
    plt.ylabel("Cd")
    plt.grid(True, alpha=0.4)
    plt.minorticks_on()
    plt.legend()
    cd_cmp = out_dir / "comparison_Cd_vs_AoA.png"
    plt.savefig(cd_cmp, dpi=300, bbox_inches="tight")
    plt.close()

    # Cl vs AoA (multi-case)
    has_cl = any("Cl" in df.columns for df in all_results.values())
    if has_cl:
        plt.figure(figsize=(9, 6))
        for case_label, df in all_results.items():
            if "Cl" not in df.columns:
                continue
            plt.plot(df[config.col_aoa], df["Cl"], marker="o", linewidth=2, label=case_label)
        plt.title("Comparison: Cl vs AoA")
        plt.xlabel("Angle of Attack [deg]")
        plt.ylabel("Cl")
        plt.grid(True, alpha=0.4)
        plt.minorticks_on()
        plt.legend()
        cl_cmp = out_dir / "comparison_Cl_vs_AoA.png"
        plt.savefig(cl_cmp, dpi=300, bbox_inches="tight")
        plt.close()

    logging.info(f"Saved multi-case comparison plots → {out_dir}")


# -----------------------------------------
# PDF Report Generation
# -----------------------------------------
def generate_pdf_report(all_results: Dict[str, pd.DataFrame],
                        config: CFDConfig,
                        out_dir: Path,
                        filename: str = "aero_report.pdf"):
    """
    Generate a multi-page PDF with standard plots for all cases.
    NOTE: This regenerates plots inside the PDF; PNGs are separate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / filename

    with PdfPages(pdf_path) as pdf:
        # One page per case: Cd vs AoA + Cl vs AoA + L/D vs AoA if available
        for case_label, df in all_results.items():
            # Cd vs AoA
            plt.figure(figsize=(8, 5))
            plt.plot(df[config.col_aoa], df["Cd"], marker="o", linewidth=2)
            plt.title(f"Cd vs AoA — {case_label}")
            plt.xlabel("Angle of Attack [deg]")
            plt.ylabel("Cd")
            plt.grid(True, alpha=0.4)
            plt.minorticks_on()
            pdf.savefig()
            plt.close()

            # Cl vs AoA
            if "Cl" in df.columns:
                plt.figure(figsize=(8, 5))
                plt.plot(df[config.col_aoa], df["Cl"], marker="o", linewidth=2)
                plt.title(f"Cl vs AoA — {case_label}")
                plt.xlabel("Angle of Attack [deg]")
                plt.ylabel("Cl")
                plt.grid(True, alpha=0.4)
                plt.minorticks_on()
                pdf.savefig()
                plt.close()

            # L/D vs AoA
            if "L_over_D" in df.columns:
                plt.figure(figsize=(8, 5))
                plt.plot(df[config.col_aoa], df["L_over_D"], marker="o", linewidth=2)
                plt.title(f"L/D vs AoA — {case_label}")
                plt.xlabel("Angle of Attack [deg]")
                plt.ylabel("L/D")
                plt.grid(True, alpha=0.4)
                plt.minorticks_on()
                pdf.savefig()
                plt.close()

        # Multi-case comparison pages
        # Cd comparison
        plt.figure(figsize=(9, 6))
        for case_label, df in all_results.items():
            if "Cd" not in df.columns:
                continue
            plt.plot(df[config.col_aoa], df["Cd"], marker="o", linewidth=2, label=case_label)
        plt.title("Comparison: Cd vs AoA (All Cases)")
        plt.xlabel("Angle of Attack [deg]")
        plt.ylabel("Cd")
        plt.grid(True, alpha=0.4)
        plt.minorticks_on()
        plt.legend()
        pdf.savefig()
        plt.close()

        # Cl comparison, if available
        has_cl = any("Cl" in df.columns for df in all_results.values())
        if has_cl:
            plt.figure(figsize=(9, 6))
            for case_label, df in all_results.items():
                if "Cl" not in df.columns:
                    continue
                plt.plot(df[config.col_aoa], df["Cl"], marker="o", linewidth=2, label=case_label)
            plt.title("Comparison: Cl vs AoA (All Cases)")
            plt.xlabel("Angle of Attack [deg]")
            plt.ylabel("Cl")
            plt.grid(True, alpha=0.4)
            plt.minorticks_on()
            plt.legend()
            pdf.savefig()
            plt.close()

    logging.info(f"PDF report generated → {pdf_path}")


# -----------------------------------------
# Export Aggregated Results
# -----------------------------------------
def export_results(all_results: Dict[str, pd.DataFrame],
                   out_dir: Path,
                   base_name: str = "aggregated_results"):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combine into a single DataFrame with case label
    combined = []
    for case_label, df in all_results.items():
        df_copy = df.copy()
        df_copy["Case"] = case_label
        combined.append(df_copy)

    if not combined:
        logging.warning("No results to export.")
        return

    combined_df = pd.concat(combined, ignore_index=True)

    csv_path = out_dir / f"{base_name}.csv"
    xlsx_path = out_dir / f"{base_name}.xlsx"

    combined_df.to_csv(csv_path, index=False)
    combined_df.to_excel(xlsx_path, index=False)

    logging.info(f"Exported aggregated results → {csv_path}, {xlsx_path}")


# -----------------------------------------
# Full Pipeline for One File
# -----------------------------------------
def process_single_file(path: Path, config: CFDConfig, out_root: Path) -> Optional[pd.DataFrame]:
    logging.info(f"Processing {path.name} ...")
    df = load_starccm_csv(path, config)
    if df is None:
        return None

    # Detect steady-state
    steady_index = detect_steady_state(
        df[config.col_drag],
        window=config.steady_window,
        threshold=config.steady_threshold
    )
    df = df.iloc[steady_index:]

    # Compute coefficients
    df = compute_coefficients(df, config)

    # Clean data
    df = clean_data(df, config)

    if df.empty:
        logging.warning(f"{path.name}: no valid data after cleaning.")
        return None

    # Aggregate by AoA
    agg = aggregate_by_aoa(df, config)

    # Case label and output directory
    case_label = infer_case_label(path)

    # FULL SANITIZATION for Windows filenames
    safe_label = (
        case_label
        .replace(" ", "_")
        .replace("|", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace("*", "_")
        .replace("?", "_")
        .replace("<", "_")
        .replace(">", "_")
    )

    case_out_dir = out_root / safe_label

    # IMPORTANT: use SAFE label everywhere
    plot_single_case(agg, safe_label, config, case_out_dir)

    return agg


# -----------------------------------------
# Batch Processing for a Directory
# -----------------------------------------
def process_folder(folder_path: str, config: Optional[CFDConfig] = None) -> Dict[str, pd.DataFrame]:
    folder = Path(folder_path)
    if config is None:
        cfg_path = folder / "config.yml"
        config = load_config_from_yaml(cfg_path)

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        logging.error(f"No CSV files found in: {folder_path}")
        return {}

    # Output root
    out_root = config.output_dir or (folder / "post_processed")
    out_root.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, pd.DataFrame] = {}

    for file in csv_files:
        agg = process_single_file(file, config, out_root)
        if agg is not None:
            raw_label = infer_case_label(file)

            safe_label = (
                raw_label
                .replace(" ", "_")
                .replace("|", "_")
                .replace("/", "_")
                .replace("\\", "_")
                .replace(":", "_")
                .replace("*", "_")
                .replace("?", "_")
                .replace("<", "_")
                .replace(">", "_")
            )

            all_results[safe_label] = agg

    if not all_results:
        logging.error("No valid CFD results processed.")
        return {}

    # Comparison plots + exports + report
    comparison_dir = out_root / "comparisons"
    plot_multi_case_comparison(all_results, config, comparison_dir)

    export_results(all_results, out_root)
    generate_pdf_report(all_results, config, out_root)

    return all_results


# -----------------------------------------
# Optional: Streamlit GUI Skeleton
# -----------------------------------------
def launch_streamlit_app():
    """
    Minimal Streamlit GUI wrapper for interactive use.
    Run with:
        streamlit run aero_postprocessing.py

    NOTE: Only a skeleton; expand as needed.
    """
    try:
        import streamlit as st
    except ImportError:
        logging.error("streamlit not installed. Install with `pip install streamlit`.")
        return

    st.set_page_config(layout="wide", page_title="Aerodynamics Post-Processing")

    st.title("Aerodynamics Post-Processing — STAR-CCM+ → Python")

    folder = st.text_input("Folder containing STAR-CCM+ CSV exports", "./cfd_exports")

    if st.button("Run Post-Processing"):
        with st.spinner("Processing CFD data..."):
            results = process_folder(folder)
        if not results:
            st.error("No valid results found.")
        else:
            st.success("Post-processing complete.")

            # Show first result as a table and plot
            first_case, first_df = next(iter(results.items()))
            st.subheader(f"Example Case: {first_case}")
            st.dataframe(first_df)

            if "Cd" in first_df.columns:
                st.line_chart(first_df.set_index(first_df.columns[0])["Cd"], height=300)
            if "Cl" in first_df.columns:
                st.line_chart(first_df.set_index(first_df.columns[0])["Cl"], height=300)


# -----------------------------------------
# CLI Entry Point
# -----------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated STAR-CCM+ Aerodynamics Post-Processing"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default="./cfd_exports",
        help="Folder containing STAR-CCM+ CSV exports"
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch Streamlit GUI instead of CLI processing"
    )

    args = parser.parse_args()

    if args.gui:
        launch_streamlit_app()
    else:
        process_folder(args.folder)
