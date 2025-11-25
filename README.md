Aerodynamics Post-Processing Toolkit (STAR-CCM+ → Python)

A production-grade Python toolkit for automating aerodynamic post-processing from STAR-CCM+ CSV exports.
Replaces manual Excel workflows with a fast, repeatable pipeline for R&D, aero development, and academic CFD studies.

🔥 Key Features

1.Load and validate STAR-CCM+ CSV files

2.Automatic steady-state detection (rolling standard deviation)

3.Compute aerodynamic coefficients: Cd, Cl, Cm, L/D

4.Outlier removal using z-scores

5.Aggregate forces by Angle of Attack (AoA)

6.Batch-process entire folders of CFD runs

7.Generate:

   a.Per-case plots

   b.Multi-case comparison plots

   c.Multi-page PDF reports

   d.Combined CSV + Excel outputs

8.Configurable via config.yml

9.CLI and optional Streamlit GUI


📁 Repository Structure

  src/
    config.py        – CFD configuration + YAML loader
    loader.py        – CSV loading and validation
    utils.py         – Case labeling + filename sanitization
    processing.py    – Full CFD pipeline (steady-state, coefficients, cleaning)
    plots.py         – Single-case and multi-case plotting
    report.py        – PDF report generation
    cli.py           – Command-line interface
    app_streamlit.py – Optional Streamlit GUI

examples/
    sample_starccm_export.csv
    example_plots/
    sample_report.pdf

🚀 Quick Start
1. Install dependencies
   pip install -r requirements.txt
2. Run the full pipeline (CLI)
   python -m src.cli ./cfd_exports
3. Launch the GUI
   python -m src.cli ./cfd_exports --gui


📊 Example Outputs

Cd vs AoA

Cl vs AoA

Multi-Case Comparison


⚙️ Requirements

See requirements.txt.
Uses:

1. numpy

2. pandas

3. matplotlib

4. PyYAML

5. openpyxl

6. streamlit (optional)
7. 📄 Sample Dataset

A small synthetic STAR-CCM+-style CSV file is included in examples/.
