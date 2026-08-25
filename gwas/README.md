# Maize stress GWAS — analysis reports

Methods documentation and analysis for stress-conditioned association mapping in a maize hybrid panel
(Genomes-to-Fields). Each report below is a self-contained, rendered notebook (open in any browser).

## Reports (`jupyter_notebook_reports/`)

| Report | What it covers |
|---|---|
| [blue_preprocessing.html](jupyter_notebook_reports/blue_preprocessing.html) | Environment-adjusted phenotypes: fixed-effects BLUEs vs. naïve per-environment averages, per trait × stress category. |
| [ld_decay.html](jupyter_notebook_reports/ld_decay.html) | LD-decay on the full unpruned panel (Hill–Weir fit) → the ±16 kb candidate-locus window; per-run candidate loci and positional concordance. |
| [avg_vs_blue_comparison.html](jupyter_notebook_reports/avg_vs_blue_comparison.html) | Head-to-head of averaged- vs BLUE-phenotype BLINK results: calibration (λ, QQ), hit counts, per-SNP agreement. |
| [liftover.html](jupyter_notebook_reports/liftover.html) | Coordinate liftover (B73 v4 → v5) of published stress meta-QTLs (CrossMap) and their overlap with prioritized loci. |
| [software_versions.html](jupyter_notebook_reports/software_versions.html) | Software and versions used across the analysis. |

## Code

- [`r_analysis/`](r_analysis/) — Nextflow + GAPIT/BLINK pipeline that produced the association results
  (16 trait × stress-category jobs), including [`compute_blue.R`](r_analysis/compute_blue.R) which
  computes the fixed-effects BLUEs used as the GWAS phenotypes. See
  [`r_analysis/README.md`](r_analysis/README.md) for inputs and usage.

## Software

R 4.5.3 · GAPIT 4.1.0 (BLINK) · PLINK v1.9.0-b.8 · Python 3.10.14 (SciPy 1.13.1 / pandas 2.2.3) ·
CrossMap 0.7.0 · Nextflow 25.10.4. Reference genome: Zm-B73-REFERENCE-NAM-5.0.

## Data availability

Genotypes and phenotypes are from the curated Genomes-to-Fields dataset (Lopez-Cruz et al.); the raw
data are not redistributed here — see that source. Liftover chain files are from MaizeGDB
(`download.maizegdb.org/Zm-B73-REFERENCE-NAM-5.0/chain_files/`).
