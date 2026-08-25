# GWAS pipeline (Nextflow + GAPIT/BLINK)

Runs one GAPIT association analysis per job listed in `gwas_jobs.csv` (one trait × stress category ×
model per row). Used to produce the BLINK results for the 16 trait × stress-category combinations.

## Files
- `compute_blue.R` — computes the fixed-effects BLUEs used as the GWAS phenotypes (base R `lm`).
- `main.nf` — workflow: reads the manifest, one process per job.
- `modules/gapit_gwas.nf` — the GAPIT process.
- `gapit_gwas.R` — the R entrypoint: `GAPIT(Y, GD, GM, PCA.total=5, model=<BLINK|MLM>)`.
- `nextflow.config` — resources + single-thread BLAS pinning.
- `gwas_jobs.csv` — job manifest (no header): `geno_csv,map_csv,pheno_csv,output_dir,model`.

## Run
```bash
nextflow run main.nf --model BLINK          # the published analysis
nextflow run main.nf --rscript /path/to/Rscript   # if GAPIT R is not `Rscript` on PATH
```

## Inputs (not redistributed here — supply under the paths in the manifest)
- `data/GENO.csv` — numeric genotype matrix (first column = Taxa). From the curated G2F genotypes
  (Lopez-Cruz et al.); too large to host here.
- `data/snpnames.csv` — SNP map: `chromosome,position,name`.
- `phenotypes/blue_pheno/<trait>_<category>.csv` — per-hybrid BLUEs produced by `compute_blue.R`.

Outputs are written to `results/<trait>_<category>_BLINK/` (GAPIT `GAPIT.*` files).

## Requirements
Nextflow 25.10.4 (JDK 17) · R 4.5.3 with GAPIT 4.1.0 installed.

> The manifest lists BLINK jobs (the published analysis). The pipeline also supports MLM via the
> `model` column / `--model MLM`; those jobs are not included here.
