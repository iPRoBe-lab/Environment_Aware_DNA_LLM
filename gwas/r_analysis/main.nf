#!/usr/bin/env nextflow

// GWAS pipeline (local). Runs GAPIT for each (phenotype, model) job listed in a
// single manifest. One job = one row.
//
//   manifest columns:  geno_csv , map_csv , pheno_csv , output_dir , model
//
// Usage:
//   nextflow run main.nf                 # run every job in the manifest
//   nextflow run main.nf --model BLINK   # only BLINK jobs
//   nextflow run main.nf --model MLM     # only MLM jobs
//   nextflow run main.nf --jobs path.csv # use a different manifest

include { gapit_gwas } from './modules/gapit_gwas.nf'

params.jobs  = "${projectDir}/gwas_jobs.csv"
params.model = null   // optional filter: 'BLINK' | 'MLM' | null (= all)

workflow {
    Channel.fromPath(params.jobs)
           .splitCsv(sep: ',')
           .filter { row -> params.model == null || row[4] == params.model }
           .map    { row -> tuple(file(row[0]), file(row[1]), file(row[2]), row[3], row[4]) }
           .set    { jobs_ch }

    gapit_gwas(jobs_ch)
}
