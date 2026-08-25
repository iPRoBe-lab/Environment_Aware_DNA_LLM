#!/usr/bin/env nextflow

// One GAPIT GWAS job: one phenotype x one model (BLINK or MLM).
// Runs on the local machine using the bioinf_agent gapit env (params.rscript).
// BLAS threads are pinned to 1 in nextflow.config so each job uses one core.

process gapit_gwas {
    tag { "${pheno.simpleName}_${model}" }
    publishDir { output_dir }, mode: 'copy'

    input:
    tuple(path(geno), path(map), path(pheno), val(output_dir), val(model))

    output:
    path "GAPIT.*"

    script:
    """
    ${params.rscript} ${params.gapit_script} \
        ${geno} ${map} ${pheno} ${model} ${params.pca_total}
    """
}
