# Environment-Aware DNA Language Model for Stress-Responsive Genomic Prioritization in Maize
Adapting AgroNT, a genomic foundation model, for plants stress response analysis.
### Abstract:
Abiotic stresses such as heat and drought severely reduce maize productivity yet identifying genomic regions that confer stress resilience remains challenging. Inspired by Large Language Models (LLMs), Genomic Foundation Models (GFMs) leverage large-scale pre-training on DNA sequences to capture regulatory patterns, however, their application to plant stress-response analysis remains unexplored. We present an environment-aware DNA-LLM that adapts AgroNT, a transformer-based GFM pre-trained on diverse plant genomes, by incorporating stress-specific prompt tokens. Through parameter-efficient fine-tuning, the model learns stress-conditioned sequence representations that form distinct structures according to environmental context in the embedding space. By combining stress-induced shifts in DNA-sequence embeddings with transformer attention patterns, we prioritized putative heat- and drought-responsive genomic regions associated with grain yield in the Genomes-to-Fields (G2F) panel. Prioritized regions were supported by the overlap with spatiotemporal differential gene-expression profiles and stress-associated quantitative trait loci, as well as transcription-factor family characterization and regulatory motif enrichment.  Attention-guided motif analysis further identified stress-associated motifs enriched within model-emphasized sequence regions. Overall, the prioritized loci were proximal to genes involved in transcriptional regulation, signaling, and metabolic pathways relevant to abiotic-stress adaptation, demonstrating the potential of stress-conditioned transformer-based sequence modeling for environment-aware genome-to-phenome analysis.

## Code 

The `Code/` directory contains scripts for constructing the stress-aware AgroNT model, training the model, extracting sequence representations, and computing the sequence-level prioritization scores used in the study.

* **`stress_aware_agront.py`**
  Implements the stress-aware AgroNT model by incorporating learned environment-specific prompt tokens into the input sequence. It supports prompt tuning (PT) and its combinations with LoRA/DoRA for parameter-efficient fine-tuning, and defines the stress-conditioned representation learning framework used during training.

* **`training_script.sh`**
  Shell script for launching stress-aware AgroNT training experiments. It specifies the training configuration and commands used to run the different parameter-efficient fine-tuning strategies.

* **`pretrained_embeddings.py`**
  Extracts sequence embeddings from the original pretrained AgroNT model prior to stress-aware fine-tuning. These embeddings provide the pretrained baseline for evaluating changes in sequence representations induced by environmental conditioning.

* **`stress_aware_embeddings.py`**
  Extracts stress-conditioned sequence representations from the fine-tuned models. DNA-token embeddings generated under the different environmental contexts are used for downstream representation analysis and locus prioritization.

* **`counterfactual_embeddings.py`**
  Generates counterfactual representations of the same DNA sequence under alternative environmental contexts. These representations enable measurement of the stress-induced embedding shift by comparing a sequence representation under a target stress condition with its corresponding control representation.

* **`sequence_scoring.py`**
  Computes sequence-level prioritization scores from the stress-aware model outputs. The script quantifies stress-induced embedding shifts and attention concentration and combines their percentile ranks to calculate the **SHIFT_GINI** score used to rank candidate genomic loci.
