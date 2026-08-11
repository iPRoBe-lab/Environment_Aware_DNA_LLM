import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, euclidean
from collections import OrderedDict

def calculate_embedding_shift(embeddings_control, embeddings_stress, distance_metric='cosine'):
    """
    Calculates the "shift" between two batches of embeddings.
    Args:
        embeddings_control (np.array): The (N, D) batch of embeddings with no stress prompt (control condition).
        embeddings_stress (np.array): The (N, D) batch of embeddings with stress prompt.
        distance_metric (str): 'cosine' (recommended) or 'euclidean'.
    Returns:
        np.array: A (N,) array of shift scores. A higher score means a larger shift.
    """
    if embeddings_control.shape != embeddings_stress.shape:
        raise ValueError("Embedding batches must have the same shape.")
    N = embeddings_control.shape[0]
    shift_scores = np.zeros(N)
    if distance_metric == 'cosine':
        # Using a list comprehension for efficiency over paired rows
        # scipy.spatial.distance.cosine(u, v) returns 1 - (u.v / ||u||||v||)
        shift_scores = np.array([cosine(embeddings_control[i], embeddings_stress[i]) for i in range(N)])
    elif distance_metric == 'euclidean':
        shift_scores = np.array([euclidean(embeddings_control[i], embeddings_stress[i]) for i in range(N)])

    else:
        raise ValueError("Unknown distance_metric. Use 'cosine' or 'euclidean'.")
    return shift_scores

def calculate_gini(p):
    """Calculates Gini coefficient for attention weights p."""
    p_sorted = np.sort(p)
    n = len(p_sorted)
    index = np.arange(1, n + 1)
    # Standard Gini formula for sorted non-negative values
    gini = (np.sum((2 * index - n - 1) * p_sorted)) / (n * np.sum(p_sorted))
    return gini

def dna_llm_prioritization(df: pd.DataFrame, method: str = 'rank_product') -> pd.Series:
    """
    Prioritize sequences based on DNA language model metrics.
    Higher score = higher priority (more likely functional)
    """
    shift = df["Embedding_Shift_Cosine"].rank(pct=True)
    gini = df["Attn_Gini_Coeff"].rank(pct=True)
    entropy_conc = (1 - df["Attn_Entropy_Norm"]).rank(pct=True)  # low entropy = high concentration
    eps = 1e-12
    if method == "shift_only":
        priority_score = shift
    elif method == "entropy_only":
        priority_score = entropy_conc
    elif method == "gini_only":
        priority_score = gini
    elif method == "shift_entropy":
        priority_score = ((shift + eps) * (entropy_conc + eps)) ** 0.5
    elif method == "shift_gini":
        priority_score = ((shift + eps) * (gini + eps)) ** 0.5
    else:
        raise ValueError(f"Unknown method: {method}.")
    return priority_score

def extract_genes(annotated_df, column="Gene_ID"):
    """Extract and flatten all gene IDs, split by '-' (for intergenic variants).
    Returns a set of unique gene IDs.
    """
    gene_list = (
        annotated_df[column]
        .dropna()  # remove NaN entries
        .astype(str)
        .str.split('-')  # split on '-'
        .explode()
        .tolist()  # flatten list of lists
    )
    unique_genes = list(OrderedDict.fromkeys(gene_list))
    # Return list
    return unique_genes

def main(args):
    stress_category = ['HEAT_STRESSED', 'DROUGHT_STRESSED', 'HEAT_DROUGHT_STRESSED']
    args.embeddings_dir = os.path.join(args.embeddings_dir, args.peft_version)
    args.embeddings_dir_control = os.path.join(args.embeddings_dir_control, args.peft_version)
    os.makedirs(f"{args.embeddings_dir}/dna_sequence_scoring", exist_ok=True)
    os.makedirs(f"{args.embeddings_dir}/prioritized_genes", exist_ok=True)
    os.makedirs(f"{args.embeddings_dir}/prioritized_genes_gwas", exist_ok=True)
    for stress in stress_category:
        print(f"Processing {stress} category...")
        stress_file_name = f"{args.phenotype}_{stress}"
        stress_df = pd.read_csv(os.path.join(args.embeddings_dir,"dna_sequences", f"{stress_file_name}.csv"))[["SNP","P.value", "Q.value"]].rename(columns={"P.value": "gwas_p_value", "Q.value": "gwas_q_value"})
        stress_emb = np.load(os.path.join(args.embeddings_dir,"dna_sequence_embeddings", f"{stress_file_name}.npy"))
        stress_emb_control = np.load(os.path.join(args.embeddings_dir_control,"dna_sequence_embeddings", f"{stress_file_name}.npy"))
        stress_emb_attn = np.load(os.path.join(args.embeddings_dir,"attention_scores", f"{stress_file_name}.npz"), allow_pickle=True)["attn_last_from_stress"]
        
        shift_scores_cosine = calculate_embedding_shift(stress_emb_control, stress_emb, distance_metric='cosine')     
        shift_scores_euclidean = calculate_embedding_shift(stress_emb_control, stress_emb, distance_metric='euclidean')               
        attn_gini_coeff = []
        attn_entropy_norm = []
        # works for variable length sequences
        for seq in stress_emb_attn:
            # Convert to numpy array
            seq = np.asarray(seq, dtype=float)
            # Select DNA tokens only (skip CLS + stress tokens)
            dna_seq = seq[2:]     
            n = len(dna_seq)
            # Normalize
            p = dna_seq / (dna_seq.sum() + 1e-12)
            # Entropy
            H = -np.sum(p * np.log2(p + 1e-12))
            h_norm = H / np.log2(n)
            attn_entropy_norm.append(h_norm)
            # Gini coefficient
            G = calculate_gini(p)
            attn_gini_coeff.append(G)
        attn_gini_coeff = np.array(attn_gini_coeff)
        attn_entropy_norm = np.array(attn_entropy_norm)
        stress_df["Embedding_Shift_Cosine"] = shift_scores_cosine
        stress_df["Embedding_Shift_Euclidean"] = shift_scores_euclidean
        stress_df["Attn_Entropy_Norm"] = attn_entropy_norm
        stress_df["Attn_Gini_Coeff"] = attn_gini_coeff
        for method in ["shift_only", "shift_gini", "shift_entropy", "gini_only", "entropy_only"]:
            column_name = f"Score_{method.upper()}"
            stress_df[column_name] = dna_llm_prioritization(stress_df, method=method)
        # Keep all rows from stress_df (left join)
        variant_annotations = pd.read_csv(args.variant_annotations, sep='\t')
        merged_df = stress_df.merge(
            variant_annotations,
            left_on="SNP",
            right_on="ID",
            how='left',
            suffixes=('', '_snpeff')
        )
        merged_df.to_csv(os.path.join(args.embeddings_dir,"dna_sequence_scoring", f"{stress_file_name}.csv"), index=False)
        for method in ["shift_only", "shift_gini", "shift_entropy", "gini_only", "entropy_only"]:
            column_name = f"Score_{method.upper()}"
            cutoff = merged_df[column_name].quantile(1-args.top_pct)
            priority_genes = extract_genes(merged_df[merged_df[column_name]>=cutoff].sort_values(by=column_name,ascending=False))
            out_file = os.path.join(args.embeddings_dir,"prioritized_genes", f"{stress_file_name}_{method}_top{int(args.top_pct*100)}pct_genes.txt")
            with open(out_file, "w") as f:
                for gene in priority_genes:
                    f.write(f"{gene}\n")
        gwas_cutoff = merged_df["gwas_p_value"].quantile(args.top_pct)

        gwas_priority_genes = extract_genes(
            merged_df[merged_df["gwas_p_value"] <= gwas_cutoff].sort_values(by="gwas_p_value", ascending=True))

        gwas_out_file = os.path.join(
            args.embeddings_dir,
            "prioritized_genes_gwas",
            f"{stress_file_name}_pvalue_top{int(args.top_pct*100)}pct_genes.txt"
        )

        with open(gwas_out_file, "w") as f:
            for gene in gwas_priority_genes:
                f.write(f"{gene}\n")

if __name__== '__main__':
    os.environ["OMP_NUM_THREADS"] = "16"          # OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "16"          # MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = "16"
    os.environ["OPENBLAS_NUM_THREADS"] = "16"

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description="Sequence scoring - embedding shift from control, attention-based gini coeffcient, normalized attention entropy")
    parser.add_argument('--embeddings_dir', type=str, default='stress_aware_embeddings_v2', help='folder to read embeddings')
    parser.add_argument('--embeddings_dir_control', type=str, default='counterfactual_embeddings_v2', help='folder to read embeddings with no stress prompt')
    parser.add_argument('--peft_version', type=str, default='StressAwareAgroNT_v2/RefSeq_p05/prompt_tuning_rslora_v1', help='peft model version')
    parser.add_argument('--phenotype', type=str, default='yield', choices = ['yield', 'anthesis', 'silking', 'ASI'], help='Phenotype to consider')
    parser.add_argument("--variant_annotations", default="../gwas/gwas_all_snpeff_annotation_16kb.tsv", help="snpeff variant annotations")
    parser.add_argument("--top_pct", type=float, default=0.10, help="Percentage to select prioritized SNPs")
    args = parser.parse_args()
    main(args)