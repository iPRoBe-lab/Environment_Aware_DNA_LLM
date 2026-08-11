"""Inference script for stress-aware AgroNT.
Loads trained stress embeddings (and optionally LoRA/DoRA adapters) to extract DNA sequence embeddings, stress token embeddings, and attention scores.
Usage:
    Prompt tuning only:
        python3 stress_aware_embeddings.py --version <peft_model_version>
    With LoRA/DoRA/rdLoRA adapters:
        python3 stress_aware_embeddings.py --version <peft_model_version> --use_lora
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from genomic_utils import read_fasta, sig_snps_to_seq

def load_model_tokenizer(peft_model_path, model_name='agro-nucleotide-transformer-1b', use_lora=False):
    """Load base AgroNT model with trained stress embeddings and optional LoRA/DoRA adapters.
    Args:
        peft_model_path: Path to directory containing stress_embeddings_final.pth, tokenizer files, and optionally lora_adapters
        model_name: HuggingFace model identifier
        use_lora: Whether to load LoRA/DoRA adapters from peft_model_path/lora_adapters/
    Returns:
        model: AgroNT model (with LoRA if enabled)
        tokenizer: Tokenizer with stress tokens added
        stress_embeddings: Trained stress token embeddings tensor
    """
    config = AutoConfig.from_pretrained(f'InstaDeepAI/{model_name}', output_attentions=True)
    print(f"No. of Attention Heads: {config.num_attention_heads}")
    config.attn_implementation = "eager"
    
    model = AutoModelForMaskedLM.from_pretrained(
        f'InstaDeepAI/{model_name}', cache_dir="./pretrained_model", config=config
    )
    # Load original tokenizer for bias resizing
    orig_tokenizer = AutoTokenizer.from_pretrained(f'InstaDeepAI/{model_name}', cache_dir="./pretrained_model")
    # Load updated tokenizer saved during training stress_aware_agroNT
    tokenizer = AutoTokenizer.from_pretrained(peft_model_path)
    # Load trained stress embeddings
    stress_embeddings = torch.load(f"{peft_model_path}/stress_embeddings_final.pth", weights_only=True)
    # Resize model embeddings to match tokenizer with stress tokens
    model.resize_token_embeddings(len(tokenizer))
    tokenizer_vocab_size = len(tokenizer)
    # Manually resize LM head bias (same logic as done suring training)
    with torch.no_grad():
        lm_head = model.lm_head
        if hasattr(lm_head, 'bias') and lm_head.bias is not None:
            lm_head_bias = lm_head.bias
            if lm_head_bias.size(0) != tokenizer_vocab_size:
                new_bias = torch.zeros(tokenizer_vocab_size, dtype=lm_head_bias.dtype, device=lm_head_bias.device)
                new_bias[:len(orig_tokenizer)-2] = lm_head_bias
                model.lm_head.bias = torch.nn.Parameter(new_bias)
    # Load LoRA/DoRA adapters if required
    if use_lora:
        lora_adapter_path = os.path.join(peft_model_path, 'lora_adapters')
        if not os.path.exists(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found at {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        model = model.merge_and_unload()  # Merge LoRA weights into base model for faster inference
        print(f"Loaded and merged LoRA/DoRA adapters from {lora_adapter_path}")
    return model, tokenizer, stress_embeddings


def extract_embeddings(sequences, raw_sequences, model, tokenizer, stress_embeddings, device, batch_size):
    """Extract tokenized sequences, sequence embeddings, stress embeddings, and attention scores.
    Args:
        sequences: List of prompted DNA sequences (stress token prepended)
        raw_sequences: List of raw DNA sequences (without stress token)
        model: AgroNT model on device
        tokenizer: Tokenizer with stress tokens
        stress_embeddings: Trained stress token embedding tensor
        device: CUDA or CPU device
        batch_size: Inference batch size
    Returns:
        seq_embeddings: Mean-pooled DNA sequence embeddings [N, hidden_dim]
        stress_embeddings_out: Stress token output embeddings [N, hidden_dim]
        avg_attn: Per-sample average attention received by each token
        attn_from_stress: Per-sample attention distribution from stress token
        attn_masks: Per-sample attention masks
        tokenized_data: List of dicts with tokenized sequences and positions
    """
    all_seq_embeddings = []
    all_stress_embeddings = []
    all_attn_received, all_attn_from_stress, all_attn_masks = [], [], []
    all_tokenized_data = []
    model = model.to(device).eval()
    # Get stress token IDs for embedding replacement
    stress_ids = [tokenizer.convert_tokens_to_ids(t) for t in tokenizer.additional_special_tokens]
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]     # prompted sequences for this batch, prepended with stress token
            batch_raw = raw_sequences[i:i+batch_size]  # raw sequences for this batch for storing purpose, without stress token
            
            batch_encodings = tokenizer(
                batch_seqs, padding="longest",
                truncation=True, max_length=1024, return_tensors='pt'
            )
            batch_tokens = batch_encodings['input_ids'].to(device)
            attention_mask = batch_encodings['attention_mask'].to(device)
            
            # Extract tokenized sequences
            for j in range(len(batch_seqs)):
                token_ids = batch_tokens[j].cpu().tolist()
                mask = attention_mask[j].cpu().numpy()
                valid_len = int(mask.sum())
                
                # Convert to token strings
                tokens = tokenizer.convert_ids_to_tokens(token_ids[:valid_len])
                
                # DNA tokens start at position 2 (after CLS and stress token)
                # Each 6-mer token covers 6bp with step=6 (non-overlapping)
                dna_tokens = tokens[2:]  # Skip CLS and stress token
                all_tokenized_data.append({
                    'tokens': dna_tokens,              # List of 6-mer strings
                    'n_tokens': len(dna_tokens),      # Number of DNA tokens
                    'raw_sequence': batch_raw[j],     # Original DNA sequence
                })
            
            # Replace stress token embeddings with trained embeddings
            inputs_embeds = model.get_input_embeddings()(batch_tokens)
            for idx, token_id in enumerate(stress_ids):
                mask = (batch_tokens == token_id)
                if mask.any():
                    inputs_embeds[mask] = stress_embeddings[idx].to(device)
            
            outs = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True
            )
            
            # Final layer attention and hidden states
            attn_last = outs.attentions[-1]             # [B, H, Q, K]
            batch_embeddings = outs.hidden_states[-1]   # [B, seq_len, hidden_dim]
            del outs
            torch.cuda.empty_cache()
            
            # Mean-pooled sequence embeddings (skip CLS at 0 and stress token at 1)
            sequence_embeddings = batch_embeddings[:, 2:, :]
            padding_mask = attention_mask[:, 2:].unsqueeze(-1)
            mean_pooled_embeds = (sequence_embeddings * padding_mask).sum(dim=1) / padding_mask.sum(dim=1)
            all_seq_embeddings.append(mean_pooled_embeds.cpu().numpy())
            
            # Stress token output embeddings (position 1)
            stress_embeds = batch_embeddings[:, 1, :]
            all_stress_embeddings.append(stress_embeds.cpu().numpy())
            
            # Attention analysis
            attn_received_avg = attn_last.mean(dim=(1, 2))          # Average attention received per token [B, T]
            attn_from_stress_prompt = attn_last[:, :, 1, :].mean(dim=1)  # Attention from stress token to all [B, T]
            all_attn_received.extend(attn_received_avg.cpu().numpy())
            all_attn_from_stress.extend(attn_from_stress_prompt.cpu().numpy())
            all_attn_masks.extend(attention_mask.cpu().numpy())
    
    return (np.vstack(all_seq_embeddings), np.vstack(all_stress_embeddings), 
            all_attn_received, all_attn_from_stress, all_attn_masks, all_tokenized_data)
    

def main(args):
    os.environ['HOME'] = './'
    device = torch.device(f"cuda:{args.gpu_device}" if args.cuda and torch.cuda.is_available() else "cpu")
    args.peft_model_path = os.path.join(args.peft_model_path, f"RefSeq_p{str(args.p_value).replace('0.', '')}", args.version)
    # Load model with trained stress embeddings and optional LoRA/DoRA
    model, tokenizer, learned_stress_embeds = load_model_tokenizer(
        peft_model_path=f"{args.peft_model_path}/peft_model",
        use_lora=args.use_lora
    )
    
    stress_category = ['NO_STRESS', 'HEAT_STRESSED', 'DROUGHT_STRESSED', 'HEAT_DROUGHT_STRESSED']
    stress_map = {
        'NO_STRESS': '<NO_STRESS>',
        'HEAT_STRESSED': '<HEAT_STRESS>',
        'DROUGHT_STRESSED': '<DROUGHT_STRESS>',
        'HEAT_DROUGHT_STRESSED': '<HEAT_DROUGHT_STRESS>'
    }
    
    args.output_folder = os.path.join(args.output_folder, args.peft_model_path)
    os.makedirs(f"{args.output_folder}/dna_sequences", exist_ok=True)
    
    print("GWAS Threshold: " + str(args.p_value))
    ref_genome = read_fasta(args.ref_genome_path)
    base_pad = int(args.sequence_length / 2)
    
    for stress in stress_category:
        print(f"Extract embeddings from {stress} category...")
        seq_df = sig_snps_to_seq(
            ref_genome, 
            gwasfile=os.path.join(args.gwas_results, f"{args.phenotype}_{stress}.csv"),
            base_pad=base_pad, 
            pval_cut=args.p_value
        )
        seq_df.to_csv(f"{args.output_folder}/dna_sequences/{args.phenotype}_{stress}.csv", index=False)
        sequences = seq_df["Sequence"].tolist()
        stress_token = stress_map[stress]
        prompted_sequences = [f"{stress_token} {seq}" for seq in sequences]
        
        seq_embeddings, stress_embeddings, avg_attn, attn_from_stress, attn_mask, tokenized_data = extract_embeddings(
            sequences=prompted_sequences,
            raw_sequences=sequences,  
            model=model, 
            tokenizer=tokenizer, 
            stress_embeddings=learned_stress_embeds,
            device=device, 
            batch_size=args.batch_size
        )
        
        # Save results
        os.makedirs(f"{args.output_folder}/dna_sequence_embeddings", exist_ok=True)
        os.makedirs(f"{args.output_folder}/stress_prompt_embeddings", exist_ok=True)
        os.makedirs(f"{args.output_folder}/attention_scores", exist_ok=True)
        os.makedirs(f"{args.output_folder}/tokenized_sequences", exist_ok=True) 

        np.save(f"{args.output_folder}/dna_sequence_embeddings/{args.phenotype}_{stress}.npy", seq_embeddings)
        np.save(f"{args.output_folder}/stress_prompt_embeddings/{args.phenotype}_{stress}.npy", stress_embeddings)
        np.savez(f"{args.output_folder}/attention_scores/{args.phenotype}_{stress}.npz", **{
            "avg_attn_last": np.array(avg_attn, dtype=object),
            "attn_last_from_stress": np.array(attn_from_stress, dtype=object),
            "attn_mask": np.array(attn_mask, dtype=object)
        })
        np.savez(
            f"{args.output_folder}/tokenized_sequences/{args.phenotype}_{stress}.npz",
            tokens=np.array([d['tokens'] for d in tokenized_data], dtype=object),
            n_tokens=np.array([d['n_tokens'] for d in tokenized_data]),
            raw_sequences=np.array([d['raw_sequence'] for d in tokenized_data], dtype=object)
        )

    
if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    os.environ["NUMEXPR_NUM_THREADS"] = "8"
    os.environ["OPENBLAS_NUM_THREADS"] = "8"
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    
    parser = argparse.ArgumentParser(description="Extract DNA embeddings using stress-aware AgroNT")
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="2", help="GPU devices to be used")
    parser.add_argument('--batch_size', type=int, default=15, help="Batch size")
    parser.add_argument('--output_folder', type=str, default='stress_aware_embeddings_v2', help='folder to save extracted sequences and embeddings')
    parser.add_argument('--peft_model_path', type=str, default='StressAwareAgroNT_v2', help='load peft model')
    parser.add_argument('--version', type=str, default='prompt_tuning_v1', help='peft version')
    parser.add_argument('--ref_genome_path', type=str, default='../ref_genome_fasta/Zm-B73-REFERENCE-NAM-5.0.fa', help='fasta file to read DNA sequences')
    parser.add_argument('--gwas_results', type=str, default='../gwas/gapit_blue_results')
    parser.add_argument('--phenotype', type=str, default='yield', choices=['yield', 'anthesis', 'silking', 'ASI'], help='Phenotype to consider')
    parser.add_argument('--p_value', type=float, default=0.05, help='p-value threshold to filter GWAS results')
    parser.add_argument('--sequence_length', type=int, default=6000, help='AgroNT is trained to process 1024 tokens ~ 6000 bp')
    # LoRA/DoRA option
    parser.add_argument('--use_lora', action='store_true', help='Load LoRA/DoRA adapters from peft_model_path/lora_adapters/')
    args = parser.parse_args()
    main(args)