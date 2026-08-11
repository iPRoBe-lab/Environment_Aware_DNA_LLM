import os
import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from genomic_utils import read_fasta, sig_snps_to_seq

def load_model_tokenizer(device, model_name='agro-nucleotide-transformer-1b', ):
    # fetch model and tokenizer from InstaDeep's hf repo
    config = AutoConfig.from_pretrained(f'InstaDeepAI/{model_name}', output_attentions=False)
    print(f"No. of Attention Heads: {config.num_attention_heads}")
    #config.attn_implementation = "eager"
    
    model = AutoModelForMaskedLM.from_pretrained(
        f'InstaDeepAI/{model_name}', cache_dir="./pretrained_model", config=config
    )
    tokenizer = AutoTokenizer.from_pretrained(f'InstaDeepAI/{model_name}', cache_dir="./pretrained_model")
    model = model.to(device)
    print(f"Loaded the {model_name} model with {model.num_parameters()} parameters and corresponding tokenizer.")
    return model, tokenizer

def extract_embeddings(sequences, model, tokenizer, device, batch_size):
    all_embeddings = []
    model.eval()
    #print(f"Tokenzied sequence: {agro_nt_tokenizer.batch_decode(batch_tokens)}")
    # inference
    with torch.no_grad():
        for i in range(0,len(sequences), batch_size):
            batch_seqs = sequences[i:i+batch_size]
            batch_encodings = tokenizer(batch_seqs,padding="longest",
                                        truncation=True, max_length=1024, 
                                        return_overflowing_tokens=True)
            batch_tokens = torch.tensor(batch_encodings['input_ids']).to(device)
            for k, overflow in enumerate(batch_encodings["overflowing_tokens"]):
                if len(overflow) > 0:
                    print(f"Sequence {k+i} was truncated, {len(overflow)} tokens removed")
            attention_mask = torch.tensor(batch_encodings['attention_mask']).to(device)
            outs = model(
                batch_tokens,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True
            )

            # get the final layer embeddings and language model head logits
            batch_embeddings = outs['hidden_states'][-1].detach().cpu().numpy()
            batch_embeddings = batch_embeddings[:, 1:, :]   # skip CLS token
            padding_mask = np.expand_dims((batch_tokens[:, 1:] != tokenizer.pad_token_id).cpu().numpy(), axis=-1)
            masked_embeddings = batch_embeddings * padding_mask  # multiply by 0 pad tokens embeddings
            sequences_lengths = np.sum(padding_mask, axis=1)
            mean_embeddings = np.sum(masked_embeddings, axis=1) / sequences_lengths
            all_embeddings.append(mean_embeddings)
    return np.vstack(all_embeddings)
    

def main(args):
    os.environ['HOME'] = './'
    device = torch.device(f"cuda:{args.gpu_device}" if args.cuda and torch.cuda.is_available() else "cpu")
    # Read reference genome to memory
    ref_genome = read_fasta(args.ref_genome_path)
    base_pad = int(args.sequence_length/2)
    # Load model and tokenizer
    model, tokenizer = load_model_tokenizer(device=device)
    stress_category = ['NO_STRESS', 'HEAT_STRESSED', 'DROUGHT_STRESSED', 'HEAT_DROUGHT_STRESSED'] 
    if args.adjust_threshold == 'bonferroni':
        args.output_folder = os.path.join(args.output_folder, f"RefSeq_p{str(args.p_value).replace("0.", "")}_bonferroni")
    elif args.adjust_threshold == 'fdr':
        args.output_folder = os.path.join(args.output_folder, f"RefSeq_p{str(args.p_value).replace("0.", "")}_fdr")
    else:
        args.output_folder = os.path.join(args.output_folder, f"RefSeq_p{str(args.p_value).replace("0.", "")}")
    os.makedirs(f"{args.output_folder}/dna_sequences", exist_ok=True)
        
    # Read gwas results into memory and pull sequence for top pvalue hits for each stress factor
    print("GWAS Threshold: " + str(args.p_value))
    for stress in stress_category:
        print(f"Extract embeddings from {stress} category...")
        seq_df = sig_snps_to_seq(ref_genome, gwasfile=os.path.join(args.gwas_results, f"{args.phenotype}_{stress}.csv"), 
                    #outfile=f"dna_sequences/{args.phenotype}_refseq_p{str(args.p_value).replace("0.", "")}_{stress}.csv", 
                    base_pad=base_pad, 
                    pval_cut=args.p_value, adjust_threshold = args.adjust_threshold)
        seq_df.to_csv(f"{args.output_folder}/dna_sequences/{args.phenotype}_{stress}.csv", index=False)
        sequences = list(seq_df["Sequence"])
        print(f"Length: {len(sequences)}")
        embeddings = extract_embeddings(sequences=sequences, model=model, tokenizer=tokenizer, device=device, batch_size = args.batch_size)
        os.makedirs(f"{args.output_folder}/dna_sequence_embeddings", exist_ok=True)
        np.save(f"{args.output_folder}/dna_sequence_embeddings/{args.phenotype}_{stress}.npy", embeddings)
        #logits = outs['logits'].detach().numpy()


if __name__== '__main__':
    os.environ["OMP_NUM_THREADS"] = "16"          # OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "16"          # MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = "16"
    os.environ["OPENBLAS_NUM_THREADS"] = "16"

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser(description="Extract DNA embeddings using pretrained AgroNT model")
    parser.add_argument('--cuda', default=True, help='enables cuda')
    parser.add_argument('--gpu_device', default="4", help="GPU devices to be used")
    parser.add_argument('--batch_size', default=60, help="Batch size")
    parser.add_argument('--output_folder', type=str, default='pretrained_embeddings_v2', help='folder to save extracted sequuences and embeddings')
    parser.add_argument('--ref_genome_path', type=str, default='../ref_genome_fasta/Zm-B73-REFERENCE-NAM-5.0.fa', help='fasta file to read DNA sequences')
    parser.add_argument('--gwas_results', type=str, default='../gwas/gapit_blue_results')
    parser.add_argument('--phenotype', type=str, default='yield', choices = ['yield', 'anthesis', 'silking', 'ASI'], help='Phenotype to consider')
    parser.add_argument('--p_value', type=float, default=0.05, help='p-value threshold to filter GWAS results')
    parser.add_argument('--adjust_threshold', type=str, default=None, choices = ['bonferroni', 'fdr'], help='Adjust threshold')
    parser.add_argument('--sequence_length', type=int, default=6000, help='AgroNT is trained to process 1024 tokens ~ 6000 bp')
    args = parser.parse_args()
    main(args)


    

