""" To run the following code: 
torchrun --nproc_per_node=4 stress_aware_agront.py --version <OUTPUT_FOLDER_NAME> → use_lora = False (prompt tuning only)
torchrun --nproc_per_node=4 stress_aware_agront.py --use_lora --version <OUTPUT_FOLDER_NAME> → use_lora = True (prompt tuning + LoRA)
torchrun --nproc_per_node=4 stress_aware_agront.py --use_lora --use_dora --version <OUTPUT_FOLDER_NAME>→ prompt tuning + DoRA
torchrun --nproc_per_node=4 stress_aware_agront.py --use_lora --no_prompt_tuning --version <OUTPUT_FOLDER_NAME>→ LoRA only    (ablation study)
torchrun --nproc_per_node=4 stress_aware_agront.py --use_lora --use_dora --no_prompt_tuning --version <OUTPUT_FOLDER_NAME> → DoRA only (ablation study)
"""

import os
import argparse
import sys
import json
from datetime import datetime
import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoConfig
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from peft import LoraConfig, get_peft_model
from genomic_utils import read_fasta, sig_snps_to_seq

def supcon_loss(features, labels, temperature=0.07, eps=1e-8):
    '''Supervised Contrastive Loss: Pulls together embeddings of samples with the same label while pushing apart
    embeddings of samples with different labels in the representation space.
    Args:
        features: Tensor of shape [batch_size, embedding_dim] - normalized embeddings
        labels: Tensor of shape [batch_size] - class labels for each sample
        temperature: Scaling factor for logits (lower = sharper distribution)
        eps: Small constant for numerical stability
    Returns: Scalar loss value
    '''
    # L2 normalize features to unit sphere
    features = torch.nn.functional.normalize(features, p=2, dim=1, eps=eps)
    device = features.device
    batch_size = features.size(0)
    # Create label mask: mask[i,j] = 1 if labels[i] == labels[j]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(features.device)
    # Compute cosine similarity matrix scaled by temperature
    anchor_dot_contrast = torch.div(torch.matmul(features, features.T), temperature)
    # Numerical stability: subtract max for stable softmax
    logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    logits = anchor_dot_contrast - logits_max.detach()
    # Mask out self-contrast (diagonal elements)
    logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
    mask = mask * logits_mask
    # Compute log-softmax over all negatives + positives (excluding self)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True).clamp_min(eps))
    # Average log-probability over positive pairs
    mask_sum = mask.sum(1).clamp_min(1.0)
    mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
    # Return negative mean (to minimize)
    loss = -mean_log_prob_pos.mean()
    return loss

class StressAwareAgroNT(torch.nn.Module):
    """
    AgroNT model with learnable stress prompt tokens and optional LoRA adaptation.
    Implements prompt tuning as the primary parameter-efficient fine-tuning (PEFT) 
    strategy: the base model is frozen while learnable stress condition token 
    embeddings are trained via contrastive learning. Optionally, LoRA (Low-Rank 
    Adaptation) can be applied to attention layers for additional model adaptation.
    
    Architecture:
        - Base (frozen): AgroNT masked language model (1B parameters)
        - Prompt tuning (PEFT): Learnable stress token embeddings (4 tokens)
        - Prompt Tuning + Optional LoRA (PEFT): Low-rank adapters on query/value attention projections
        - LoRA/DoRA (PEFT) only without prompt tuning
    """
    def __init__(self, output_folder, stress_tokens, rank, 
                 model_name='agro-nucleotide-transformer-1b',
                 use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.05, use_dora=False, use_rslora=False, no_prompt_tuning=False):
        super().__init__()
        self.model_name = model_name
        self.rank = rank
        self.stress_tokens = stress_tokens
        self.stress_embeddings = None   # Learnable stress embedding parameters
        self.stress_ids = None          # Token IDs for stress tokens in vocabulary
        self.output_folder = output_folder
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.use_dora = use_dora
        self.use_rslora = use_rslora
        self.no_prompt_tuning = no_prompt_tuning
        
        self.load_model_tokenizer()
        self.add_special_tokens()
        if self.use_lora:
            self.apply_lora()           # PEFT automatically freezes base model + marks LoRA trainable
        else:
            self.freeze_base_model()     # Manually freeze base model (no-LoRA case)

    def load_model_tokenizer(self):
        ''' Fetch model and tokenizer from InstaDeep's huggingface repository'''
        config = AutoConfig.from_pretrained(f'InstaDeepAI/{self.model_name}', 
                                            output_attentions=True)
        config.attn_implementation = "eager"        # Use eager attention implementation for compatibility
        self.model = AutoModelForMaskedLM.from_pretrained(f'InstaDeepAI/{self.model_name}', 
                                                          cache_dir="./pretrained_model", 
                                                          config=config)
        self.tokenizer = AutoTokenizer.from_pretrained(f'InstaDeepAI/{self.model_name}', 
                                                       cache_dir="./pretrained_model")
        self.tokenizer.padding_side = "right"
        if self.rank == 0:
            print(f"Loaded the {self.model_name} model with {self.model.num_parameters()} parameters and corresponding tokenizer.")
    
    def add_special_tokens(self):
        """
        Add stress condition tokens to vocabulary and initialize their embeddings.
        The stress tokens are added as special tokens, and their embeddings are
        extracted as learnable parameters separate from the frozen model embeddings.
        When no_prompt_tuning=True, stress_embeddings are created but frozen 
        (requires_grad=False) so all code paths work without modification.
        """
        original_vocab_size = len(self.tokenizer)
        num_added_tokens = self.tokenizer.add_special_tokens({"additional_special_tokens": self.stress_tokens})
        tokenizer_vocab_size = len(self.tokenizer)
        expected_vocab_size = original_vocab_size + len(self.stress_tokens)
        assert tokenizer_vocab_size == expected_vocab_size, \
            f"Vocab size mismatch: expected {expected_vocab_size}, got {tokenizer_vocab_size}"
        if self.rank == 0: 
            print("=" * 100)
            print(f"Added {num_added_tokens} tokens - Vocabulary Size: {original_vocab_size} -> {tokenizer_vocab_size}")
        # Resize model embeddings to accommodate new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        # Manually resize LM head bias to match new vocabulary size
        with torch.no_grad():
            lm_head = self.model.lm_head
            if hasattr(lm_head, 'bias') and lm_head.bias is not None:
                lm_head_bias = lm_head.bias
                if lm_head_bias.size(0) != tokenizer_vocab_size:
                    new_bias = torch.zeros(tokenizer_vocab_size, dtype=lm_head_bias.dtype, device=lm_head_bias.device)
                    new_bias[:original_vocab_size-2] = lm_head_bias         # Subtracting 2 is intended to match as vocab was later updated with "<eos>", "<bos>", not there in AgroNT.
                    self.model.lm_head.bias = torch.nn.Parameter(new_bias)
        # Verify all embedding/output sizes match
        input_emb_size = self.model.get_input_embeddings().weight.size(0)
        lm_head_weight_size = self.model.lm_head.decoder.weight.size(0)
        lm_head_bias_size = self.model.lm_head.bias.size(0)
        assert input_emb_size == lm_head_weight_size == lm_head_bias_size == tokenizer_vocab_size, \
            f"Size mismatch: input_emb={input_emb_size}, lm_head_weight={lm_head_weight_size}, lm_head_bias={lm_head_bias_size}, tokenizer={tokenizer_vocab_size}"
        
        # Get token IDs for stress tokens
        self.stress_ids = [self.tokenizer.convert_tokens_to_ids(token) for token in self.stress_tokens]
        input_embeddings = self.model.get_input_embeddings().weight
        self.avg_vocab_norm = input_embeddings.detach().norm(dim=-1).mean()
        # Initialize stress embeddings from the model's initialized embeddings
        # When no_prompt_tuning=True, frozen (requires_grad=False) so code paths work but embeddings don't learn
        stress_embed_data = torch.stack([input_embeddings[sid].clone() for sid in self.stress_ids])
        self.stress_embeddings = torch.nn.Parameter(stress_embed_data, requires_grad=(not self.no_prompt_tuning))
        if self.rank == 0:
            output_dir = os.path.join(self.output_folder, "peft_model")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(stress_embed_data, os.path.join(output_dir, 'stress_embeddings_init.pth'))
            self.tokenizer.save_pretrained(output_dir)
            print(f" - Saved initial stress embeddings and updated tokenizer in {output_dir}")
            if self.no_prompt_tuning:
                print(f" - Stress embeddings are FROZEN (no prompt tuning)")
            # Log pairwise distances between initial stress embeddings
            stress_tokens = [self.tokenizer.convert_ids_to_tokens(sid) for sid in self.stress_ids]
            n = len(stress_tokens)
            print(" - Pairwise L2 norm differences between initial stress embeddings:")
            for i in range(n):
                for j in range(i + 1, n):
                    diff = torch.norm(stress_embed_data[i] - stress_embed_data[j], p=2).item()
                    print(f"  {stress_tokens[i]} <-> {stress_tokens[j]}: {diff:.4f}")
            print("-" * 100)
            print(" - Pairwise cosine similarities between initial stress embeddings:")
            for i in range(n):
                for j in range(i + 1, n):
                    # Cosine similarity: (a · b) / (||a|| * ||b||)
                    cos_sim = torch.nn.functional.cosine_similarity(stress_embed_data[i].unsqueeze(0),stress_embed_data[j].unsqueeze(0),dim=1).item()
                    print(f"{stress_tokens[i]} <-> {stress_tokens[j]}: {cos_sim:.4f}")
            print("-" * 100)
        
    def apply_lora(self):
        """
        Apply LoRA or DoRA to the model's attention layers.
        Freezes pretrained weights and injects trainable low-rank decomposition 
        matrices into attention layers. DoRA further decomposes weights into 
        magnitude and direction, applying LoRA only to the direction component.
        Target modules: query and value projections in self-attention
        """
        
        lora_config = LoraConfig(
            r=self.lora_r,                    # Rank of the low-rank matrices
            lora_alpha=self.lora_alpha,       # Scaling factor
            lora_dropout=self.lora_dropout,   # Dropout for LoRA layers
            bias="none",                       # Don't train biases
            target_modules=["query", "value"], # Target attention Q and V projections
            modules_to_save=None,              # We handle embeddings separately
            use_dora=self.use_dora,            # DoRA: weight-decomposed low-rank adaptation
            use_rslora = self.use_rslora
        )
        
        # Wrap model with LoRA/DoRA
        self.model = get_peft_model(self.model, lora_config)
        
        if self.rank == 0:
            method = "DoRA" if self.use_dora else "rsLoRA" if self.use_rslora else "LoRA"
            print("=" * 100)
            print(f"{method} Configuration:")
            print(f"  - Rank (r): {self.lora_r}")
            print(f"  - Alpha: {self.lora_alpha}")
            print(f"  - Dropout: {self.lora_dropout}")
            print(f"  - Target modules: {lora_config.target_modules}")
            print(f"  - DoRA: {self.use_dora}")
            self.model.print_trainable_parameters()
            print("=" * 100)

    def freeze_base_model(self):
        """Freeze all base model parameters when LoRA is not used.
        When LoRA is enabled, PEFT handles freezing internally via get_peft_model().
        Freezing before DDP wrapping ensures DDP only registers trainable 
        parameters (stress_embeddings) for gradient synchronization.
        """
        for param in self.model.parameters():
            param.requires_grad = False
        # stress_embeddings is an nn.Parameter on PromptedAgroNT (not inside self.model),
        # so it's not affected by the loop above and stays requires_grad=True.

    def forward(self, input_ids, **kwargs):
        """
        Forward pass with learnable stress token embeddings injected.
        The stress token positions in the input are identified and their embeddings
        are replaced with the learnable stress_embeddings parameters, allowing
        gradients to flow through them during training.
        When no_prompt_tuning=True, stress_embeddings are frozen so torch.where
        still runs but no gradients flow to stress_embeddings.
        Args:
            input_ids: Tensor of shape [batch, seq_len] with token IDs
            **kwargs: Additional arguments passed to the model (attention_mask, etc.)
        Returns:
            Model outputs (logits, hidden states, attentions depending on config)
        """
        # Get standard embeddings for all tokens
        input_embeddings = self.model.get_input_embeddings()
        inputs_embeds = input_embeddings(input_ids)
        # Replace stress token embeddings with learned parameters
        for i, stress_id in enumerate(self.stress_ids):
            mask = (input_ids == stress_id).unsqueeze(-1)  # [batch, seq, 1]
            if mask.any():
                learned_embedding = self.stress_embeddings[i]
                learned_embedding = learned_embedding.view(1, 1, -1)
                inputs_embeds = torch.where(mask, learned_embedding, inputs_embeds)
        outputs = self.model(inputs_embeds=inputs_embeds, **kwargs)
        return outputs

class AgroNT_PEFT():
    """
    Parameter-Efficient Fine-tuning for stress tokens and inference class for StressAwareAgroNT.
    Implements:
        - Contrastive learning to train stress token embeddings
        - Optional LoRA/DoRA fine-tuning of attention layers with warmup
    """
    def __init__(self, model, device, rank, output_folder):
        """
        Args:
            model: DDP-wrapped PromptedAgroNT model
            device: CUDA device for this process
            rank: DDP process rank
            output_folder: Directory for saving outputs
        """
        self.rank = rank
        self.output_folder = output_folder
        self.model = model
        self.device = device
        
    def training(self, prompted_sequences_dict, batch_size, num_epochs=10, 
                      learning_rate=0.0001, lora_lr=0.0001, warmup_epochs=3):
        """Train stress embeddings (and optionally LoRA/DoRA) using supervised contrastive loss.
        
        Training schedule when LoRA/DoRA is enabled:
            - Epochs 1 to warmup_epochs: Only stress embeddings are trained (LoRA frozen)
            - Epochs warmup_epochs+1 to num_epochs: Both stress embeddings and LoRA are trained
        This warmup allows stress embeddings to learn meaningful representations before
        LoRA begins adapting the attention layers.
        When no_prompt_tuning=True, only LoRA parameters are trained (no warmup).
        """
        prompted_sequences = []
        labels = []
        stress_list = list(prompted_sequences_dict.keys())
        for stress_idx, stress in enumerate(stress_list):
            sequences = prompted_sequences_dict[stress]
            prompted_sequences.extend(sequences)
            labels.extend([stress_idx] * len(sequences))
        del prompted_sequences_dict

        no_pt = self.model.module.no_prompt_tuning
        use_lora = self.model.module.use_lora

        # Ensure stress embeddings are trainable (only when prompt tuning is enabled)
        if not no_pt:
            self.model.module.stress_embeddings.requires_grad = True

        # Collect LoRA/DoRA parameters
        lora_params = []
        if use_lora:
            lora_params = [p for n, p in self.model.module.model.named_parameters() 
                          if 'lora_' in n.lower() and p.requires_grad]

        # Build optimizer based on training mode and warmup strategy
        if no_pt:
            # LoRA-only mode: no prompt tuning, no warmup
            optimizer = torch.optim.AdamW([
                {'params': lora_params, 'lr': lora_lr, 'name': 'lora_params'}
            ])
        elif use_lora and warmup_epochs > 0:
            # Warmup: freeze LoRA, start with stress-only optimizer
            for p in lora_params:
                p.requires_grad = False
            optimizer = torch.optim.AdamW([
                {'params': [self.model.module.stress_embeddings], 'lr': learning_rate, 'name': 'stress_embeddings'}
            ])
        elif use_lora:
            # No warmup: train both from the start
            optimizer = torch.optim.AdamW([
                {'params': [self.model.module.stress_embeddings], 'lr': learning_rate, 'name': 'stress_embeddings'},
                {'params': lora_params, 'lr': lora_lr, 'name': 'lora_params'}
            ])
        else:
            # No LoRA: stress embeddings only
            optimizer = torch.optim.AdamW([
                {'params': [self.model.module.stress_embeddings], 'lr': learning_rate, 'name': 'stress_embeddings'}
            ])

        if self.rank == 0:
            lora_params_count = sum(p.numel() for p in lora_params)
            print(f"Trainable parameters:")
            if not no_pt:
                stress_params = self.model.module.stress_embeddings.numel()
                print(f"  - Stress embeddings: {stress_params:,}")
            else:
                stress_params = 0
                print(f"  - Stress embeddings: frozen (no prompt tuning)")
            if use_lora:
                method = "DoRA" if self.model.module.use_dora else "LoRA"
                if no_pt:
                    print(f"  - {method} parameters: {lora_params_count:,} (LoRA-only mode)")
                elif warmup_epochs > 0:
                    print(f"  - {method} parameters: {lora_params_count:,} (frozen for first {warmup_epochs} epochs)")
                else:
                    print(f"  - {method} parameters: {lora_params_count:,} (no warmup, training from start)")
            print(f"  - Total: {stress_params + lora_params_count:,}")
        
        # Tokenize all prompted sequences
        encodings = self.model.module.tokenizer(prompted_sequences, padding=True, truncation=True, max_length=1024, return_tensors='pt')
        dataset = TensorDataset(encodings['input_ids'], 
                                encodings['attention_mask'],
                                torch.tensor(labels))
        sampler = DistributedSampler(dataset, num_replicas=torch.distributed.get_world_size(), rank=self.rank, shuffle=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=8, pin_memory=True)

        # Prompt tuning with contrastive learning
        self.model.train()
        for epoch in range(num_epochs):
            # After warmup, enable LoRA parameters and rebuild optimizer (skip if no_prompt_tuning)
            if use_lora and not no_pt and epoch == warmup_epochs and warmup_epochs > 0:
                for p in lora_params:
                    p.requires_grad = True
                optimizer = torch.optim.AdamW([
                    {'params': [self.model.module.stress_embeddings], 'lr': learning_rate, 'name': 'stress_embeddings'},
                    {'params': lora_params, 'lr': lora_lr, 'name': 'lora_params'}
                ])
                if self.rank == 0:
                    method = "DoRA" if self.model.module.use_dora else "LoRA"
                    print(f"{'='*100}")
                    print(f"Epoch {epoch + 1}: Warmup complete. Enabling {method} parameters.")
                    print(f"{'='*100}")

            sampler.set_epoch(epoch)  # Ensure shuffling is different each epoch
            total_loss = 0
            for b_ids, b_mask, b_lab in loader:
                b_ids, b_mask, b_lab = b_ids.to(self.device, non_blocking=True), b_mask.to(self.device, non_blocking=True), b_lab.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outs = self.model(b_ids, attention_mask=b_mask, encoder_attention_mask=b_mask, output_hidden_states=True, output_attentions=False)
                hidden = outs.hidden_states[-1]
                stress_embeddings_output = hidden[:, 1, :]  # Output embeddings of stress prompt token
                loss = supcon_loss(stress_embeddings_output, b_lab)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # Aggregate and log loss from all ranks
            loss_tensor = torch.tensor(total_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)

            if self.rank == 0:
                avg_loss = loss_tensor.item() / (dist.get_world_size() * len(loader))
                if no_pt:
                    phase_label = "lora-only"
                elif use_lora and epoch < warmup_epochs:
                    phase_label = "warmup"
                else:
                    phase_label = "full"
                print(f"Epoch {epoch + 1}/{num_epochs} [{phase_label}]: Average loss = {avg_loss:.4f}") 
                # Log pairwise distances between stress embeddings
                stress_tokens = [self.model.module.tokenizer.convert_ids_to_tokens(sid) for sid in self.model.module.stress_ids]
                n = len(stress_tokens)  
                print(f"Pairwise L2 norm differences between stress embeddings after epoch {epoch+1}:")
                current_emb = self.model.module.stress_embeddings.detach().cpu()
                for i in range(n):
                    for j in range(i + 1, n):
                        diff = torch.norm(current_emb[i] - current_emb[j], p=2).item()
                        print(f"{stress_tokens[i]} <-> {stress_tokens[j]} = {diff:.4f}")
                print("=" * 100)
                print(f"Pairwise cosine similarities between stress embeddings after epoch {epoch+1}:")
                for i in range(n):
                    for j in range(i + 1, n):
                        # Cosine similarity: (a · b) / (||a|| * ||b||)
                        cos_sim = torch.nn.functional.cosine_similarity(current_emb[i].unsqueeze(0), current_emb[j].unsqueeze(0),dim=1).item()
                        print(f"{stress_tokens[i]} <-> {stress_tokens[j]}: {cos_sim:.4f}")
                print("-" * 100)
                if not no_pt:
                    torch.save(current_emb, 
                               f"{self.output_folder}/peft_model/stress_embeddings_epoch_{epoch}.pth")
            dist.barrier()  

        if self.rank == 0:
                print("Saving tuned parameters...")
                output_dir = os.path.join(self.output_folder, "peft_model")
                os.makedirs(output_dir, exist_ok=True)
                # Save stress embeddings (only if prompt tuning was active)
                if not no_pt:
                    tuned_embeddings = self.model.module.stress_embeddings.detach().cpu()
                    torch.save(tuned_embeddings, os.path.join(output_dir, 'stress_embeddings_final.pth'))
                # Save LoRA adapters
                if use_lora:
                    self.model.module.model.save_pretrained(os.path.join(output_dir, 'lora_adapters'))
                print(f"Saved tuned parameters to {output_dir}")
        dist.barrier()

    def extract_embeddings(self, sequences, batch_size, outfile_name):
        """ Extract final layer embeddings in a distributed fashion, preserving the original sequence order.
        Extracts two types of embeddings:
        1. Sequence embeddings: Mean-pooled hidden states over DNA tokens (excluding CLS and stress)
        2. Stress embeddings: Hidden state at the stress token position
        """
        self.model.eval()
        world_size = dist.get_world_size()
        # Tokenize sequences
        encodings = self.model.module.tokenizer(sequences, padding="longest", truncation=True, max_length=1024, return_tensors='pt')
        # Include original indices to reconstruct order after distributed processing
        dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.arange(len(sequences)))
        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=self.rank, shuffle=False)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        local_seq_embeds, local_stress_embeds, local_indices = [], [], []
        with torch.no_grad():
            for b_ids, b_mask, b_idx in loader:
                b_ids, b_mask, b_idx = b_ids.to(self.device, non_blocking=True), b_mask.to(self.device, non_blocking=True), b_idx.to(self.device, non_blocking=True)
                outputs = self.model(b_ids, attention_mask=b_mask, encoder_attention_mask=b_mask, output_hidden_states=True, output_attentions=False)

                batch_hidden_states = outputs.hidden_states[-1] # Get final layer hidden states
                sequence_embeds = batch_hidden_states[:, 2:, :] # skip CLS at 0, stress token at 1
                pad_mask = b_mask[:, 2:].unsqueeze(-1)
                mean_pooled_embeds = (sequence_embeds * pad_mask).sum(dim=1) / pad_mask.sum(dim=1)  # Mean pooling over non-padded positions
                stress_embeds = batch_hidden_states[:, 1, :]    # Stress token embedding: position 1

                local_seq_embeds.append(mean_pooled_embeds)
                local_stress_embeds.append(stress_embeds)
                local_indices.append(b_idx)  
        # Concatenate local results
        local_seq_embeds = torch.cat(local_seq_embeds)
        local_stress_embeds = torch.cat(local_stress_embeds)
        local_indices = torch.cat(local_indices)
        # Gather embeddings from all ranks
        all_seq_embeds = [torch.zeros_like(local_seq_embeds) for _ in range(world_size)]
        all_stress_embeds = [torch.zeros_like(local_stress_embeds) for _ in range(world_size)]
        all_indices = [torch.zeros_like(local_indices) for _ in range(world_size)]
        dist.barrier()
        dist.all_gather(all_seq_embeds, local_seq_embeds)
        dist.all_gather(all_stress_embeds, local_stress_embeds)
        dist.all_gather(all_indices, local_indices)
        # Reconstruct original order and remove duplicates (from DistributedSampler padding)
        if self.rank == 0:
            sequence_embeds = torch.cat(all_seq_embeds).cpu()
            stress_embeds = torch.cat(all_stress_embeds).cpu()
            indices = torch.cat(all_indices).cpu()
            sorted_indices, order = torch.sort(indices)
            sequence_embeds = sequence_embeds[order]
            stress_embeds = stress_embeds[order]
            mask = torch.cat([torch.tensor([True]), sorted_indices[1:] != sorted_indices[:-1]])
            num_original_sequences = len(sequences)
            final_sequence_embeds = sequence_embeds[mask]
            final_stress_embeds = stress_embeds[mask]
            assert len(final_sequence_embeds) == num_original_sequences
            # Save final layer embeddings
            os.makedirs(f"{self.output_folder}/dna_sequence_embeddings/", exist_ok=True)
            os.makedirs(f"{self.output_folder}/stress_prompt_embeddings", exist_ok=True)
            seq_out_path = os.path.join(self.output_folder, "dna_sequence_embeddings", outfile_name)
            stress_out_path = os.path.join(self.output_folder, "stress_prompt_embeddings", outfile_name)
            np.save(seq_out_path, final_sequence_embeds.numpy())
            np.save(stress_out_path, final_stress_embeds.numpy())
        dist.barrier()
        
def main(args):
    """Main training pipeline for AgroNT stress prompt tuning.
    1. Initialize distributed training environment
    2. Load and preprocess GWAS-derived sequences (rank 0 only)
    3. Broadcast data to all ranks
    4. Initialize model with stress tokens and optional LoRA
    5. Extract embeddings with initial (untrained) stress tokens
    6. Train stress embeddings via contrastive learning
    7. Extract embeddings with trained stress tokens
    """
    try:
        # Initialize DDP environment
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        torch.backends.cuda.matmul.allow_tf32 = True  # Enable TensorFloat-32
        torch.set_float32_matmul_precision('medium')
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=local_rank)
        # Set up output folder
        args.output_folder = os.path.join(args.output_folder, 
                                          f"RefSeq_p{str(args.p_value).replace('0.', '')}", 
                                          f"{args.version}")
        # Define stress condition tokens
        stress_tokens = ['<NO_STRESS>', '<HEAT_STRESS>', '<DROUGHT_STRESS>', '<HEAT_DROUGHT_STRESS>']
        # Set seeds for reproducibility
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        if rank == 0:
            os.makedirs(args.output_folder, exist_ok=True)
            sys.stdout = open(os.path.join(args.output_folder, "log.txt"), "w")
            print("=============================== CONFIGURATION ==================================")
            print(json.dumps(vars(args), indent=4))
            print(f"\n{'='*100}")
            print(f"Execution started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*100}\n")
            print(f"Stress tokens: {stress_tokens}")
            print(f"{'='*100}\n")
        
        '''Data preparation on rank 0 only.'''
        # Map stress condition file names to stress tokens
        stress_map = {
            'NO_STRESS': '<NO_STRESS>',
            'HEAT_STRESSED': '<HEAT_STRESS>',
            'DROUGHT_STRESSED': '<DROUGHT_STRESS>',
            'HEAT_DROUGHT_STRESSED': '<HEAT_DROUGHT_STRESS>'
        }
        prompted_seq_train_dict = {}
        prompted_seq_test_dict = {}
        if rank==0:
            '''Read reference genome to memory'''
            ref_genome = read_fasta(args.ref_genome_path)
            # Calculate padding for sequence extraction around SNPs
            base_pad = int(args.sequence_length/2)              # To compute upstream and downstream bases
            # Train/test split by chromosome
            train_chroms = ['1','2','3','4','5','6','7','10']   # Chromosomes 1–7, 10
            test_chroms  = ['8','9']  # Chromosomes 8–9
            os.makedirs(f"{args.output_folder}/dna_sequences", exist_ok=True)    # Folder to store dna sequences, e.g., +/-3kb around GWAS hits
            ''' Read gwas results into memory and pull sequence for top p-value hits for each stress factor'''
            for stress in list(stress_map.keys()):
                seq_df = sig_snps_to_seq(ref_genome, gwasfile=os.path.join(args.gwas_results, f"{args.phenotype}_{stress}.csv"),
                            base_pad=base_pad, 
                            pval_cut=args.p_value)
                seq_df_train = seq_df[seq_df["Chr"].isin(train_chroms)]
                seq_df_test = seq_df[seq_df["Chr"].isin(test_chroms)]
                seq_df_train.to_csv(f"{args.output_folder}/dna_sequences/{args.phenotype}_{stress}_train.csv", index=False)
                seq_df_test.to_csv(f"{args.output_folder}/dna_sequences/{args.phenotype}_{stress}_test.csv", index=False)
                train_sequences = seq_df_train["Sequence"]
                test_sequences = seq_df_test["Sequence"]
                stress_token = stress_map[stress]
                print(f"Train sequences in {stress_token} category: {len(train_sequences)}")
                print(f"Test sequences in {stress_token} category: {len(test_sequences)}")
                prompted_seq_train_dict[stress] = [f"{stress_token} {seq}" for seq in train_sequences]
                prompted_seq_test_dict[stress] = [f"{stress_token} {seq}" for seq in test_sequences]
            del seq_df
            del ref_genome
        
        # Broadcast data to all ranks
        dist.barrier()
        if rank == 0:
            obj_list = [prompted_seq_train_dict, prompted_seq_test_dict] 
        else:
            obj_list = [None, None]
        dist.broadcast_object_list(obj_list, src=0)
        prompted_seq_train_dict, prompted_seq_test_dict = obj_list
        # Initialize model
        agront = StressAwareAgroNT(
                args.output_folder, 
                stress_tokens, 
                rank,
                use_lora=args.use_lora,
                lora_r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                use_dora=args.use_dora,
                use_rslora=args.use_rslora,
                no_prompt_tuning=args.no_prompt_tuning
            ).to(device)
        dist.barrier()
        dist.broadcast(agront.stress_embeddings.data, src=0)
        agront = DDP(agront, device_ids=[local_rank], find_unused_parameters=True)  
                                  
        trainer = AgroNT_PEFT(agront, device, rank, args.output_folder)
        # Extract embeddings with initial (untrained) stress tokens
        if rank == 0: 
            print(f"{'='*100}")
            print("Extract embeddings with initial stress embeddings:")
            print(f"{'-'*100}")
        for phase in ['train', 'test']:
            sequences_dict = prompted_seq_train_dict if phase == 'train' else prompted_seq_test_dict
            for stress in list(stress_map.keys()):
                sequences = sequences_dict[stress]
                outfile_name = f"{args.phenotype}_{stress}_{phase}_init.npy"
                trainer.extract_embeddings(
                    sequences=sequences, 
                    batch_size=args.inference_batch_size,
                    outfile_name=outfile_name
                )
        dist.barrier()
        # Train stress embeddings
        if rank==0:
            print(f"{'='*100}")
            print(f"Start training........")
            print(f"{'-'*100}")
        trainer.training(prompted_seq_train_dict, 
                              batch_size=args.train_batch_size, 
                              num_epochs=args.num_epochs, 
                              learning_rate=args.learning_rate,
                              lora_lr=args.lora_lr,
                              warmup_epochs=args.lora_warmup_epochs)
        dist.barrier()
        # Extract embeddings with trained stress tokens
        if rank==0:
            print(f"Training completed........")
            print(f"{'='*100}")
            print("Extract embeddings with final stress embeddings:")
            print(f"{'-'*100}")
        for phase in ['train', 'test']:
            sequences_dict = prompted_seq_train_dict if phase == 'train' else prompted_seq_test_dict
            for stress in list(stress_map.keys()):
                sequences = sequences_dict[stress]
                outfile_name = f"{args.phenotype}_{stress}_{phase}_tuned.npy"
                trainer.extract_embeddings(
                    sequences=sequences, 
                    batch_size=args.inference_batch_size,
                    outfile_name=outfile_name
                )
        dist.barrier()
        if rank == 0:
            print(f"Execution ended at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*100}\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()

def get_args_parser():
    parser = argparse.ArgumentParser(description="AgroNT - Stress token adaptation")
    parser.add_argument('--gpu_device', type = str, default="1,2,3,4", help="GPU devices to be used")
    parser.add_argument('--train_batch_size', type = int, default=8, help="Batch size per GPU during training")
    parser.add_argument('--inference_batch_size', type = int, default=50, help="Batch size during inference per GPU")
    parser.add_argument('--num_epochs', type = int, default=10, help="Number of training epochs")
    parser.add_argument('--learning_rate', type = float, default=0.0001, help="Learning rate in AdamW optimizer")
    parser.add_argument('--output_folder', type=str, default='StressAwareAgroNT_v2', help='folder to save models and results')
    parser.add_argument('--version', type=str, default='prompt_tuning_v1', help='Versioning each execution') 
    parser.add_argument('--ref_genome_path', type=str, default='../ref_genome_fasta/Zm-B73-REFERENCE-NAM-5.0.fa', help='fasta file to read DNA sequences')
    parser.add_argument('--gwas_results', type=str, default='../gwas/gapit_blue_results', help='GWAS results')
    parser.add_argument('--phenotype', type=str, default='yield', choices = ['yield', 'anthesis', 'silking', 'ASI'], help='Phenotype to consider')
    parser.add_argument('--p_value', type=float, default=0.05, help='p-value threshold to filter GWAS results')
    parser.add_argument('--sequence_length', type=int, default=6000, help='AgroNT is trained to process 1024 tokens ~ 6000 bp')
    # LoRA/DoRA arguments
    parser.add_argument('--use_lora', action='store_true', help='Enable LoRA adaptation')
    parser.add_argument('--use_dora', action='store_true', help='Use DoRA instead of standard LoRA. Requires --use_lora')
    parser.add_argument('--use_rslora', action='store_true', help='Use rsLoRA instead of standard LoRA. Requires --use_lora')
    parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout')
    parser.add_argument('--lora_lr', type=float, default=0.00005, help='Learning rate for LoRA parameters')
    parser.add_argument('--lora_warmup_epochs', type=int, default=3, help='Epochs to train only stress embeddings before enabling LoRA/DoRA')
    parser.add_argument('--no_prompt_tuning', action='store_true',
                    help='Disable prompt tuning (freeze default stress embeddings). Use with --use_lora for LoRA-only mode.')
    return parser

if __name__== '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    if args.use_dora and not args.use_lora:
        parser.error("--use_dora requires --use_lora")
    if args.no_prompt_tuning and not args.use_lora:
        parser.error("--no_prompt_tuning requires --use_lora (otherwise nothing is trainable)")
    if args.use_dora and args.use_rslora:
        parser.error("--use_dora and --use_rslora are mutually exclusive")
    if args.use_rslora and not args.use_lora:
        parser.error("--use_rslora requires --use_lora")
    os.environ["OMP_NUM_THREADS"] = "16"          # OpenMP threads
    os.environ["MKL_NUM_THREADS"] = "16"          # MKL threads
    os.environ["NUMEXPR_NUM_THREADS"] = "16"
    os.environ["OPENBLAS_NUM_THREADS"] = "16"
    os.environ['HOME'] = './'
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
    main(args)