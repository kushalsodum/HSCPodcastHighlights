import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel
import torch.nn as nn
from tqdm import tqdm
import json

from preprocessData import filterHighlights, calculateMetrics

# Set random seed
torch.manual_seed(42)
np.random.seed(42)

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DATA_DIR = "../rhapsody/data"
TEST_FILE = "test-00000-of-00001.parquet"
MODEL_DIR = "rhapsody_model_linear"
THRESHOLD = 0.532

class RhapsodyDataset(Dataset):
    def __init__(self, parquet_file, tokenizer):
        self.df = pd.read_parquet(parquet_file)
        self.tokenizer = tokenizer
        allReplayData = self.df['gt'].to_numpy()
        self.allGTHighlights = filterHighlights(allReplayData)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        summaries = row['segment_summaries']
        gtHighlights = self.allGTHighlights[idx]
        
        # Create binary labels
        labels = [0.0] * 100
        for highlight in gtHighlights:
            labels[highlight] = 1
        
        # Tokenize and concatenate
        input_ids = []
        segment_end_indices = []
        
        # Use the tokenizer's eos_token_id as the separator
        sep_token_id = self.tokenizer.eos_token_id
        
        for summary in summaries:
            tokens = self.tokenizer.encode(summary, add_special_tokens=False)
            input_ids.extend(tokens)
            input_ids.append(sep_token_id)
            segment_end_indices.append(len(input_ids) - 1)
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'segment_end_indices': torch.tensor(segment_end_indices, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float),
            'gt_indices': torch.tensor(gtHighlights, dtype=torch.long) # Pass GT for evaluation
        }

class SegmentClassificationLLM(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        
        # Classification head: Linear + Sigmoid
        # Input dim is hidden size of LLM.
        self.head = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, 1),
            nn.Sigmoid()
        )
        
    def forward(self, input_ids, segment_end_indices):
        # input_ids: (batch_size, seq_len)
        # segment_end_indices: (batch_size, 100)
        
        outputs = self.base_model(input_ids=input_ids)
        hidden_states = outputs.last_hidden_state # (batch_size, seq_len, hidden_size)
        
        # Extract hidden states at segment_end_indices
        batch_size = input_ids.size(0)
        num_segments = segment_end_indices.size(1)
        
        # Create batch indices
        batch_indices = torch.arange(batch_size, device=input_ids.device).unsqueeze(1).expand(-1, num_segments)

        # Gather
        selected_hidden_states = hidden_states[batch_indices, segment_end_indices, :] # (batch_size, 100, hidden_size)

        # print("\nPrinting sizes")
        # print(num_segments)
        # print(batch_indices.size())
        # print(selected_hidden_states.size())
        
        # Apply head
        probs = self.head(selected_hidden_states.float()) # (batch_size, 100, 1)
        return probs.squeeze(-1) # (batch_size, 100)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not os.path.exists(MODEL_DIR):
        print(f"Model directory {MODEL_DIR} not found. Please train the model first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load Base Model
    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )
        print("Using QLoRA (4-bit quantization)")
        base_model = AutoModel.from_pretrained(
            MODEL_ID,
            quantization_config=quantization_config,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    except Exception as e:
        print(f"Could not load with quantization: {e}")
        print("Falling back to standard loading")
        base_model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16
        )
        base_model.to(device)

    # Load Adapters
    print(f"Loading adapters from {MODEL_DIR}")
    base_model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    
    # Initialize Model Wrapper
    model = SegmentClassificationLLM(base_model)
    
    # Load LSTM and Classifier weights
    print("Loading Classifier weights...")
    model.head.load_state_dict(torch.load(os.path.join(MODEL_DIR, "head.pt"), map_location=device))
    
    model.head.to(device)
    
    # Dataset
    test_path = os.path.join(DATA_DIR, TEST_FILE)
    if not os.path.exists(test_path):
        print(f"Test file not found at {test_path}")
        return

    dataset = RhapsodyDataset(test_path, tokenizer)
    # Batch size 1 is safest for variable length sequences without complex padding logic for now
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    
    all_metrics = {
        'hit_rate': [],
        'precision': [],
        'recall': [],
        'f1': [],
    }
    
    results = []
    
    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            segment_end_indices = batch['segment_end_indices'].to(device)
            gt_indices = batch['gt_indices'][0].tolist() # Batch size 1
            
            logits = model(input_ids, segment_end_indices) # (1, 100)
            probs = torch.sigmoid(logits)
            probs = probs.cpu().numpy()[0]
            
            # Get predicted indices > Threshold
            pred_indices = np.where(probs > THRESHOLD)[0].tolist()
            
            # Calculate metrics
            metrics = calculateMetrics(pred_indices, gt_indices)
            for k, v in metrics.items():
                all_metrics[k].append(v)
                
            results.append({
                'gt_indices': gt_indices,
                'pred_indices': pred_indices,
                'probs': probs.tolist(),
                'metrics': metrics
            })
            
    # Average metrics
    avg_metrics = {k: np.mean(v) for k, v in all_metrics.items()}
    
    print("\nEvaluation Results:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")
        
    # Save results
    output_file = "prediction_results_linear.json"
    with open(output_file, 'w') as f:
        json.dump(results, f)
    print(f"\nDetailed results saved to {output_file}")

if __name__ == "__main__":
    main()
