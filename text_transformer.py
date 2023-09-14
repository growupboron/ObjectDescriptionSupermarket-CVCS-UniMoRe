import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

class SceneDescriptionDataset(Dataset):
    def __init__(self, descriptions):
        self.descriptions = descriptions
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    def __len__(self):
        return len(self.descriptions)
    
    def __getitem__(self, idx):
        return self.tokenizer.encode(self.descriptions[idx], return_tensors='pt')

class SceneDescriptionGenerator:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.to(device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    def generate_description(self, objects_count, positions_3d, relationships):
        # Construct input text
        input_text = f"There are {objects_count} objects in the scene. "
        for i, pos in enumerate(positions_3d):
            input_text += f"Object {i+1} is located at ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}). "
        input_text += "Relationships between objects: "
        for rel in relationships:
            input_text += f"{rel[0]} is {rel[1]} {rel[2]}. "
        
        # Generate scene description
        input_ids = self.tokenizer.encode(input_text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)[0]
        
        generated_description = self.tokenizer.decode(output, skip_special_tokens=True)
        return generated_description

