#Code referred from Andrej Karpathy Video
import torch

class LoRAdata():
    def __init__(self, data):
        # read it in to inspect it
        with open(data, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # here are all the unique characters that occur in this text
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)

    def encode(self, s):
        stoi = { ch:i for i,ch in enumerate(self.chars) }
        return [stoi[c] for c in s]
    
    def decode(self, l):
        itos = { i:ch for i,ch in enumerate(self.chars) }
        return ''.join([itos[i] for i in l])

    def getTrain(self):
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        # Let's now split up the data into train and validation sets
        n = int(0.9*len(data)) # first 90% will be train, rest val
        part = data[:n]
        x = torch.tensor(part[:-1], dtype=torch.long)
        y = torch.tensor(part[1:], dtype=torch.long)
        return x, y
    
    def getVal(self):
        data = torch.tensor(self.encode(self.text), dtype=torch.long)
        # Let's now split up the data into train and validation sets
        n = int(0.9*len(data)) # first 90% will be train, rest val
        part = data[n:]
        x = torch.tensor(part[:-1], dtype=torch.long)
        y = torch.tensor(part[1:], dtype=torch.long)
        return x, y
        
