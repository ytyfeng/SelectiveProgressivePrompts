import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Tokenizer, T5EncoderModel
import numpy as np


# currentInput = current input embedding vec with shape of [batch_size, seq_len, embedding_size]
# prev_Inputs = a list of previous input embeddings of shape [embedding_size]
def similarityScore(currentInput, prev_Inputs):
        cos = nn.CosineSimilarity(dim=0)
        similarities = []
        # embedding for a single element in a batch with shape of [512,1024]
        # 1D embedding vec of [1024] for the whole sequence
        input_embed_1024 = currentInput.squeeze()
        input_embed_tensor = F.normalize(input_embed_1024, p=2, dim=0)
        
        for prev in prev_Inputs:
            prev = prev.squeeze()
            if torch.equal(input_embed_tensor, prev):
                print("SAME EMBEDDING VEC!!!")
                continue
            prev_tensor = F.normalize(prev, p=2, dim=0)
            sim = cos(input_embed_tensor, prev_tensor)
            similarities.append(sim.item())
            print("cos similarities: " , similarities)
        if not similarities:
            return 0
        similarity = F.softmax(torch.tensor(similarities), dim=0)
        print("similarity: ", similarity.detach().cpu().numpy().tolist())
        max = torch.max(similarity).item()
        return max

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-large")
    model = T5EncoderModel.from_pretrained("t5-large")
    prev_Inputs = []
    text = "hello world"
    text2 = "hello world"
    text3 = "hello world"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)
    inputs3 = tokenizer(text3, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
        outputs2 = model(input_ids=inputs2['input_ids'], attention_mask=inputs2['attention_mask'])
        outputs3 = model(input_ids=inputs3['input_ids'], attention_mask=inputs3['attention_mask'])

    embeddings = outputs.last_hidden_state
    embeddings_mean = torch.mean(embeddings, dim=1)
    embeddings2 = outputs2.last_hidden_state
    embeddings_mean2 = torch.mean(embeddings2, dim=1)
    embeddings3 = outputs3.last_hidden_state
    embeddings_mean3 = torch.mean(embeddings3, dim=1)
    prev_Inputs.append(embeddings_mean2)
    prev_Inputs.append(embeddings_mean3)
    print("EMBEDDING SHAPE: ")
    print(embeddings.detach().cpu().numpy().shape)
    print(embeddings2.detach().cpu().numpy().shape)
    print(embeddings3.detach().cpu().numpy().shape)

    # [1, 1024]
    print("EMBEDDING MEAN SHAPE: ")
    print(embeddings_mean.detach().cpu().numpy().shape)
    print(embeddings_mean2.detach().cpu().numpy().shape)
    print(embeddings_mean3.detach().cpu().numpy().shape)

    print("SIMILARITY SCORE: ")
    print(similarityScore(embeddings_mean, prev_Inputs))
     