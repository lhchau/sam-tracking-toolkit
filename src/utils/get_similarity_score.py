import torch

def get_similarity_score(grad1, grad2):
    cos = torch.nn.CosineSimilarity(dim=0)

    similar_scores = [
        cos(tensor1, tensor2).mean().item()
        for tensor1, tensor2 in zip(grad1, grad2)
    ]
    
    return sum(similar_scores) / len(similar_scores)
        