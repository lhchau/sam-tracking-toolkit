import torch

def get_named_parameters(net):
    return [
        name
        for name, _ in net.named_parameters()
    ]

def get_similarity_score(grad1, grad2, named_parameters):
    cos = torch.nn.CosineSimilarity(dim=0)

    similar_scores = {
        name: cos(tensor1, tensor2).mean().item()
        for tensor1, tensor2, name in zip(grad1, grad2, named_parameters)
    }
    
    return sum(similar_scores.values()) / len(similar_scores), similar_scores
        