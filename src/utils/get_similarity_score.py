import torch

def get_named_parameters(net):
    return [
        name
        for name, _ in net.named_parameters()
    ]

def get_similarity_score(grad1, grad2, named_parameters):
    cos = torch.nn.CosineSimilarity(dim=0)
    
    total_sim = 0
    
    dic = {
        "conv": (0, 0, 0, 0), "bn": (0, 0, 0, 0), "shortcut": (0, 0, 0, 0), "linear": (0, 0, 0, 0)
    }
    
    for tensor1, tensor2, name in zip(grad1, grad2, named_parameters):
        sim_score = cos(tensor1, tensor2).mean().item()
        total_sim += sim_score
        for named_layer in dic.keys():
            if named_layer in name and 'weight' in name:
                dic[named_layer][0] += sim_score
                dic[named_layer][1] += 1
                break
            if named_layer in name and 'bias' in name:
                dic[named_layer][2] += sim_score
                dic[named_layer][3] += 1
                break

    similar_scores = {
        "conv_weight": dic['conv'][0] / dic['conv'][1],
        "conv_bias": dic['conv'][2] / dic['conv'][3],
        "bn_weight": dic['bn'][0] / dic['bn'][1],
        "bn_bias": dic['bn'][2] / dic['bn'][3],
        "shortcut_weight": dic['shortcut'][0] / dic['shortcut'][1],
        "shortcut_bias": dic['shortcut'][2] / dic['shortcut'][3],
        "linear_weight": dic['linear'][0] / dic['linear'][1],
        "linear_bias": dic['linear'][2] / dic['linear'][3]
    }
    
    return total_sim / len(named_parameters), similar_scores
        