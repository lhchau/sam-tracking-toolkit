import torch
import numpy as np

def get_feature_matrix(batches, model, return_block, n_batches=-1):
    with torch.no_grad():
        features_list = []
        for i, (X, y) in enumerate(batches):
            if n_batches != -1 and i > n_batches:
                break
            X, y = X.cuda(), y.cuda()
            features = model(X, return_features=True, return_block=return_block).cpu().numpy()
            features_list.append(features)

        phi = np.vstack(features_list)
        # ResNet18 (default: preact): 1: no, 2: no, 3: no, 4: yes, 5: yes
        # ResNet34 (default: plain): 1: yes, 2: yes, 3: yes, 4: yes, 5: yes
        phi = phi.reshape(phi.shape[0], np.prod(phi.shape[1:]))
        # if return_block in [3, 4]:
        #     # ResNet34: 16384, 8192
        #     # DenseNet100: 76800, 21888
        #     print(phi.shape)  
        return phi
    
def get_feature_sparsity(batches, model, return_block, corr_threshold=0.95, n_batches=-1, n_relu_max=1000):
    with torch.no_grad():
        phi = get_feature_matrix(batches, model, return_block, n_batches)

        
        if phi.shape[1] > n_relu_max:  # if there are too many neurons, we speed it up by random subsampling
            random_idx = np.random.choice(phi.shape[1], n_relu_max, replace=False)
            phi = phi[:, random_idx]

        sparsity = (phi > 0).sum() / (phi.shape[0] * phi.shape[1])

        if corr_threshold < 1.0:
            idx_keep = np.where((phi > 0.0).sum(0) > 0)[0]

            phi_filtered = phi[:, idx_keep]  # filter out always-zeros
            corr_matrix = np.corrcoef(phi_filtered.T) # get correlation matrix 
            corr_matrix -= np.eye(corr_matrix.shape[0]) # filter value 1 on diagonal 

            idx_to_delete, i, j = [], 0, 0
            while i != corr_matrix.shape[0]:
                if (np.abs(corr_matrix[i]) > corr_threshold).sum() > 0: # Delete row, col if exist at least 1 corr > 0.95 in this row 
                    corr_matrix = np.delete(corr_matrix, (i), axis=0)
                    corr_matrix = np.delete(corr_matrix, (i), axis=1)
                    idx_to_delete.append(j)
                else:
                    i += 1
                j += 1
            assert corr_matrix.shape[0] == corr_matrix.shape[1] # check if corr_matrix is square matrix
            idx_keep = np.delete(idx_keep, [idx_to_delete]) # 
            sparsity_rmdup = (phi[:, idx_keep] > 0).sum() / (phi.shape[0] * phi.shape[1])
            n_highly_corr = phi.shape[1] - len(idx_keep)
        
        else:
            sparsity_rmdup, n_highly_corr = sparsity, 0

        # (% weight duong, % weight duong + corr < 0.95, so luong weight co corr > 0.95)
        return (1-sparsity_rmdup) / sparsity