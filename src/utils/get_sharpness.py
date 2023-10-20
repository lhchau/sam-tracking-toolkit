import torch
import numpy as np
import copy
import torch.nn.functional as F


def get_avg_sharpness(model, scaler, batches, noisy_examples, sigma, n_repeat=5, test='train'):
    # TODO: implement "filter normalization" inside the perturb_weights() function
    loss_diff = 0
    for i in range(n_repeat):
        # rob_err: get loss of current weights
        _, loss_before, _ = rob_err(batches, model, 0, 0, scaler, 0, 1, noisy_examples=noisy_examples, n_batches=1)
        if test == 'val':
            breakpoint()
        # perturb_weights: add random weight into current weights. Gaussian
        weights_delta_dict = perturb_weights(model, add_weight_perturb_scale=sigma, mul_weight_perturb_scale=0, weight_perturb_distr='gauss')
        
        # rob_err: get loss of current weights (after perturbation addition)
        _, loss_after, _ = rob_err(batches, model, 0, 0, scaler, 0, 1, noisy_examples=noisy_examples, n_batches=1)
        
        # move back to initial weights (before perturbation addition)
        subtract_weight_delta(model, weights_delta_dict)

        # measure the average delta of loss as sharpness measure
        # delta1/5 + delta2/5 + delta3/5 + delta4/5 + delta5/5
        loss_diff += (loss_after - loss_before) / n_repeat

    return loss_diff


def eval_sharpness(model, batches, loss_f, rho, step_size, n_iters, n_restarts, apply_step_size_schedule=False, no_grad_norm=False, layer_name_pattern='all', random_targets=False, batch_transfer=False, rand_init=False, verbose=False):
    orig_model_state_dict = copy.deepcopy(model.state_dict())

    n_batches, best_obj_sum, final_err_sum, final_grad_norm_sum = 0, 0, 0, 0
    for _, (x, y) in enumerate(batches):
        x, y = x.cuda(), y.cuda()

        # TODO: for SGD, make f accept a loader (i.e., `batches`) and sample (x, y) from it if sgd=True, thus overriding the usage of the closure's (x, y). then make sure we do it on one "batch" (x, y) from above only; also: do eval of the CE/01 loss on a sufficient n batches
        def f(model):
            obj = loss_f(model(x), y)
            # TODO: put a minus and random targets except `y`
            return obj

        obj_orig = f(model).detach()  
        err_orig = (model(x).max(1)[1] != y).float().mean().item()

        delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}
        orig_param_dict = {param: param.clone() for param in model.parameters()}
        best_obj, final_err, final_grad_norm = 0, 0, 0
        for restart in range(n_restarts):
            # random init on the sphere of radius `rho`
            if rand_init:
                delta_dict = sam.random_init_on_sphere_delta_dict(delta_dict, rho)
                for param in model.parameters():
                    param.data += delta_dict[param]
            else:
                delta_dict = {param: torch.zeros_like(param) for param in model.parameters()}

            if rand_init:
                n_cls = 10
                y_target = torch.clone(y)
                for i in range(len(y_target)):
                    lst_classes = list(range(n_cls))
                    lst_classes.remove(y[i])
                    y_target[i] = np.random.choice(lst_classes)
            def f_opt(model):
                if not rand_init:
                    return f(model)
                else:
                    return -loss_f(model(x), y_target)

            for iter in range(n_iters):
                # for the USAM paper, just use a constant step size
                step_size_curr = step_size_schedule(step_size, iter / n_iters) if apply_step_size_schedule else step_size
                delta_dict = sam.weight_ascent_step(model, f_opt, orig_param_dict, delta_dict, step_size_curr, rho, layer_name_pattern, no_grad_norm=no_grad_norm, verbose=False)
            
            if batch_transfer:
                delta_dict_loaded = torch.load('deltas/gn_erm/batch{}.pth'.format(restart))  
                delta_dict_loaded = {param: delta for param, delta in zip(model.parameters(), delta_dict_loaded.values())}  # otherwise `param` doesn't work directly as a key
                for param in model.parameters():
                    param.data = orig_param_dict[param] + delta_dict_loaded[param]

            obj, grad_norm = utils.eval_f_val_grad(model, f)
            err = (model(x).max(1)[1] != y).float().mean().item()

            # if obj > best_obj:
            if err > final_err:
                best_obj, final_err, final_grad_norm = obj, err, grad_norm
            model.load_state_dict(orig_model_state_dict)
            if verbose:
                delta_norm_total = torch.cat([delta_param.flatten() for delta_param in delta_dict.values()]).norm()
                print('[restart={}] Sharpness: obj={:.4f}, err={:.2%}, delta_norm={:.2f} (step={:.3f}, rho={}, n_iters={})'.format(
                      restart+1, obj - obj_orig, err - err_orig, delta_norm_total, step_size, rho, n_iters))

                # for (param_name, param), delta_param in zip(model.named_parameters(), delta_dict.values()):
                #     assert param.shape == delta_param.shape, 'the order of param and delta_param does not match'
                #     frac_squared_norm = (delta_param**2).sum() / delta_norm_total**2
                #     squared_norm_per_param = (delta_param**2).sum() / np.prod(delta_param.shape)
                #     print('{} ({} params, {:.2f} norm): {:.1%} squared norm ({:.2} per parameter)'.format(param_name, np.prod(delta_param.shape), param.norm(), frac_squared_norm, squared_norm_per_param))

        best_obj, final_err = best_obj - obj_orig, final_err - err_orig  # since we evaluate sharpness, i.e. the difference in the loss
        best_obj_sum, final_err_sum, final_grad_norm_sum = best_obj_sum + best_obj, final_err_sum + final_err, final_grad_norm_sum + final_grad_norm
        n_batches += 1


    # if batch_transfer:
    #     # torch.save(delta_dict, 'deltas/bn_erm/batch{}.pth'.format(i_batch))
    
    if type(best_obj_sum) == torch.Tensor:
        best_obj_sum = best_obj_sum.item()
    if type(final_grad_norm_sum) == torch.Tensor:
        final_grad_norm_sum = final_grad_norm_sum.item()
        
    return best_obj_sum / n_batches, final_err_sum / n_batches, final_grad_norm_sum / n_batches


def rob_err(batches, model, eps, pgd_alpha, scaler, attack_iters, n_restarts, rs=True, linf_proj=True,
            l2_grad_update=False, verbose=False, cuda=True, noisy_examples='default', loss_f=F.cross_entropy,
            n_batches=-1):
    n_corr_classified, train_loss_sum, n_ex = 0, 0.0, 0
    pgd_delta_list, pgd_delta_proj_list = [], []

    for i, (X, y) in enumerate(batches):
        if n_batches != -1 and i > n_batches:  # limit to only n_batches
            break
        if cuda:
            X, y = X.cuda(), y.cuda()
        
        # if noisy_examples == 'none':
        #     X, y = X[~ln], y[~ln]
        # elif noisy_examples == 'all':
        #     X, y = X[ln], y[ln]
        # else:
        assert noisy_examples == 'default'

        if eps > 0:
            pgd_delta = attack_pgd(model, X, y, eps, pgd_alpha, scaler, attack_iters, n_restarts, rs=rs,
                                verbose=verbose, linf_proj=linf_proj, l2_grad_update=l2_grad_update, cuda=cuda)
        else:
            pgd_delta = torch.zeros_like(X)
        with torch.no_grad():
            output = model(X + pgd_delta)
            loss = loss_f(output, y)

        n_corr_classified += (output.max(1)[1] == y).sum().item()
        train_loss_sum += loss.item() * y.size(0)
        n_ex += y.size(0)
        pgd_delta_list.append(pgd_delta.cpu().numpy())

    robust_acc = n_corr_classified / n_ex
    avg_loss = train_loss_sum / n_ex
    pgd_delta_np = np.vstack(pgd_delta_list)

    return 1 - robust_acc, avg_loss, pgd_delta_np

def perturb_weights(model, add_weight_perturb_scale, mul_weight_perturb_scale, weight_perturb_distr):
    with torch.no_grad():
        weights_delta_dict = {}
        for param in model.parameters():
            if param.requires_grad:
                if weight_perturb_distr == 'gauss':
                    delta_w_add = add_weight_perturb_scale * torch.randn(param.shape).cuda()  # N(0, std)
                    delta_w_mul = 1 + mul_weight_perturb_scale * torch.randn(param.shape).cuda()  # N(1, std)
                elif weight_perturb_distr == 'uniform':
                    delta_w_add = add_weight_perturb_scale * (torch.rand(param.shape).cuda() - 0.5)  # U(-0.5, 0.5)
                    delta_w_mul = 1 + mul_weight_perturb_scale * (torch.rand(param.shape).cuda() - 0.5)  # U(1 - 0.5*scale, 1 + 0.5*scale)
                else:
                    raise ValueError('wrong weight_perturb_distr')
                param_new = delta_w_mul * param.data + delta_w_add
                delta_w = param_new - param.data
                param.add_(delta_w)
                weights_delta_dict[param] = delta_w  # only the ref to the `param.data` is used as the key
    return weights_delta_dict

def subtract_weight_delta(model, weights_delta_dict):
    with torch.no_grad():
        for param in model.parameters():
            if param.requires_grad:
                param.sub_(weights_delta_dict[param])  # get back to `w` from `w + delta`

def attack_pgd(model, X, y, eps, alpha, scaler, attack_iters, n_restarts, rs=True, verbose=False,
               linf_proj=True, l2_proj=False, l2_grad_update=False, cuda=True):
    if n_restarts > 1 and not rs:
        raise ValueError('no random step and n_restarts > 1!')
    max_loss = torch.zeros(y.shape[0])
    max_delta = torch.zeros_like(X)
    if cuda:
        max_loss, max_delta = max_loss.cuda(), max_delta.cuda()
    for i_restart in range(n_restarts):
        delta = torch.zeros_like(X)
        if cuda:
            delta = delta.cuda()
        if attack_iters == 0:
            return delta.detach()
        if rs:
            delta.uniform_(-eps, eps)

        delta.requires_grad = True
        for _ in range(attack_iters):

            with torch.cuda.amp.autocast():
                output = model(X + delta)  # + 0.25*torch.randn(X.shape).cuda())  # adding noise (aka smoothing)
                loss = F.cross_entropy(output, y)

            grad = torch.autograd.grad(scaler.scale(loss), delta)[0]
            grad = grad.detach() / scaler.get_scale()

            if not l2_grad_update:
                delta.data = delta + alpha * torch.sign(grad)
            else:
                delta.data = delta + alpha * grad / (grad**2).sum([1, 2, 3], keepdim=True)**0.5

            delta.data = clamp(X + delta.data, 0, 1, cuda) - X
            if linf_proj:
                delta.data = clamp(delta.data, -eps, eps, cuda)
            if l2_proj:
                delta_norms = (delta.data**2).sum([1, 2, 3], keepdim=True)**0.5
                delta.data = eps * delta.data / torch.max(eps*torch.ones_like(delta_norms), delta_norms)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=model.half_prec):
            output = model(X + delta)
            all_loss = F.cross_entropy(output, y, reduction='none')  # .detach()  # prevents a memory leak
            max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]

            max_loss = torch.max(max_loss, all_loss)
            if verbose:  # and n_restarts > 1:
                print('Restart #{}: best loss {:.3f}'.format(i_restart, max_loss.mean()))
    max_delta = clamp(X + max_delta, 0, 1, cuda) - X
    return max_delta

def clamp(X, l, u, cuda=True):
    if type(l) is not torch.Tensor:
        if cuda:
            l = torch.cuda.FloatTensor(1).fill_(l)
        else:
            l = torch.FloatTensor(1).fill_(l)
    if type(u) is not torch.Tensor:
        if cuda:
            u = torch.cuda.FloatTensor(1).fill_(u)
        else:
            u = torch.FloatTensor(1).fill_(u)
    return torch.max(torch.min(X, u), l)