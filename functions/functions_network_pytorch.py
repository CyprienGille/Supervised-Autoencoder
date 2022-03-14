import functions.functions_torch as ft


def weights_and_sparsity(model, tol=1.0e-3):
    """
    It extracts the weights and calculate their spasity (using the tol as the threshold to judge zero element)
    respectively from the model, and return two dict type results.
    """
    state = model.state_dict()
    weights = {}
    spsty = {}
    for key in state.keys():
        if "weight" in key.lower():
            w = state[key].cpu().numpy()
            # w[w<tol]=0.0
            weights[key] = w
            spsty[key] = ft.sparsity(state[key], tol=tol)
    return weights, spsty
