import torch as th

def create_th_grid(dim_0_vals, dim_1_vals):
    curves = []
    for dim_0_val in dim_0_vals:
        this_curves = []
        for dim_1_val in dim_1_vals:
            vals = th.stack((dim_0_val, dim_1_val))
            this_curves.append(vals)
        curves.append(th.stack(this_curves))
    curves = th.stack(curves)
    return curves