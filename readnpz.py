import sys
import os
import numpy as np

idx = sys.argv[1]

for idx in [386, 387, 393]: 
    file_path = f'/home/wdebang/workspace/3dgs-avatar-release/exp/zju_{idx}_mono-direct-mlp_field-ingp-shallow_mlp-default/test-view/results.npz'

    di = np.load(file_path)
    print(idx)
    for k in di.keys():
        if k == 'lpips':
            print(k, float(di[k]) * 1000)
        else:
            print(k, di[k])
        if di[k].shape:
            print(di[k][0])
    print()