DNN_hyperparams = {
    'maccs_FP': {
        'dims_hidden': 512,
        'dropout': 0.4,
        'weight_decay': 1e-07,
        'batch_size': 32,
        'max_lr': 0.00075,
        'T0': 25,
        'early_stop': 25
    },
    'morgan_FP': {
        'dims_hidden': 512,
        'dropout': 0.4,
        'weight_decay': 1e-06,
        'batch_size': 1024,
        'max_lr': 0.00075,
        'T0': 50,
        'early_stop': 25
    },
    'pubchem_FP': {
        'dims_hidden': 128,
        'dropout': 0.4,
        'weight_decay': 1e-07,
        'batch_size': 1024,
        'max_lr': 0.0005,
        'T0': 50,
        'early_stop': 50
    },
    'rdkit_MD': {
        'dims_hidden': 128,
        'dropout': 0.4,
        'weight_decay': 1e-07,
        'batch_size': 256,
        'max_lr': 0.001,
        'T0': 25,
        'early_stop': 25
    },
    'mordred_MD': {
        'dims_hidden': 128,
        'dropout': 0.2,
        'weight_decay': 1e-06,
        'batch_size': 1024,
        'max_lr': 0.0005,
        'T0': 25,
        'early_stop': 50
    }
}


GNN_hyperparams = {
    'GCN':   {'hidden': 256, 'embedding': 16, 'weight_decay': 1e-07, 'batch_size': 32,
              'max_lr': 0.001, 'T0': 25, 'early_stop': 25},
    'EGCN':  {'hidden': 256, 'embedding': 16, 'weight_decay': 1e-07, 'batch_size': 32,
              'max_lr': 0.001, 'T0': 25, 'early_stop': 25},
    'GAT':   {'hidden': 128, 'embedding': 16, 'weight_decay': 1e-07, 'batch_size': 32,
              'max_lr': 0.00075, 'T0': 25, 'early_stop': 25}
}