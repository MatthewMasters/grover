import pickle
import os

import numpy as np
from torch.utils.data import DataLoader

from grover.data import MolCollator
from grover.util.utils import create_logger, load_checkpoint, get_data


def generate_embeddings(args, logger):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    checkpoint_path = args.checkpoint_paths[0]
    if logger is None:
        logger = create_logger('fingerprints', quiet=False)
    print('Loading data...')

    test_data = get_data(path=args.data_path, args=args, use_compound_names=True, skip_invalid_smiles=False)
    molecule_ids = test_data.compound_names()

    logger.info(f'Total size = {len(test_data):,}')
    logger.info(f'Generating...')
    # Load model
    model = load_checkpoint(checkpoint_path, cuda=args.cuda, current_args=args, logger=logger)

    model.eval()
    args.bond_drop_rate = 0

    mol_collator = MolCollator(args=args, shared_dict={})

    mol_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=mol_collator)
    curr_row = 0
    for item in mol_loader:
        _, batch, _, _, _ = item
        _, _, _, _, _, a_scope, b_scope, _ = batch
        batch_preds = model(batch)
        batch_preds = {k: v.detach().cpu().numpy() for k, v in batch_preds.items()}
        atom_embeddings = np.hstack([batch_preds['atom_from_atom'], batch_preds['atom_from_bond']])
        bond_embeddings = np.hstack([batch_preds['bond_from_bond'], batch_preds['bond_from_atom']])

        # save each molecule embedding separately
        for i, ((a_start, a_len), (b_start, b_len)) in enumerate(zip(a_scope, b_scope)):
            mol_id = molecule_ids[curr_row]
            mol_id = str(curr_row) if mol_id is None else mol_id
            curr_row += 1
            save_path = os.path.join(args.output_path, f'{mol_id}_embeddings.pkl')
            atom_emb = atom_embeddings[a_start-1:a_start+a_len-1]
            bond_emb = bond_embeddings[b_start-1:b_start+b_len-1]
            with open(save_path, 'wb') as handle:
                pickle.dump({'atoms': atom_emb, 'bonds': bond_emb}, handle)
