import os
import sys
import numpy as np
import csv


def _read_results(result_path):
    targets, preds, poly_ids = [], [], None
    with open(result_path) as f:
        rows = [row for row in csv.reader(f)]
    targets = np.array([float(row[1]) for row in rows])
    preds = np.array([float(row[2]) for row in rows])
    poly_ids = [row[0] for row in rows]

    # Remove duplicates
    unique_poly_ids = list(set(poly_ids))
    unique_poly_id_idx = []
    for poly_id in unique_poly_ids:
        for idx in range(len(poly_ids)):
            if poly_id == poly_ids[idx]:
                unique_poly_id_idx.append(idx)
                break
    assert len(unique_poly_id_idx) == len(unique_poly_ids)
    poly_ids = [poly_ids[idx] for idx in unique_poly_id_idx]
    targets = np.array([targets[idx] for idx in unique_poly_id_idx])
    preds = np.stack([preds[idx] for idx in unique_poly_id_idx])
    assert len(poly_ids) == targets.shape[0] == preds.shape[0]

    return poly_ids, targets, preds


def print_logp_errors(result_dir):
    noises = '0.00 0.01 0.02 0.04 0.08 0.16 0.32 0.64 1.28 2.56 5.12'.split()
    targets, preds = [], []
    cur_poly_id = None
    for noise in noises:
        poly_id, target, pred = _read_results(os.path.join(
            result_dir, noise, 'test_results.csv'))
        targets.append(target)
        preds.append(pred)
        if cur_poly_id is None:
            cur_poly_id = poly_id
        assert cur_poly_id == poly_id
    for noise, target, pred in zip(noises, targets, preds):
        true_mae = np.mean(np.abs(pred - targets[0]))
        apparent_mae = np.mean(np.abs(pred - target))
        print('For noise = {} split 0, true MAE is {:0.3f}, apparent MAE is {:0.3f}'.format(
            noise, true_mae, apparent_mae))


def _get_cond(cond_csv):
    with open(cond_csv) as f:
        poly_id_to_cond = {row[0]: np.log10(float(row[2]))
                           for row in csv.reader(f)}
    return poly_id_to_cond

def print_poly_rand_errors(result_dir):
    poly_id, target, pred = _read_results(
        os.path.join(result_dir, 'test_results.csv'))
    poly_id_to_cond = {pid: t for pid, t in zip(poly_id, target)}
    poly_id_to_pred = {pid: p for pid, p in zip(poly_id, pred)}
    poly_id_to_cond2 = _get_cond('data/conductivity/cond_5ns_new_config.csv')

    poly_id = [pid for pid in poly_id if pid in poly_id_to_cond2]
    target = np.array([poly_id_to_cond[pid] for pid in poly_id])
    pred = np.array([poly_id_to_pred[pid] for pid in poly_id])
    target2 = np.array([poly_id_to_cond2[pid] for pid in poly_id])
    apparent_mae = np.mean(np.abs(pred - target))
    simulation_mae = np.mean(np.abs(target - target2))
    print('For 5 ns polymer conductivity, apparent MAE for split 0 is {:0.3f}'.format(apparent_mae))
    print('MAE between two indepdent MD simulation is {:0.3f}'.format(simulation_mae))


def print_poly_systematic_errors(result_dir):
    props = ['conductivity', 'li_diff', 'tfsi_diff', 'poly_diff']
    if 'systematic_int' in result_dir:
        mae_5ns_list = [0.528, 0.503, 0.455, 0.612]
        mae_5ns_linear_list = [0.152, 0.148, 0.096, 0.072]
    elif 'systematic_ext' in result_dir:
        mae_5ns_list = [0.278, 0.419, 0.249, 0.528]
        mae_5ns_linear_list = [0.275, 0.247, 0.297, 0.110]
    else:
        raise NotImplementedError
    for prop, mae_5ns, mae_5ns_linear in zip(props, mae_5ns_list, mae_5ns_linear_list):
        poly_id, target, pred = _read_results(
            os.path.join(result_dir, prop, 'exp_test_results.csv'))
        mae = np.mean(np.abs(target - pred))
        print('For 50 ns polymer {}, corrected MAE for split 0 is {:0.3f}, MAE to 5ns MD is {:0.3f}, linear corrected MAE is {:0.3f}'.format(prop, mae, mae_5ns, mae_5ns_linear))


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('[Usage] task result_dir')
        sys.exit(1)
    task, result_dir = sys.argv[1:]
    if task == 'logp':
        print_logp_errors(result_dir)
    elif task == 'poly_rand':
        print_poly_rand_errors(result_dir)
    elif task == 'poly_sys':
        print_poly_systematic_errors(result_dir)
    else:
        raise NotImplementedError
