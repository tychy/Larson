import json
import numpy as np
import os
import shutil


def read_json():
    with open("configs.json", "r") as f:
        config = json.load(f)
    return config


def copy_json(base_dir):
    shutil.copy("configs.json", os.path.join(base_dir, "configs.json"))


def write_index(base_dir, idx):
    path = base_dir + "/index.txt"
    with open(path, mode="w") as f:
        f.write(str(idx))
    return


def read_index(base_dir):
    path = base_dir + "/index.txt"
    with open(path) as f:
        s = f.read()
    return int(s)


def save(base_dir, idx, t_h, deltat, v, r, rho, p, tmp, r_h, r_l, p_l, t):
    np.save(base_dir + "/step_t_h.npy".format(idx), t_h)
    np.save(base_dir + "/step_deltat.npy".format(idx), deltat)
    np.save(base_dir + "/step_v.npy".format(idx), v)
    np.save(base_dir + "/step_r.npy".format(idx), r)
    np.save(base_dir + "/step_rho.npy".format(idx), rho)
    np.save(base_dir + "/step_p.npy".format(idx), p)
    np.save(base_dir + "/step_tmp.npy".format(idx), tmp)
    np.save(base_dir + "/step_r_h.npy".format(idx), r_h)
    np.save(base_dir + "/step_r_l.npy".format(idx), r_l)
    np.save(base_dir + "/step_p_l.npy".format(idx), p_l)
    np.save(base_dir + "/step_t.npy".format(idx), t)
    write_index(base_dir, idx)
    return


def save_with_energy(base_dir, idx, v, r, rho, p, tmp, r_h, t, Q, e):
    np.save(base_dir + "/step_v.npy".format(idx), v)
    np.save(base_dir + "/step_r.npy".format(idx), r)
    np.save(base_dir + "/step_rho.npy".format(idx), rho)
    np.save(base_dir + "/step_p.npy".format(idx), p)
    np.save(base_dir + "/step_tmp.npy".format(idx), tmp)
    np.save(base_dir + "/step_r_h.npy".format(idx), r_h)
    np.save(base_dir + "/step_t.npy".format(idx), t)
    np.save(base_dir + "/step_Q.npy".format(idx), Q)
    np.save(base_dir + "/step_e.npy".format(idx), e)
    write_index(base_dir, idx)
    return


def save_with_ionization(base_dir, idx, v, r, rho, p, tmp, r_h, t, Q, e, fh, fht, fion):
    np.save(base_dir + "/step_v.npy".format(idx), v)
    np.save(base_dir + "/step_r.npy".format(idx), r)
    np.save(base_dir + "/step_rho.npy".format(idx), rho)
    np.save(base_dir + "/step_p.npy".format(idx), p)
    np.save(base_dir + "/step_tmp.npy".format(idx), tmp)
    np.save(base_dir + "/step_r_h.npy".format(idx), r_h)
    np.save(base_dir + "/step_t.npy".format(idx), t)
    np.save(base_dir + "/step_Q.npy".format(idx), Q)
    np.save(base_dir + "/step_e.npy".format(idx), e)
    np.save(base_dir + "/step_fh.npy".format(idx), fh)
    np.save(base_dir + "/step_fht.npy".format(idx), fht)
    np.save(base_dir + "/step_fion.npy".format(idx), fion)

    write_index(base_dir, idx)
    return