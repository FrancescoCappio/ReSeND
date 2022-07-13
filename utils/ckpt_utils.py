import torch
import os

def check_resume(resume_path):

    if not os.path.isdir(resume_path):
        return False

    if not os.path.isfile(resume_path + "/last_checkpoint.txt"):
        return False

    return True

def resume(models_dict: dict, resume_path: str):
    for key in models_dict.keys():
        ckpt_dict = torch.load(resume_path + '/' + key + '.pkl',map_location="cpu")

        missing, unexpected = models_dict[key].load_state_dict(ckpt_dict,strict=False)
        if len(missing) > 0:
            print("Missing keys: ", missing)
        if len(unexpected) > 0:
            print("Unexpected keys: ", unexpected)


    print("Loaded ckpt")
    with open(resume_path + "/last_checkpoint.txt", "r") as f:
        lines = f.readlines()

    last_iter = int(lines[0].strip())

    return last_iter


def load_ckpt(models_dict: dict, folder: str):
    assert os.path.isdir(folder), f"Ckpt folder {folder} does not exist!"

    for key in models_dict.keys():
        ckpt_dict = torch.load(folder + '/' + key + '.pkl',map_location="cpu")

        missing, unexpected = models_dict[key].load_state_dict(ckpt_dict,strict=False)
        if len(missing) > 0:
            print("Missing keys: ", missing)
        if len(unexpected) > 0:
            print("Unexpected keys: ", unexpected)

    print("Loaded ckpt")

def save_ckpt(models_dict: dict, folder: str, current_it: int, keep_all: bool = False, keep_period: int = 1000):
    assert os.path.isdir(folder), f"Ckpt folder {folder} does not exist!"

    for key in models_dict.keys():
        torch.save(models_dict[key].state_dict(),folder + '/' + key + '.pkl')

    if keep_all:
        if current_it % keep_period == 0:
            for key in models_dict.keys():
                torch.save(models_dict[key].state_dict(),folder + '/' + key + '_' + str(current_it) + '.pkl')

    with open(folder + "/last_checkpoint.txt", "w") as f:
        f.write(str(current_it) + "\n")

    print("Saved ckpt")
