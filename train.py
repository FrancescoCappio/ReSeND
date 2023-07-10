import argparse
from os import environ

import torch
from torchlars import LARS
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import os
import sys
import ast
import time
import datetime
import math

from models.resnet import ResNetFc, CLS, CLS_binary
from optimizer.optimizer_helper import get_optim_and_scheduler
from models.data_helper import get_train_dataloader, get_val_dataloader
from utils.ckpt_utils import load_ckpt, save_ckpt, check_resume, resume
from utils.log_utils import LogUnbuffered, count_parameters
from utils.dist_utils import all_gather
from evals.eval import do_test_target_with_prototypes, do_knn_distance_eval, do_test_target_with_prototypes_new
from utils.utils import get_coreset_idx


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--local_rank", type=int)  # automatically passed by torch.distributed.launch

    parser.add_argument("--dataset", default="ImageNet", help="Dataset name",
                        choices=["ImageNet", "OfficeHome_DG", "PACS_DG", "MultiDatasets_DG", 'DTD', 
                                 'DomainNet_IN_OUT','DomainNet_Painting','OfficeHome_SS_DG','PACS_SS_DG','DomainNet_Sketch',
                                 "imagenet_ood", "imagenet_ood_small", "Places", "DomainNet_DG"])
    parser.add_argument("--source",
                        help="Source_OH: no_Product, no_Art, no_Clipart, no_RealWorld | Source_PACS: no_ArtPainting, no_Cartoon, no_Photo, no_Sketch | Source_MultiDatasets: Sources")
    parser.add_argument("--target",
                        help="Target_OH: Product, Art, Clipart, RealWorld | Target_PACS: ArtPainting, Cartoon, Photo, Sketch | Target_MultiDatasets: Clipart,Painting,Real,Sketch")

    parser.add_argument("--path_to_txt", default="data/txt_lists/", help="Path to the txt files")

    # data augmentation
    parser.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
    parser.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
    parser.add_argument("--random_horiz_flip", default=0.5, type=float, help="Chance of random horizontal flip")
    parser.add_argument("--jitter", default=0.8, type=float, help="Color jitter amount")
    parser.add_argument("--prob_jitter", default=0.5, type=float, help="Probability that the color jitter is applied")
    parser.add_argument("--random_grayscale", default=0.1, type=float, help="Randomly greyscale the image")

    # training parameters
    parser.add_argument("--network", default="resnet18", help="Network: resnet18") # backbone
    parser.add_argument("--n_source_domains", type=int, default=3, help="Source Domains")
    parser.add_argument("--pretrained", type=str, default=None, help="Path to ckpt (folder) to be used as pretrained")

    parser.add_argument("--image_size", type=int, default=224, help="Image size")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=0.008, help="Learning rate")
    parser.add_argument("--iterations", type=int, default=13000, help="Number of iterations")
    parser.add_argument("--step_after", type=int, default=13000, help="Step after")
    parser.add_argument("--warmup", type=int, default=500, help="Number of warmup iters")

    # learning objective
    parser.add_argument("--loss_function", type=str, default="L2", choices=['CE', 'L2'], help="Choose learning objective")

    parser.add_argument("--resume", action='store_true', help="Resume training from last checkpoint if it exists")

    # relational estimator
    parser.add_argument("--transf_depth", type=int, default=4,
                        help="Number of self attention blocks in relational transformer")
    parser.add_argument("--transf_n_heads", type=int, default=12,
                        help="Number of heads in self attention modules for the relational transformer")

    # run params
    parser.add_argument("--seed", type=int, default=42, help="Random seed for data splitting")
    parser.add_argument("--few_shot", type=int, default=-1, help="Number of training samples for each class, -1 means use whole dataset")
    # save model
    parser.add_argument("--suffix", type=str, help="Suffix for output folder name", default="")

    # checkpoint evaluation
    parser.add_argument("--only_eval", action='store_true', default=False,
                        help="If you want only evaluate a checkpoint")
    parser.add_argument("--checkpoint_folder_path", default="outputs/", help="Folder in which the checkpoint is saved")

    # pairing
    parser.add_argument("--class_balancing", type=ast.literal_eval, default=True,
                        help="Enable source class balancing during training?")
    parser.add_argument("--neg_to_pos_ratio", type=int, default=20, help="How many negative pairs for each positive")

    # regression loss params    
    parser.add_argument("--sigmoid_compression", type=float, default=10.0, help="Horizontal compression of the translated sigmoid for regression loss")
    parser.add_argument("--sigmoid_expansion", type=float, default=2.0, help="Vertical expansion of the translated sigmoid for regression loss")

    parser.add_argument("--most_significant_prototypes", action="store_true", help="Select NNK most significant samples for each class as prototypes")
    parser.add_argument("--knn_distance_evaluator", action="store_true", help="Use similarity with nearest K train samples as normality score")
    parser.add_argument("--NNK", default=1, type=int, help="K for knn distance evaluator")

    args = parser.parse_args()

    dataset = args.dataset

    if dataset == "ImageNet":
        args.known_classes = 1000
        args.tot_classes = 1000

        # eval dataset is DomainNet_IN_OUT
        args.eval_known_classes = 25
    elif dataset == "OfficeHome_DG":
        args.known_classes = 54
        args.tot_classes = 65
        args.source = f"no_{args.target}"
    elif dataset == "PACS_DG" or dataset=='PACS_SS_DG':
        args.known_classes = 6
        args.tot_classes = 7
        if dataset == "PACS_DG":
            args.source = f"no_{args.target}"
    elif dataset == "OfficeHome_SS_DG":
        args.known_classes = 25
        args.tot_classes = 65
    elif dataset == "MultiDatasets_DG":
        args.known_classes = 48
        args.tot_classes = 68
    elif dataset == "DTD":
        args.known_classes = 23
        args.tot_classes = 47
    elif "DomainNet" in dataset or dataset == "Places":
        args.known_classes = 25
        args.tot_classes = 50
    elif dataset == "imagenet_ood":
        args.known_classes = 500
        args.tot_classes = 1000
    elif dataset == "imagenet_ood_small":
        args.known_classes = 25
        args.tot_classes = 50
    else:
        raise NotImplementedError(f"Unknown dataset {dataset}")

    if os.path.isdir("/scratch/ImageNet"):
        args.imagenet_path_dataset = os.path.expanduser('/scratch/')
        args.executing_from_scratch = True
    else:
        args.imagenet_path_dataset = os.path.expanduser('~/data/')
        args.executing_from_scratch = False

    args.path_dataset = os.path.expanduser('~/data/')
    print(f"Loading data from: {args.path_dataset}")
    return args


class Trainer:
    def __init__(self, args, device, folder_name):
        self.args = args
        self.device = device
        self.args.device = device
        self.args.folder_name = folder_name
        self.folder_name = folder_name

        # prepare model
        # feat
        self.feature_extractor = ResNetFc(self.device, self.args.network)
        self.output_num = self.feature_extractor.output_num()
        if self.args.loss_function == "L2":
            num_classes = 1
        elif self.args.loss_function == "CE":
            num_classes = 2

        # rel module
        from models.relational_transformer import RelationalTransformer

        self.cls_rel = RelationalTransformer(self.output_num, num_classes=num_classes, depth=args.transf_depth,
                                             num_heads=args.transf_n_heads)

        print(f"Feature extractor params: {count_parameters(self.feature_extractor)}")
        print(f"Relational module params: {count_parameters(self.cls_rel)}")
        print(f"Head params: {count_parameters(self.cls_rel.head)}")

        # when training on ImageNet we test on DomainNet Real 
        if args.dataset == "ImageNet":
            self.target_loader, self.source_loader_test = get_val_dataloader(self.args, eval_dataset = "DomainNet_IN_OUT")
        else:
            self.target_loader, self.source_loader_test = get_val_dataloader(self.args)

        self.models = {
            "feature_extractor": self.feature_extractor,
            "cls_rel": self.cls_rel}

        if not self.args.pretrained is None:
            load_ckpt(self.models, self.args.pretrained)
            print("Loaded pretrained module")

        self.to_device(device)

        # resume if necessary
        if args.resume:
            if check_resume(folder_name):
                self.start_it = resume(self.models, folder_name)
            else:
                print("Cannot resume training, starting from 0")
                self.start_it = 0
        else:
            self.start_it = 0

        # move to cuda
        if self.args.loss_function == "BCE":
            self.criterion = nn.functional.binary_cross_entropy_with_logits
        elif self.args.loss_function == "CE":
            if self.args.rebalance_loss:
                loss_weights = torch.tensor([self.args.neg_to_pos_ratio, 1]).float()
                print("Loss rebalancing enabled with weights: ", loss_weights)
                self.criterion = nn.CrossEntropyLoss(weight=loss_weights).to(device)
            else:
                self.criterion = nn.CrossEntropyLoss().to(device)
        elif self.args.loss_function in ["L2", "L1", "LX"]:
            pass
        else:
            raise NotImplementedError(f"Unknown learning objective: {self.args.loss_function}")


        num_its = self.args.iterations
        step_after = self.args.step_after
        train_modules = [self.feature_extractor, self.cls_rel]
        self.optimizer, self.scheduler = get_optim_and_scheduler(train_modules, self.args, num_its, step_after, self.start_it, warmup_its=self.args.warmup)
        if args.distributed:
            self.optimizer = LARS(self.optimizer)

        # move to distributed
        if args.distributed:
            self.feature_extractor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.feature_extractor)
            self.models["feature_extractor"] = self.feature_extractor
            self.feature_extractor = DDP(self.feature_extractor, device_ids=[self.args.local_rank],
                                         find_unused_parameters=True)
            self.cls_rel = DDP(self.cls_rel, device_ids=[self.args.local_rank], find_unused_parameters=True)

    def _do_iteration(self, log=False):
        self.optimizer.zero_grad()

        try:
            data1, data2, _, relation_l = next(self.source_iter)
        except StopIteration:
            print('New training file')
            self.to_eval()
            self.source_loader = get_train_dataloader(self.args, self.folder_name)

            self.source_iter = iter(self.source_loader)
            data1, data2, _, relation_l = next(self.source_iter)
            self.to_train()

        data1, data2, relation_l = data1.to(self.device), data2.to(self.device), relation_l.to(self.device)

        # forward
        batch_size = data1.shape[0]
        data_tot = torch.cat((data1, data2))
        data_tot = self.feature_extractor(data_tot)
        data1_feat = data_tot[:batch_size]
        data2_feat = data_tot[-batch_size:]

        data12_aggregation = torch.cat((data1_feat, data2_feat), 1)

        # compute relation
        relation_logit = self.cls_rel(data12_aggregation).squeeze()

        # loss computation
        if self.args.loss_function == "L2":
            # prepare regression targets: (-1) for same class, (1) for diff class
            total_range = self.args.sigmoid_expansion
            half_range = total_range / 2
            reg_targets = half_range*torch.ones_like(relation_l).float()
            reg_targets[~relation_l.bool()] = -half_range

            # move network output in [-half_range, half_range] range
            reg_outputs = total_range*(torch.sigmoid(self.args.sigmoid_compression*relation_logit)) - half_range

            # compute the loss
            relation_loss = F.mse_loss(reg_outputs, reg_targets, reduction='mean')
            relation_pred = (reg_outputs > 0).float() # if > 0 it means predict diff class

        elif self.args.loss_function == "CE":
            relation_loss = self.criterion(relation_logit, relation_l)
            with torch.no_grad():
                # class prediction
                _, relation_pred = relation_logit.max(dim=1)

        loss = relation_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # we compute relation accuracy separately per each class and then output the avg
        acc_count = 0
        if torch.sum(relation_l==0) > 0:
            cls_0_acc = torch.sum((relation_pred[relation_l==0] == relation_l[relation_l==0]))/torch.sum(relation_l==0) # class 0 = same class
            cls_0_acc_r = cls_0_acc
            acc_count += 1
        else: 
            cls_0_acc = torch.tensor(0)
            cls_0_acc_r = torch.tensor(-1)

        if torch.sum(relation_l==1) > 0:
            cls_1_acc = torch.sum((relation_pred[relation_l==1] == relation_l[relation_l==1]))/torch.sum(relation_l==1) # class 1 = diff class
            cls_1_acc_r = cls_1_acc
            acc_count += 1
        else:
            cls_1_acc = torch.tensor(0)
            cls_1_acc_r = torch.tensor(-1)

        return relation_loss.item(), (cls_0_acc.item()+cls_1_acc.item())/acc_count, cls_0_acc_r.item(), cls_1_acc_r.item()

    def to_device(self, device):
        self.feature_extractor = self.feature_extractor.to(device)
        self.cls_rel = self.cls_rel.to(device)

    def to_eval(self):
        self.feature_extractor.eval()
        self.cls_rel.eval()

    def to_train(self):
        self.feature_extractor.train()
        self.cls_rel.train()

    @torch.no_grad()
    def do_final_eval(self, known_classes=None):

        if self.args.most_significant_prototypes:
            prototypes = self.compute_most_significant_prototypes(self.source_loader_test, n_proto=self.args.NNK, known_classes=known_classes, log=False)
            
            auroc = do_test_target_with_prototypes_new(self.args, self.models, prototypes, self.target_loader, known_classes=known_classes)

        elif self.args.knn_distance_evaluator:
            auroc = do_knn_distance_eval(self.args, self.models, self.source_loader_test, self.target_loader, known_classes=known_classes)
        else:

            if known_classes is None:
                known_classes = self.args.known_classes
            self.to_eval()
            prototypes = self.compute_source_prototypes(self.source_loader_test, known_classes=known_classes, log=False)
            print('Prototypes evaluation')
            auroc = do_test_target_with_prototypes(self.args, self.models, prototypes, self.target_loader, known_classes=known_classes)
        return auroc

    def compute_source_prototypes(self, sources_loader, known_classes=None, log=False, return_feats = False):
        if known_classes is None:
            known_classes = self.args.known_classes
        # prepare structures to hold prototypes
        self.to_eval()
        prototypes = np.zeros((known_classes, self.output_num), dtype=np.float32)

        features = {}
        labels = {}

        # forward source data
        for it_s, (data_s, class_l_s, indices) in enumerate(tqdm(sources_loader)):
            # forward
            data_s, class_l_s = data_s.to(self.device), class_l_s.to(self.device)
            data_s_feat = self.feature_extractor.forward(data_s)

            for f, l, i in zip(data_s_feat, class_l_s, indices):
                features[i.item()] = f.cpu()
                labels[i.item()] = l.cpu()

        source_features = {}
        source_labels = {}

        if self.args.distributed:
            all_feats = all_gather(features)  # returns a list of dicts
            all_labels = all_gather(labels)
            for dic in all_feats:
                source_features.update(dic)
            for dic in all_labels:
                source_labels.update(dic)
        else:
            source_features = features
            source_labels = labels

        source_features = torch.stack([source_features[k] for k in sorted(source_features.keys())])
        source_labels = torch.tensor([source_labels[k] for k in sorted(source_labels.keys())])

        # derive prototypes
        for category in tqdm(range(0, known_classes)):
            mask = source_labels == category
            feats = source_features[mask]
            prototype = feats.mean(0)
            prototypes[category] = prototype

        if return_feats: 
            return prototypes, source_features, source_labels

        return prototypes

    def compute_most_significant_prototypes(self, sources_loader, n_proto=5, known_classes=None, log=False):
        # the most significant prototype for each class is the nearest sample to the feats mean
        feats_prototypes, src_feats, src_labels = self.compute_source_prototypes(sources_loader, known_classes=known_classes, log=log, return_feats=True)
        feats_prototypes = torch.tensor(feats_prototypes)

        per_class_significant_feats = []
         

        if not self.args.distributed or self.args.global_rank == 0:
        
            for idx in range(len(feats_prototypes)):
                proto = feats_prototypes[idx]
                class_mask = src_labels == idx
                cls_feats = src_feats[class_mask]

                nearest_idx = torch.norm(cls_feats - proto, dim=1).argmin()

                # put the nearest sample in the first position
                cls_feats[[0,nearest_idx],:] = cls_feats[[nearest_idx,0],:]

                # apply coreset
                most_significant_ids = get_coreset_idx(cls_feats, n=n_proto)
                per_class_significant_feats.append(cls_feats[most_significant_ids])

        if self.args.distributed:

            all_sign_feats = all_gather(per_class_significant_feats)

            for feat_list in all_sign_feats:
                if len(feat_list) > 0: 
                    per_class_significant_feats = feat_list
                    break

        return per_class_significant_feats

    def do_training(self):
        # prepare eval data
        self.to_eval()

        if self.args.only_eval:
            load_ckpt(self.models, self.args.checkpoint_folder_path)
            self.do_final_eval()
            exit()

        # prepare train data
        self.source_loader = get_train_dataloader(self.args, self.folder_name)
        self.len_dataloader = len(self.source_loader)

        print("Source %s , Target %s" % (self.args.source, self.args.target))
        print("Dataset size: train %d, test %d" % (
        len(self.source_loader.dataset), len(self.target_loader.dataset)))

        tot_relation_loss, tot_relation_acc, tot_cls_0, cls_0_count, tot_cls_1, cls_1_count = 0, 0, 0, 0, 0, 0

        self.source_iter = iter(self.source_loader)

        log_period = 10
        save_period = 500
        self.to_train()

        self.current_iteration = self.start_it

        start_training = time.time()

        for self.current_iteration in range(self.start_it, self.args.iterations):
            self.args.current_iteration = self.current_iteration

            # perform train iter
            relation_loss, relation_acc, cls_0_acc, cls_1_acc = self._do_iteration(log=self.current_iteration % log_period == 0)
            self.scheduler.step()

            tot_relation_loss += relation_loss
            tot_relation_acc += relation_acc
            if cls_0_acc >= 0:
                tot_cls_0 += cls_0_acc
                cls_0_count += 1
            if cls_1_acc >= 0:
                tot_cls_1 += cls_1_acc
                cls_1_count += 1

            if self.current_iteration % log_period == 0 and self.current_iteration > 0:
                tot_iter = log_period

                iter_time_avg = (time.time()-start_training) / (self.current_iteration - self.start_it + 1)
                
                eta_sec = (self.args.iterations - self.current_iteration)*iter_time_avg
                eta_hour = eta_sec // 3600
                eta_sec = eta_sec % 3600
                eta_min = eta_sec // 60
                eta_sec = eta_sec % 60

                current_lr = self.optimizer.param_groups[0]['lr']
                used_cuda_mem = torch.cuda.memory_allocated()/(math.pow(2,20))

                if cls_0_count > 0:
                    acc_cls_0 = tot_cls_0 / cls_0_count
                else:
                    acc_cls_0 = -1
                if cls_1_count > 0:
                    acc_cls_1 = tot_cls_1 / cls_1_count
                else:
                    acc_cls_1 = -1

                print("[%s] [Iter %3d] [Avg time %.2fs] [ETA %02dh%02dm%02ds] [LR %.8f] [Avg rel Loss %.6f] [Avg rel acc %.6f] [CLS_0 acc %.6f] [CLS_1 acc %.6f] [MEM %.6f MiB]" % (\
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            self.current_iteration,
                            iter_time_avg,
                            eta_hour, eta_min, eta_sec, 
                            current_lr,
                            tot_relation_loss / tot_iter, 
                            tot_relation_acc / tot_iter,
                            acc_cls_0,
                            acc_cls_1,
                            used_cuda_mem))

                tot_relation_loss, tot_relation_acc, tot_cls_0, cls_0_count, tot_cls_1, cls_1_count = 0, 0, 0, 0, 0, 0
            if self.current_iteration > 0 and self.current_iteration%save_period == 0:
                if not self.args.distributed or self.args.global_rank == 0:
                    save_ckpt(self.models, self.folder_name, self.current_iteration)

            if self.current_iteration % 500 == 0 and self.current_iteration > 0:
                self.to_eval()
                with torch.no_grad():
                    auroc = self.do_final_eval(known_classes=self.args.eval_known_classes)
                self.to_train()

        self.to_eval()
        self.do_final_eval(known_classes=self.args.eval_known_classes)


def main():
    args = get_args()
    ### Set torch device ###
    if torch.cuda.is_available():
        if hasattr(args, 'local_rank') and not args.local_rank is None:
            assert False, "Please use torchrun for distributed execution"

        if "LOCAL_RANK" in environ:
            args.local_rank = int(environ["LOCAL_RANK"])
            args.distributed = True
            torch.cuda.set_device(args.local_rank)
        else:
            args.distributed = False
            args.n_gpus = 1

        device = torch.device("cuda")
    else:
        print("WARNING. Running in CPU mode")
        args.distributed = False
        device = torch.device("cpu")

    if args.distributed:
        torch.distributed.init_process_group(
            'nccl',
            init_method='env://',
        )
        # get world size from torch distributed
        args.n_gpus = torch.distributed.get_world_size()
        # global rank identifies process on multiple nodes
        # local rank on a single node. If this is a single node
        # training they should be the same
        args.global_rank = int(os.environ['RANK'])
        print("Process rank", args.global_rank, "starting")

    if args.only_eval:
        output_name = args.checkpoint_folder_path
        assert os.path.isdir(f"{output_name}"), "Cannot perform eval! Checkpoint path does not exist!"
        output_txt = "eval_"+args.dataset+".txt"
        folder_name = output_name
    else:
        output_name = f"{args.dataset}_{args.target}_{args.network}_rel_transformer"
        if args.target is None: 
            output_name = f"{args.dataset}_{args.network}_rel_transformer"
        if not args.suffix == "":
            output_name = f"{output_name}_{args.suffix}"

        folder_name = args.checkpoint_folder_path + '/' + output_name
        # if path already exists append random

        create = False
        if not args.distributed or args.global_rank == 0:
            if os.path.exists(folder_name):
                if args.resume:
                    print("Launching in resume mode")
                else:
                    rand = np.random.randint(200000)
                    print(f"Folder {folder_name} already exists. Appending random: {rand}")
                    folder_name = f"{folder_name}_{rand}"
                    create = True
            else:
                create = True
        else:
            folder_name = ""
        if create:
            os.makedirs(folder_name)

        if args.distributed:
            folder_name_list = all_gather(folder_name)
            folder_name = folder_name_list[0]
            print(f"Rank: {args.global_rank}: folder name: {folder_name}")

        output_txt = "out.txt"

    # print on both log file and stdout
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    if not args.distributed or args.global_rank == 0:
        f = open(folder_name + '/' + output_txt, 'a')
        sys.stdout = LogUnbuffered(args, orig_stdout, f)
        f1 = open(folder_name + '/stderr.txt', 'a')
        sys.stderr = LogUnbuffered(args, orig_stderr, f1)

    if args.distributed:
        print(f"Total number of processes: {args.n_gpus}")

    args.output_folder = folder_name
    print(args)
    torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(args, device, folder_name)
    trainer.do_training()

    # restore stdout and close log file
    sys.stdout = orig_stdout
    sys.stderr = orig_stderr

    if not args.distributed or args.global_rank == 0:
        f.close()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    main()
