from os.path import join, dirname
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform
from timm.data.random_erasing import RandomErasing
from PIL import Image,ImageFile
from random import sample
from models.create_pairs import create_pairs_source_v2
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names1 = []
    file_names2 = []
    labels_obj = []
    labels_rel = []
    for row in images_list:
        row = row.split(' ')
        file_names1.append(row[0])
        file_names2.append(row[1])
        labels_obj.append(row[2])
        labels_rel.append(int(row[3]))

    return file_names1, file_names2, labels_obj, labels_rel


def _dataset_info_from_list(pairs):
    images_list = pairs

    file_names1 = []
    file_names2 = []
    labels_obj = []
    labels_rel = []
    for row in images_list:
        row = row.split(' ')
        file_names1.append(row[0])
        file_names2.append(row[1])
        labels_obj.append(row[2])
        labels_rel.append(int(row[3]))

    return file_names1, file_names2, labels_obj, labels_rel


def _dataset_info_standard(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []

    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


class Dataset(data.Dataset):
    def __init__(self, names1,names2,labels_obj, labels_rel, path_dataset,img_transformer=None):
        self.data_path = path_dataset
        self.names1 = names1
        self.names2 = names2
        self.labels_obj = labels_obj
        self.labels_rel = labels_rel
        self._image_transformer = img_transformer


    def __getitem__(self, index):
        framename1 = self.data_path + '/' + self.names1[index]
        img1 = Image.open(framename1).convert('RGB')

        framename2 = self.data_path + '/' + self.names2[index]
        img2 = Image.open(framename2).convert('RGB')

        img1 = self._image_transformer(img1)
        img2 = self._image_transformer(img2)

        return img1,img2,int(self.labels_obj[index]), int(self.labels_rel[index])

    def __len__(self):
        return len(self.names1)

class EvalDataset(data.Dataset):
    def __init__(self, names,labels, path_dataset,img_transformer=None,base_index=0):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self.base_index = base_index
        self._image_transformer = img_transformer

    def __getitem__(self, index):

        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        img = self._image_transformer(img)

        return img, int(self.labels[index]), index+self.base_index

    def __len__(self):
        return len(self.names)

def get_train_dataloader(args,folder_name):

    enable_class_balancing = args.class_balancing
    path_dataset = args.path_dataset

    if args.dataset == "ImageNet":
        source = "train"
        path_dataset = args.imagenet_path_dataset
    else:
        raise NotImplementedError(f"Dataset {args.dataset} unknown")

    path_to_source_txt = args.path_to_txt + args.dataset +'/'+ source + '.txt'
    pairs = create_pairs_source_v2(path_to_source_txt,
            enable_class_balancing=enable_class_balancing,
            neg_to_pos_ratio=args.neg_to_pos_ratio)

    img_transformer = get_train_transformers(args)

    name1_train,  name2_train, labels_obj, labels_rel = _dataset_info_from_list(pairs)

    train_dataset = Dataset(name1_train,name2_train,labels_obj, labels_rel, path_dataset, img_transformer=img_transformer)
    dataset = train_dataset
    if args.distributed:
        sampler = DistributedSampler(dataset=dataset, shuffle=True)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=sampler, drop_last=True)
    else:
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    return loader

def get_val_dataloader(args, eval_dataset=None):

    if eval_dataset is None: 
        eval_dataset = args.dataset
    assert not eval_dataset == "ImageNet", "ImageNet eval not implemented"

    dataset = eval_dataset
    target = args.target
    source = args.source

    if dataset == "DomainNet_IN_OUT":
        source = "in_distribution"
        target = "out_distribution"
    elif dataset == 'MultiDatasets_DG':
        source = "Sources"

    dataset_tgt_path = args.path_to_txt + '/' + dataset + '/' + target +'.txt'
    dataset_srcs_path = args.path_to_txt + '/' + dataset + '/' + source + '.txt'

    names_target,labels_target = _dataset_info_standard(dataset_tgt_path)
    names_sources, labels_sources = _dataset_info_standard(dataset_srcs_path)

    img_tr = get_val_transformer(args)
    path_dataset = args.path_dataset

    if "DomainNet" in dataset: 
        path_dataset = path_dataset + '/DomainNet'

    sources_dataset = EvalDataset(names_sources,labels_sources,path_dataset, img_transformer=img_tr)
    target_dataset = EvalDataset(names_target,labels_target,path_dataset, img_transformer=img_tr)

    if args.distributed:
        target_sampler = DistributedSampler(dataset=target_dataset, shuffle=False)
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=1, num_workers=0, pin_memory=True, sampler=target_sampler, drop_last=False)
        sources_sampler = DistributedSampler(dataset=sources_dataset, shuffle=False)
        sources_loader = torch.utils.data.DataLoader(sources_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True, sampler=sources_sampler, drop_last=False)
    else:
        target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False) 
        sources_loader = torch.utils.data.DataLoader(sources_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)

    return target_loader, sources_loader

def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]

    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))

    if args.jitter > 0.0:
        transform_jitter = transforms.RandomApply(torch.nn.ModuleList(
            [transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=0.1)]),
                                                  p=args.prob_jitter)
        img_tr.append(transform_jitter)
    if args.random_grayscale:
        img_tr.append(transforms.RandomGrayscale(args.random_grayscale))

    img_tr = img_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)


def get_val_transformer(args):

    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr)
