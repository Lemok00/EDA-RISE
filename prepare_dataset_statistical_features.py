import argparse
import torch
import os
import warnings
from torchvision import transforms as TF
from torch.utils.data import DataLoader
import numpy as np

from dataset import set_dataset
from inception import InceptionV3

warnings.simplefilter('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--dataset_type', choices=['official_lmdb', 'prepared_lmdb', 'image_folder'])
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--save_result_dir', type=str, default='./dataset_statistics/')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_tqdm', action='store_true')
    args = parser.parse_args()

    device = args.device

    '''
    Prepare tqdm
    '''
    if args.use_tqdm:
        from tqdm import tqdm
    else:
        def tqdm(x):
            return x

    '''
    Path for saving results
    '''
    result_dir = args.save_result_dir
    result_npz_path = os.path.join(result_dir, f'{args.dataset_name}.npz')
    os.makedirs(result_dir, exist_ok=True)

    '''
    Load InceptionV3 model
    '''
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inception = InceptionV3([block_idx]).to(device)
    inception.eval()

    '''
    Set dataset
    '''
    transforms = TF.Compose([TF.Resize((299, 299)), TF.ToTensor()])
    dataset = set_dataset(args.dataset_type, args.dataset_path, transforms)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            drop_last=True, num_workers=4)

    '''
    Get activations
    '''
    pred_array = np.empty((len(dataset), 2048))
    start_idx = 0
    for batch in tqdm(dataloader):
        batch = batch.to(device)
        with torch.no_grad():
            pred = inception(batch)[0]
        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_array[start_idx:start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    print(pred_array.shape)

    '''
    Calculate activation statistics
    '''
    mu = np.mean(pred_array, axis=0)
    sigma = np.cov(pred_array, rowvar=False)

    '''
    Save statistical features
    '''

    np.savez(result_npz_path, mu=mu, sigma=sigma)
