import argparse
import torch
import os
import warnings
from torchvision.utils import save_image

from models import Generator, StructureGenerator
from mapping_functions import message_to_tensor
from llcs import llcs

warnings.simplefilter('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_result_dir', type=str, default='./synthesised_samples/')
    parser.add_argument('--sigma', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
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
    ckpt_path = args.checkpoint_path
    ckpt_name = '.'.join(ckpt_path.split('/')[-1].split('.')[:-1])
    result_dir = os.path.join(args.save_result_dir, f'synthesised_{ckpt_name}_samples')
    os.makedirs(result_dir, exist_ok=True)

    '''
    Load model
    '''
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    ckpt_args = ckpt['args']

    generator = Generator(ckpt_args.channel).to(device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()

    structure_generator = StructureGenerator(ckpt_args.channel, ckpt_args.N).to(device)
    structure_generator.load_state_dict(ckpt['structure_generator'])
    structure_generator.eval()

    print(f'Model loaded from: {ckpt_path}')

    '''
    Testing codes 
    '''
    test_num, batch_size, sigma = args.test_num, args.batch_size, args.sigma
    image_size, tensor_size, texture_size = 256, 16, 2048
    hidden_capacity = ckpt_args.N * tensor_size * tensor_size * sigma

    secret_messages = torch.randint(low=0, high=2, size=(test_num, hidden_capacity))
    secret_keys = torch.rand(size=(test_num,), dtype=torch.float64).to(device)
    textures = torch.rand(size=(test_num, 2048)).to(device) * 2 - 1

    # encrypt the secret message
    encrypted_messages = llcs(secret_messages, secret_keys, hidden_capacity)
    secret_tensors = message_to_tensor(encrypted_messages, sigma)

    with torch.no_grad():
        for i in tqdm(range(test_num // batch_size)):
            # synthesise a container image
            secret_tensor = secret_tensors[i * batch_size:(i + 1) * batch_size] \
                .view(-1, ckpt_args.N, tensor_size, tensor_size).to(device)
            structure = structure_generator(secret_tensor)
            texture = textures[i * batch_size:(i + 1) * batch_size]
            container_image = generator(structure, texture)

            # save the synthesised image
            for j in range(batch_size):
                save_path = os.path.join(result_dir, f'{i * batch_size + j + 1:05d}.png')
                save_image(container_image[0],
                           save_path,
                           normalize=True,
                           range=(-1, 1))
