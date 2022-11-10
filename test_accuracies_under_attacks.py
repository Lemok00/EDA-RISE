import argparse
import torch
import os
import pandas as pd
import warnings

from models import Encoder, Generator, StructureGenerator, Extractor
from mapping_functions import message_to_tensor, tensor_to_message
from llcs import llcs
from attacks import (BaseAttack,
                     GaussianNoiseAttacker, SaltAndPepperNoiseAttacker,
                     JPEGCompressionAttacker, WebPCompressionAttacker,
                     GaussianBlurAttacker, AveragingBlurAttacker,
                     BrightnessAdjustAttacker, ContrastAdjustAttacker,
                     RotationAttacker, EdgeCroppingAttacker, CenterCroppingAttacker,
                     ResizingAttacker, ShearingAttacker)

warnings.simplefilter('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_result_dir', type=str, default='./accuracy_results/')
    parser.add_argument('--sigma', type=int, default=1)
    parser.add_argument('--test_num', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_tqdm', action='store_true')
    parser.add_argument('--use_llcs', action='store_true')
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
    result_dir = args.save_result_dir
    os.makedirs(result_dir, exist_ok=True)
    result_csv_path = os.path.join(result_dir, f'test_{ckpt_name}_accuracy.csv')

    '''
    Setting image attacks with different intensities
    '''
    image_attacks = [
        # singal processing attacks
        (BaseAttack(), ['']),
        (GaussianNoiseAttacker(), [0.05, 0.075, 0.1]),
        (SaltAndPepperNoiseAttacker(), [0.05, 0.075, 0.1]),
        (JPEGCompressionAttacker(), [90, 70, 50]),
        (WebPCompressionAttacker(), [90, 70, 50]),
        (GaussianBlurAttacker(), [3, 5, 7]),
        (AveragingBlurAttacker(), [3, 5, 7]),
        # geometry attacks
        (RotationAttacker(), [2, 3, 4, 5, 6, 7, 8, 9, 10]),
        (EdgeCroppingAttacker(), [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01]),
        (CenterCroppingAttacker(), [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.01]),
        (ResizingAttacker(), [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]),
        (ShearingAttacker(), [2, 3, 4, 5, 6, 7, 8, 9, 10]),
    ]

    '''
    Initialize a dict to save accuracy results
    '''
    accuracy_dict = {}
    num_attacks = 0
    for idx, (attack, intensities) in enumerate(image_attacks):
        accuracy_dict[attack.get_attack_name()] = {}
        for intensity in intensities:
            accuracy_dict[attack.get_attack_name()][attack.get_intensity_name(intensity)] = []
            num_attacks += 1

    '''
    Load model
    '''
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    ckpt_args = ckpt['args']

    robust_encoder = Encoder(ckpt_args.channel).to(device)
    robust_encoder.load_state_dict(ckpt['robust_encoder'])
    robust_encoder.eval()

    generator = Generator(ckpt_args.channel).to(device)
    generator.load_state_dict(ckpt['generator'])
    generator.eval()

    structure_generator = StructureGenerator(ckpt_args.channel, ckpt_args.N).to(device)
    structure_generator.load_state_dict(ckpt['structure_generator'])
    structure_generator.eval()

    robust_extractor = Extractor(ckpt_args.channel, ckpt_args.N).to(device)
    robust_extractor.load_state_dict(ckpt['robust_extractor'])
    robust_extractor.eval()

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

    with torch.no_grad():
        for i in tqdm(range(test_num // batch_size)):
            secret_message = secret_messages[i * batch_size:(i + 1) * batch_size]
            secret_key = secret_keys[i * batch_size:(i + 1) * batch_size]
            encrypted_message = encrypted_messages[i * batch_size:(i + 1) * batch_size]

            # synthesis a container image
            secret_tensor = message_to_tensor(encrypted_message, sigma) \
                .view(-1, ckpt_args.N, tensor_size, tensor_size).to(device)
            structure = structure_generator(secret_tensor)
            texture = textures[i * batch_size:(i + 1) * batch_size]
            container_image = generator(structure, texture)

            # record the extracted messages under different image attacks
            # for decrypting them in parallel to decrease the testing time
            extracted_messages = []
            for attack, intensities in image_attacks:
                for intensity in intensities:
                    # tamper the container image using different image attacks
                    attacked_container_image = attack.attack(container_image.clone(), intensity)
                    # extract the hidden message from the attacked image
                    extracted_tensor = robust_extractor(robust_encoder(attacked_container_image)[0])
                    extracted_tensor = extracted_tensor.view(batch_size, -1)
                    extracted_message = tensor_to_message(extracted_tensor, sigma)
                    extracted_messages.append(extracted_message)

            # decrypt the extracted messages under various attacks in parallel
            extracted_messages = torch.cat(extracted_messages, dim=0)
            decrypted_messages = llcs(extracted_messages, secret_key.repeat(num_attacks), hidden_capacity)
            decrypted_messages = torch.chunk(decrypted_messages, chunks=num_attacks)

            att_idx = 0
            for attack, intensities in image_attacks:
                for intensity in intensities:
                    acc = 1 - torch.mean(torch.abs(decrypted_messages[att_idx] - secret_message)).item()
                    accuracy_dict[attack.get_attack_name()][attack.get_intensity_name(intensity)].append(acc)
                    att_idx += 1

    '''
    Record accuracy results
    '''
    attack_name_list = []
    accuracy_list = []
    for attack, intensities in image_attacks:
        for intensity in intensities:
            attack_name_list.append(f'{attack.get_attack_name()} {attack.get_intensity_name(intensity)}')
            accuracy_under_attack = accuracy_dict[attack.get_attack_name()][attack.get_intensity_name(intensity)]
            accuracy_list.append(sum(accuracy_under_attack) / len(accuracy_under_attack))

    dataframe = pd.DataFrame()
    dataframe['Attack Name'] = attack_name_list
    dataframe['Accuracy'] = accuracy_list
    dataframe.to_csv(result_csv_path)
