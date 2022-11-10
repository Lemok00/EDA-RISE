import torch


def message_to_tensor(message, sigma):
    secret_tensor = torch.zeros(size=(message.shape[0], message.shape[1] // sigma))
    message_nums = torch.zeros_like(secret_tensor)
    for i in range(sigma):
        message_nums += message[:, i::sigma] * 2 ** (sigma - i - 1)
    secret_tensor = (message_nums + 0.5) / 2 ** (sigma - 1) - 1
    return secret_tensor


def tensor_to_message(secret_tensor, sigma):
    message = torch.zeros(size=(secret_tensor.shape[0], secret_tensor.shape[1] * sigma))
    secret_tensor = torch.clamp(secret_tensor, min=-1, max=1)
    message_nums = torch.floor((secret_tensor + 1) * 2 ** (sigma - 1))
    zeros = torch.zeros_like(message_nums)
    ones = torch.ones_like(message_nums)
    for i in range(sigma):
        zero_one_map = torch.where(message_nums >= 2 ** (sigma - i - 1), ones, zeros)
        message[:, i::sigma] = zero_one_map
        message_nums -= zero_one_map * 2 ** (sigma - i - 1)
    return message
