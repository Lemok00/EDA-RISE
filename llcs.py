import torch


def llcs(message, key, length, alpha=5, beta=14, tau=100):
    # generate a chaotic sequence
    chaotic_sequence = torch.zeros(size=(key.shape[0], tau + length + 1))
    chaotic_sequence[:, 0] = h = key

    for i in range(1, tau + length + 1):
        temp = alpha * h * (1 - h) * 2 ** beta
        chaotic_sequence[:, i] = h = temp - torch.floor(temp)

    chaotic_sequence = chaotic_sequence[:, tau + 1:]

    # binarise the chaotic sequence
    average_value = torch.mean(chaotic_sequence, dim=1).view(-1, 1).repeat(1, length)
    binarised_sequence = torch.ceil(chaotic_sequence - average_value)

    # encrypt / decrypt the message by XOR operation
    xor_message = torch.abs(binarised_sequence - message)

    return xor_message

if __name__ == '__main__':
    message = torch.randint(low=0, high=2, size=(2, 4))
    key = torch.rand(size=(2,), dtype=torch.float64)
    print('Original message:\n', message)
    print('Secret key:\n', key)
    encrypted_message = llcs(message, key, length=4, tau=2)
    print('Encrypted message:\n', encrypted_message)
    decrypted_message = llcs(encrypted_message, key, length=4, tau=2)
    print('Decrypted message:\n', decrypted_message)
