import os.path
from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
from glob import glob


def set_dataset(type, path, transform):
    if type == 'official_lmdb':
        datatype = OfficialLmdbDataset
    elif type == 'prepared_lmdb':
        datatype = PreparedLmdbDataset
    elif type == 'image_folder':
        datatype = ImageFolderDataset
    else:
        raise NotImplementedError
    return datatype(path, transform)


class PreparedLmdbDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(path, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)
        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class OfficialLmdbDataset(Dataset):
    def __init__(self, path, transform):
        self.env = lmdb.open(path, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        self.keys = []
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for idx, (key, _) in enumerate(cursor):
                self.keys.append(key)

        self.length = len(self.keys)
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


class ImageFolderDataset(Dataset):
    def __init__(self, path, transform):
        self.EXTs = ['webp', '.png', '.jpg', '.jpeg', '.ppm', '.bmp', '.pgm', '.tif', '.tiff']
        self.files = []
        for file in sorted(list(glob(f'{path}/*'))):
            if any(file.lower().endswith(ext) for ext in self.EXTs):
                self.files.append(file)

            self.transform = transform
            self.length = len(self.files)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img = Image.open(self.files[index]).convert('RGB')
        img = self.transform(img)

        return img
