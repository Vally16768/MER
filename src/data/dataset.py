from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import pickle
import torchaudio
import subprocess
import torch
import os
from tqdm import tqdm
from typing import Any, Iterable

def convert_mp4_to_mp3(path, sampling_rate=16000):

    path_save = path[:-3] + 'wav'
    if not os.path.exists(path_save):
        ff_audio = 'ffmpeg -i {} -vn -acodec pcm_s16le -ar 44100 -ac 2 {}'.format(path, path_save)
        subprocess.call(ff_audio, shell = True)
    wav, sr = torchaudio.load(path_save)

    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != sampling_rate:
        transform = torchaudio.transforms.Resample(orig_freq=sr,
                                                    new_freq=sampling_rate)
        wav = transform(wav)
        sr = sampling_rate

    assert sr == sampling_rate
    return wav.squeeze(0)

def img_processing(fp):
    class PreprocessInput(torch.nn.Module):
        def init(self):
            super(PreprocessInput, self).init()

        def forward(self, x):
            x = x.to(torch.float32)
            x = torch.flip(x, dims=(0,))
            x[0, :, :] -= 91.4953
            x[1, :, :] -= 103.8827
            x[2, :, :] -= 131.0912
            return x

    def get_img_torch(img, target_size=(224, 224)):
        transform = transforms.Compose([transforms.PILToTensor(), PreprocessInput()])
        img = img.resize(target_size, Image.Resampling.NEAREST)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        return img

    return get_img_torch(fp)

def pad_sequence(faces, max_length):
    current_length = faces.shape[0]

    if current_length < max_length:
        repetitions = (max_length + current_length - 1) // current_length
        faces = torch.cat([faces] * repetitions, dim=0)[:max_length, ...]
        
    elif current_length > max_length:
        faces = faces[:max_length, ...]

    return faces

def pad_wav(wav, max_length):
    current_length = len(wav)

    if current_length < max_length:
        repetitions = (max_length + current_length - 1) // current_length
        wav = torch.cat([wav]*repetitions, dim=0)[:max_length]
    elif current_length > max_length:
        wav = wav[:max_length]

    return wav

class AVTDataset(Dataset):
    def __init__(self, feature_paths = ''):
        self.feature_paths = feature_paths
    
    def load_data(self, filename):
        with open(filename, 'rb') as handle:
            meta = pickle.load(handle)
        return meta
        
    def get_meta(self, path):
        meta = self.load_data(path)

        return meta['video_features'], meta['audio_features'], meta['text_features'], meta['true_label']

    def __getitem__(self, index):
        v_f, a_f, t_f, l = self.get_meta(self.feature_paths[index])
        return torch.FloatTensor(a_f.cpu()[0]), torch.FloatTensor(v_f.cpu()[0]), torch.FloatTensor(t_f.cpu()[0]), l
            
    def __len__(self):
        return len(self.feature_paths)
    
class VDataset(Dataset):
    def __init__(self, feature_paths = ''):
        self.feature_paths = feature_paths
        self.video_names, self.features, self.labels = self.get_meta(self.feature_paths)
    
    def load_data(self, filename):
        with open(filename, 'rb') as handle:
            meta = pickle.load(handle)
        return meta
        
    def get_meta(self, paths):
        video_names = []
        features = []
        labels = []
        for curr_path in tqdm(paths):
            curr_data = self.load_data(curr_path)
            features.extend(curr_data['video_features'])
            video_names.extend([curr_path]*curr_data['video_features'].shape[0])
            labels.extend([curr_data['true_label']]*curr_data['video_features'].shape[0])

        return video_names, features, labels

    def __getitem__(self, index):
        return self.video_names[index], torch.FloatTensor(self.features[index]), self.labels[index]
            
    def __len__(self):
        return len(self.video_names)


def _to_1d_feature_tensor(x: Any) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
    else:
        t = torch.as_tensor(x).detach().cpu()

    if t.ndim == 0:
        t = t.unsqueeze(0)
    if t.ndim >= 2 and t.shape[0] == 1:
        t = t[0]
    return t.to(dtype=torch.float32, copy=False)


class AVTDictDataset(Dataset):
    """Pickle feature dataset that returns a dict with modality keys.

    Output format:
        {"audio": Tensor|None, "text": Tensor|None, "label": int}
    """

    def __init__(self, feature_paths: Iterable[str], modalities: Iterable[str]):
        self.feature_paths = list(feature_paths)
        self.modalities = {str(m).strip().upper() for m in modalities if str(m).strip()}
        if not self.modalities:
            raise ValueError("At least one modality must be selected.")
        invalid = sorted([m for m in self.modalities if m not in {"A", "T"}])
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Allowed: ['A', 'T']")

    def load_data(self, filename: str) -> dict[str, Any]:
        with open(filename, "rb") as handle:
            meta = pickle.load(handle)
        if not isinstance(meta, dict):
            raise ValueError(f"Expected dict in {filename}, got {type(meta)}")
        return meta

    def __getitem__(self, index: int) -> dict[str, Any]:
        meta = self.load_data(self.feature_paths[index])

        audio = _to_1d_feature_tensor(meta["audio_features"]) if "A" in self.modalities else None
        text = _to_1d_feature_tensor(meta["text_features"]) if "T" in self.modalities else None

        label = meta["true_label"]
        if isinstance(label, torch.Tensor):
            label = int(label.detach().cpu().item())
        else:
            label = int(label)

        return {"audio": audio, "text": text, "label": label}

    def __len__(self) -> int:
        return len(self.feature_paths)


def collate_avt_dict(batch: list[dict[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Empty batch")

    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)

    def _stack_optional(key: str) -> torch.Tensor | None:
        values = [item.get(key) for item in batch]
        if all(v is None for v in values):
            return None
        if any(v is None for v in values):
            raise ValueError(f"Mixed None/non-None values for key={key!r}; check modality routing.")
        return torch.stack(values, dim=0)  # type: ignore[arg-type]

    return {
        "audio": _stack_optional("audio"),
        "text": _stack_optional("text"),
        "label": labels,
    }

if __name__ == "__main__":
    path = 'C:/Work/MER_PRL_features/AFEW/Val/'
    feature_paths = [os.path.join(path, i) for i in os.listdir(path)]

    data = AVTDataset(feature_paths)
