import torch
import torchvision
import os
from sklearn.model_selection import train_test_split


class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, transform=None, output_format="THWC", encoder=None):
        super(VideoDataset, self).__init__()
        self.base_path = base_path
        self.categories = [folder_name for folder_name in os.listdir(self.base_path) if not folder_name.startswith(".")]
        self.transform = transform
        self.output_format = output_format
        self.encoder = encoder

        self.label_name_idx = dict(zip(self.categories, range(len(self.categories))))
        self.label_idx_name = dict(zip(range(len(self.categories)), self.categories))

        self.label_index = []
        self.videos_path = []
        for category in self.categories:
            for video_name in os.listdir(os.path.join(self.base_path, category)):
                self.videos_path.append(os.path.join(self.base_path, category, video_name))
                self.label_index.append(self.label_name_idx[category])

    def __len__(self):
        return len(self.videos_path)


    def __getitem__(self, idx):
        video_data = torchvision.io.read_video(self.videos_path[idx], end_pts=64, pts_unit="pts", output_format=self.output_format)

        video_frames = video_data[0]
        
        if self.transform:
            video_frames = self.transform(image=video_frames.cpu().numpy())['image']

        label = self.label_index[idx]

        return video_frames, label


class SubDatasets(torch.utils.data.Dataset):
  def __init__(self, videos_path, categories, transform, seq_lenght=32, output_format="THWC"):
    super(SubDatasets, self).__init__()
    self.videos_path = videos_path
    self.label_index = categories
    self.transform = transform
    self.output_format = output_format
    self.seq_lenght = seq_lenght

  def __len__(self):
      return len(self.videos_path)

  def __getitem__(self, idx):
      video_data = torchvision.io.read_video(self.videos_path[idx], end_pts=self.seq_lenght, pts_unit="sec",
                                             output_format=self.output_format)

      video_frames = video_data[0]
      if self.transform:
          video_frames = self.transform(image=video_frames.cpu().numpy())['image']

      if video_frames.shape[0] >= self.seq_lenght:
        video_frames = video_frames[:self.seq_lenght]
      else:
          zero_tensor = torch.zeros((self.seq_lenght, video_frames.shape[1], video_frames.shape[2], video_frames.shape[3]))
          zero_tensor[:video_frames.shape[0]] = video_frames
          return zero_tensor, self.label_index[idx]

      return video_frames, self.label_index[idx]


def get_data_split(base_path, batch_size, transform=None, seq_lenght=32, num_workers=0):
    video_data = VideoDataset(base_path=base_path)

    x, y = video_data.videos_path, video_data.label_index

    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.1, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, shuffle=True, test_size=0.15, random_state=42, stratify=y_train)

    train = SubDatasets(videos_path=X_train, categories=y_train, output_format="TCHW", transform=transform, seq_lenght=seq_lenght)
    val = SubDatasets(videos_path=X_val, categories=y_val, output_format="TCHW", transform=None, seq_lenght=seq_lenght)
    test = SubDatasets(videos_path=X_test, categories=y_test, output_format="TCHW", transform=None, seq_lenght=seq_lenght)

    train_dataset = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dataset, val_dataset, test_dataset, video_data.label_idx_name
