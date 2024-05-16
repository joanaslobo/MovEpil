import torch.utils.data
from torchvision.models.video import r3d_18, R3D_18_Weights


def create_video_ResNet(n_classes, device):
    weights = R3D_18_Weights.DEFAULT
    model = r3d_18(weights=weights)
    preprocess = weights.transforms()

    sequential = torch.nn.Sequential(torch.nn.Linear(out_features=n_classes, in_features=512, bias=True))

    model.fc = sequential
    model.requires_grad = True

    model = model.to(device=device)

    return model, preprocess
