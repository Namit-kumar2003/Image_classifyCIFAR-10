import argparse
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from src.config import config
from src.model import build_model
from src.dataset import cifar10_loaders


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook for forward pass to get activations
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        # Hook for backward pass to get gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        device = next(self.model.parameters()).device
        input_tensor = input_tensor.to(device)

        outputs = self.model(input_tensor)

        if target_class is None:
            target_class = outputs.argmax(dim=1).item()

        self.model.zero_grad()
        score = outputs[0, target_class]
        score.backward(retain_graph=True)

        grads = self.gradients[0]
        acts = self.activations[0]

        # Global average pooling of gradients
        weights = grads.mean(dim=(1, 2))

        # Weighted sum of activations
        cam = (weights[:, None, None] * acts).sum(dim=0).cpu().numpy()

        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam


def show_cam_on_image(img, cam, colormap=cv2.COLORMAP_JET):
    """
    Overlay Grad-CAM heatmap on image.
    img: numpy array of shape (H, W, 3) with values in [0,1]
    cam: Grad-CAM array of shape (h, w)
    """
    # Resize CAM to match image size
    cam_resized = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0

    overlay = np.clip(heatmap * 0.5 + img * 0.5, 0, 1)
    return heatmap, overlay


def preprocess_pil(img_pil):
    """
    Convert PIL image to normalized tensor
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])
    return transform(img_pil).unsqueeze(0)


def run_gradcam_on_image(model, device, img_path, save_dir):
    img_pil = Image.open(img_path).convert("RGB")
    input_tensor = preprocess_pil(img_pil).to(device)
    img_np = np.array(img_pil.resize((32, 32))) / 255.0

    # Target layer for Grad-CAM
    target_layer = model.conv3[3]

    gc = GradCAM(model, target_layer)
    cam = gc.generate(input_tensor)

    heatmap, overlay = show_cam_on_image(img_np, cam)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "gradcam_result.png")

    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np); axes[0].set_title("Input"); axes[0].axis("off")
    axes[1].imshow(heatmap); axes[1].set_title("Heatmap"); axes[1].axis("off")
    axes[2].imshow(overlay); axes[2].set_title("Overlay"); axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[INFO] GradCAM saved: {out_path}")


def run_gradcam_on_test(model, device, n_examples, save_dir):
    _, test_loader = cifar10_loaders()

    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for imgs, labels in test_loader:
        for i in range(imgs.size(0)):
            if count >= n_examples:
                return

            img_tensor = imgs[i].unsqueeze(0).to(device)

            img_np = imgs[i].permute(1, 2, 0).numpy()
            img_np = (img_np * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]
            img_np = np.clip(img_np, 0, 1)

            target_layer = model.conv3[3]

            gc = GradCAM(model, target_layer)
            cam = gc.generate(img_tensor)

            heatmap, overlay = show_cam_on_image(img_np, cam)

            out_path = os.path.join(save_dir, f"test_{count}.png")
            plt.imsave(out_path, np.hstack([img_np, heatmap, overlay]))
            print(f"[INFO] Saved {out_path}")
            count += 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained model checkpoint")
    p.add_argument("--image", type=str, default=None, help="Path to a single image")
    p.add_argument("--from_test", type=int, default=0, help="Number of test images to visualize")
    p.add_argument("--output_dir", type=str, default=os.path.join(config.RESULT_DIR, "gradcam"))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model()
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    print("[INFO] Loaded checkpoint.")

    if args.image:
        run_gradcam_on_image(model, device, args.image, args.output_dir)
    elif args.from_test > 0:
        run_gradcam_on_test(model, device, args.from_test, args.output_dir)
    else:
        print("No input specified. Use --image PATH or --from_test N")


if __name__ == "__main__":
    main()
