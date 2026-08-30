# ================================================================
# DCDiCL-DELTA: Medical Image Denoising
# Dictionary + Sparse Representation + Residual CNN
# ================================================================

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.cluster import MiniBatchKMeans

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ================================================================
# 1. CONFIGURATION
# ================================================================

IMAGE_PATH = "medical_image.png"

PATCH_SIZE = 64
DICT_SIZE = 64

NOISE_SIGMA = 25

BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

print("Running on:", DEVICE)


# ================================================================
# 2. LOAD MEDICAL IMAGE
# ================================================================

def load_image(path):

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(
            "Image not found. Check IMAGE_PATH."
        )

    image = image.astype(np.float32) / 255.0

    return image


# ================================================================
# 3. ADD GAUSSIAN NOISE
# ================================================================

def add_gaussian_noise(image, sigma):

    noise = np.random.normal(
        loc=0.0,
        scale=sigma / 255.0,
        size=image.shape
    )

    noisy = image + noise

    noisy = np.clip(noisy, 0, 1)

    return noisy


# ================================================================
# 4. CLAHE PREPROCESSING
# ================================================================

def apply_clahe(image):

    image_uint8 = np.uint8(image * 255)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(image_uint8)

    enhanced = enhanced.astype(np.float32) / 255.0

    return enhanced


# ================================================================
# 5. EXTRACT IMAGE PATCHES
# ================================================================

def get_patches(image, patch_size):

    patches = extract_patches_2d(
        image,
        (patch_size, patch_size),
        max_patches=5000,
        random_state=42
    )

    patches = patches.reshape(
        patches.shape[0],
        -1
    )

    return patches


# ================================================================
# 6. DICTIONARY LEARNING
# ================================================================

def learn_dictionary(patches, dictionary_size):

    print("\nLearning dictionary...")

    kmeans = MiniBatchKMeans(
        n_clusters=dictionary_size,
        random_state=42,
        batch_size=256,
        n_init=3
    )

    kmeans.fit(patches)

    dictionary = kmeans.cluster_centers_

    # Normalize dictionary atoms
    dictionary = dictionary - np.mean(
        dictionary,
        axis=1,
        keepdims=True
    )

    norm = np.linalg.norm(
        dictionary,
        axis=1,
        keepdims=True
    )

    dictionary = dictionary / (norm + 1e-8)

    print("Dictionary shape:", dictionary.shape)

    return dictionary


# ================================================================
# 7. SPARSE CODING
# ================================================================

def sparse_code(patch, dictionary, sparsity=8):

    # Correlation between patch and dictionary atoms
    correlation = np.abs(
        dictionary @ patch
    )

    # Select strongest atoms
    indices = np.argsort(
        correlation
    )[-sparsity:]

    selected_dictionary = dictionary[indices]

    # Least-square coefficient estimation
    coefficients = np.linalg.lstsq(
        selected_dictionary.T,
        patch,
        rcond=None
    )[0]

    sparse_vector = np.zeros(
        dictionary.shape[0]
    )

    sparse_vector[indices] = coefficients

    return sparse_vector


# ================================================================
# 8. DICTIONARY BASED PATCH RECONSTRUCTION
# ================================================================

def dictionary_denoise_patch(
        patch,
        dictionary,
        sparsity=8):

    coefficients = sparse_code(
        patch,
        dictionary,
        sparsity
    )

    reconstructed = (
        coefficients @ dictionary
    )

    return reconstructed


# ================================================================
# 9. IMAGE RECONSTRUCTION USING DICTIONARY
# ================================================================

def dictionary_denoising(
        noisy_image,
        dictionary,
        patch_size,
        sparsity=8):

    h, w = noisy_image.shape

    output = np.zeros_like(
        noisy_image
    )

    weight = np.zeros_like(
        noisy_image
    )

    for i in range(
        0,
        h - patch_size + 1,
        patch_size
    ):

        for j in range(
            0,
            w - patch_size + 1,
            patch_size
        ):

            patch = noisy_image[
                i:i+patch_size,
                j:j+patch_size
            ]

            original_shape = patch.shape

            patch_vector = patch.flatten()

            reconstructed = dictionary_denoise_patch(
                patch_vector,
                dictionary,
                sparsity
            )

            reconstructed = reconstructed.reshape(
                original_shape
            )

            output[
                i:i+patch_size,
                j:j+patch_size
            ] += reconstructed

            weight[
                i:i+patch_size,
                j:j+patch_size
            ] += 1

    output = output / (
        weight + 1e-8
    )

    return np.clip(
        output,
        0,
        1
    )


# ================================================================
# 10. RESIDUAL CNN
# ================================================================

class ResidualCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Conv2d(
                1, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                32, 1,
                kernel_size=3,
                padding=1
            )
        )

    def forward(self, x):

        residual = self.network(x)

        # Residual learning
        output = x - residual

        return output


# ================================================================
# 11. DATASET FOR RESIDUAL CNN
# ================================================================

class ImageDataset(Dataset):

    def __init__(
            self,
            clean,
            noisy):

        self.clean = clean
        self.noisy = noisy

    def __len__(self):

        return 1

    def __getitem__(self, index):

        noisy = torch.tensor(
            self.noisy,
            dtype=torch.float32
        ).unsqueeze(0)

        clean = torch.tensor(
            self.clean,
            dtype=torch.float32
        ).unsqueeze(0)

        return noisy, clean


# ================================================================
# 12. TRAIN RESIDUAL CNN
# ================================================================

def train_residual_cnn(
        model,
        clean,
        noisy,
        epochs=20):

    dataset = ImageDataset(
        clean,
        noisy
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    model.train()

    for epoch in range(epochs):

        total_loss = 0

        for noisy_batch, clean_batch in loader:

            noisy_batch = noisy_batch.to(
                DEVICE
            )

            clean_batch = clean_batch.to(
                DEVICE
            )

            optimizer.zero_grad()

            output = model(
                noisy_batch
            )

            loss = criterion(
                output,
                clean_batch
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:

            print(
                "Epoch [{}/{}] Loss = {:.6f}".format(
                    epoch + 1,
                    epochs,
                    total_loss
                )
            )

    return model


# ================================================================
# 13. PSNR AND SSIM
# ================================================================

def evaluate_image(
        original,
        denoised):

    psnr = peak_signal_noise_ratio(
        original,
        denoised,
        data_range=1.0
    )

    ssim = structural_similarity(
        original,
        denoised,
        data_range=1.0
    )

    return psnr, ssim


# ================================================================
# 14. MAIN DCDiCL-DELTA PIPELINE
# ================================================================

print("\n==========================================")
print(" DCDiCL-DELTA MEDICAL IMAGE DENOISING")
print("==========================================")

# Load image
original = load_image(
    IMAGE_PATH
)

print(
    "Image size:",
    original.shape
)


# ------------------------------------------------
# Add Gaussian noise
# ------------------------------------------------

noisy = add_gaussian_noise(
    original,
    NOISE_SIGMA
)


# ------------------------------------------------
# CLAHE
# ------------------------------------------------

clahe_image = apply_clahe(
    noisy
)


# ------------------------------------------------
# Patch extraction
# ------------------------------------------------

patches = get_patches(
    clahe_image,
    PATCH_SIZE
)

print(
    "Training patches:",
    patches.shape
)


# ------------------------------------------------
# Dictionary learning
# ------------------------------------------------

dictionary = learn_dictionary(
    patches,
    DICT_SIZE
)


# ------------------------------------------------
# Sparse dictionary denoising
# ------------------------------------------------

dictionary_output = dictionary_denoising(
    clahe_image,
    dictionary,
    PATCH_SIZE,
    sparsity=8
)


# ------------------------------------------------
# Residual CNN
# ------------------------------------------------

model = ResidualCNN().to(
    DEVICE
)

model = train_residual_cnn(
    model,
    original,
    dictionary_output,
    EPOCHS
)


# ------------------------------------------------
# CNN prediction
# ------------------------------------------------

model.eval()

input_tensor = torch.tensor(
    dictionary_output,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0)

with torch.no_grad():

    final_output = model(
        input_tensor.to(DEVICE)
    )

final_output = final_output.squeeze().cpu().numpy()

final_output = np.clip(
    final_output,
    0,
    1
)


# ================================================================
# 15. PERFORMANCE EVALUATION
# ================================================================

psnr_noisy, ssim_noisy = evaluate_image(
    original,
    noisy
)

psnr_dict, ssim_dict = evaluate_image(
    original,
    dictionary_output
)

psnr_final, ssim_final = evaluate_image(
    original,
    final_output
)


print("\n==========================================")
print(" PERFORMANCE")
print("==========================================")

print(
    "Noisy Image     : PSNR = {:.2f} dB, SSIM = {:.4f}".format(
        psnr_noisy,
        ssim_noisy
    )
)

print(
    "Dictionary      : PSNR = {:.2f} dB, SSIM = {:.4f}".format(
        psnr_dict,
        ssim_dict
    )
)

print(
    "DCDiCL-Delta    : PSNR = {:.2f} dB, SSIM = {:.4f}".format(
        psnr_final,
        ssim_final
    )
)


# ================================================================
# 16. DISPLAY RESULTS
# ================================================================

plt.figure(figsize=(15, 4))

plt.subplot(1, 5, 1)

plt.imshow(
    original,
    cmap="gray"
)

plt.title("Original")

plt.axis("off")


plt.subplot(1, 5, 2)

plt.imshow(
    noisy,
    cmap="gray"
)

plt.title(
    "Gaussian Noise\nσ={}".format(
        NOISE_SIGMA
    )
)

plt.axis("off")


plt.subplot(1, 5, 3)

plt.imshow(
    clahe_image,
    cmap="gray"
)

plt.title("CLAHE")

plt.axis("off")


plt.subplot(1, 5, 4)

plt.imshow(
    dictionary_output,
    cmap="gray"
)

plt.title("Sparse Dictionary")

plt.axis("off")


plt.subplot(1, 5, 5)

plt.imshow(
    final_output,
    cmap="gray"
)

plt.title("DCDiCL-Delta")

plt.axis("off")


plt.tight_layout()

plt.show()


# ================================================================
# 17. SAVE RESULT
# ================================================================

output_uint8 = np.uint8(
    final_output * 255
)

cv2.imwrite(
    "DCDiCL_Delta_Denoised.png",
    output_uint8
)

print(
    "\nSaved: DCDiCL_Delta_Denoised.png"
)