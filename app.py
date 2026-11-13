
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt, label
from pathlib import Path


# --------------------------------------------------
# 1. CNN MODEL
# --------------------------------------------------

class CrackCNN(nn.Module):
    def __init__(self):
        super(CrackCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# --------------------------------------------------
# 2. U-NET MODEL
# --------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        p1 = self.pool(s1)

        s2 = self.enc2(p1)
        p2 = self.pool(s2)

        s3 = self.enc3(p2)
        p3 = self.pool(s3)

        s4 = self.enc4(p3)
        p4 = self.pool(s4)

        b = self.bottleneck(p4)

        d4 = self.up4(b)
        d4 = torch.cat([d4, s4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, s3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, s2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, s1], dim=1)
        d1 = self.dec1(d1)

        return self.final_conv(d1)


# --------------------------------------------------
# 3. DEVICE AND MODEL LOADING
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = Path("models")


@st.cache_resource
def load_models():

    cnn = CrackCNN().to(device)
    unet = UNet().to(device)

    cnn.load_state_dict(
        torch.load(
            MODELS_DIR / "custom_cnn_crack_model.pth",
            map_location=device
        )
    )

    unet.load_state_dict(
        torch.load(
            MODELS_DIR / "best_unet_crack_model.pth",
            map_location=device
        )
    )

    cnn.eval()
    unet.eval()

    return cnn, unet


# --------------------------------------------------
# 4. IMAGE PREPROCESSING
# --------------------------------------------------

def preprocess_cnn(image):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return transform(image).unsqueeze(0).to(device)


def preprocess_unet(image):

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform(image).unsqueeze(0).to(device)


# --------------------------------------------------
# 5. CRACK DETECTION AND SEGMENTATION
# --------------------------------------------------

def detect_crack(image, cnn, unet):

    # Stage 1: CNN classification

    cnn_input = preprocess_cnn(image)

    with torch.no_grad():
        output = cnn(cnn_input)
        prediction = torch.argmax(output, dim=1).item()

    probabilities = torch.softmax(output, dim=1)[0]

    st.write("Non-Cracked Probability:", f"{probabilities[0].item():.4f}")
    st.write("Cracked Probability:", f"{probabilities[1].item():.4f}")

    

    # Stage 2: U-Net segmentation

    unet_input = preprocess_unet(image)

    with torch.no_grad():
        output = unet(unet_input)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()


    st.image(
    (mask * 255).astype(np.uint8),
    caption="U-Net Probability Map",
    clamp=True
    )

    binary_mask = (mask > 0.3).astype(np.uint8) * 255
    if not np.any(binary_mask):

        return None

    return binary_mask


# --------------------------------------------------
# 6. CREATE RED OVERLAY
# --------------------------------------------------

def create_overlay(image, mask):

    mask_image = Image.fromarray(mask).resize(
        image.size,
        Image.Resampling.NEAREST
    )

    original = np.array(image).copy()
    crack_pixels = np.array(mask_image) > 0

    original[crack_pixels] = (
        0.5 * original[crack_pixels] +
        0.5 * np.array([255, 0, 0])
    ).astype(np.uint8)

    return Image.fromarray(original)


# --------------------------------------------------
# 7. CRACK MEASUREMENT
# --------------------------------------------------

def measure_crack(mask):

    binary = mask > 0

    # Crack area
    crack_area = np.sum(binary)

    # Crack skeleton
    skeleton = skeletonize(binary)

    # Crack length
    crack_length = np.sum(skeleton)

    # Crack width
    distance = distance_transform_edt(binary)
    widths = distance[skeleton] * 2

    if len(widths) > 0:
        average_width = np.mean(widths)
        maximum_width = np.max(widths)
    else:
        average_width = 0
        maximum_width = 0

    return (
        crack_area,
        crack_length,
        average_width,
        maximum_width
    )


# --------------------------------------------------
# 8. CRACK SEVERITY CLASSIFICATION
# --------------------------------------------------

def classify_severity(crack_area, maximum_width):

    if crack_area == 0:
        return "No Crack Detected"

    elif maximum_width >= 8 or crack_area >= 2500:
        return "High"

    elif maximum_width < 3 and crack_area < 500:
        return "Low"

    else:
        return "Medium"


# --------------------------------------------------
# 9. STREAMLIT INTERFACE
# --------------------------------------------------

st.set_page_config(
    page_title="Concrete Crack Detection",
    layout="wide"
)

st.title("Concrete Crack Detection System")

st.write(
    "Upload a concrete image to detect, segment "
    "and measure cracks."
)

uploaded_file = st.file_uploader(
    "Choose a concrete image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=500
    )

    if st.button("Analyze Image"):

        with st.spinner("Analyzing concrete image..."):

            cnn, unet = load_models()

            mask = detect_crack(image, cnn, unet)

        if mask is None:

            st.success("No crack detected!")

        else:

            st.success("Crack detected!")

            # Display segmentation mask

            st.image(
                mask,
                caption="Crack Segmentation Mask",
                clamp=True,
                width=500
            )

            # Display red overlay

            overlay = create_overlay(image, mask)

            st.image(
                overlay,
                caption="Cracks Highlighted in Red",
                width=500
            )

            # Calculate measurements

            (
                crack_area,
                crack_length,
                average_width,
                maximum_width
            ) = measure_crack(mask)

            # Calculate severity

            severity = classify_severity(
                crack_area,
                maximum_width
            )

            # Display measurements

            st.subheader("Crack Measurements")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Crack Area",
                    f"{crack_area} pixels"
                )

                st.metric(
                    "Crack Length",
                    f"{crack_length:.2f} pixels"
                )

            with col2:
                st.metric(
                    "Average Crack Width",
                    f"{average_width:.2f} pixels"
                )

                st.metric(
                    "Maximum Crack Width",
                    f"{maximum_width:.2f} pixels"
                )

            # Display severity

            st.subheader("Crack Severity")

            if severity == "High":
                st.error("High Severity")

            elif severity == "Medium":
                st.warning("Medium Severity")

            elif severity == "Low":
                st.success("Low Severity")

            else:
                st.info("No Crack Detected")
