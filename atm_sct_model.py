import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.metrics as metrics
import os

def get_dark_channel(image, size=15):
    min_img = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_atmospheric_light(image, dark_channel, top_percent=0.001):
    num_pixels = dark_channel.size
    num_brightest = int(max(top_percent * num_pixels, 1))
    brightest_pixels = np.argpartition(dark_channel.flatten(), -num_brightest)[-num_brightest:]
    atmospheric_light = np.mean(image.reshape(-1, 3)[brightest_pixels], axis=0)
    return atmospheric_light

def get_transmission_map(image, atmospheric_light, omega=0.95, size=15):
    norm_image = image / atmospheric_light
    transmission = 1 - omega * get_dark_channel(norm_image, size)
    return transmission

def refine_transmission(image, transmission, radius=60, eps=1e-3):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    refined_transmission = cv2.ximgproc.guidedFilter(guide=gray, src=transmission.astype(np.float32), radius=radius, eps=eps)
    return refined_transmission

def recover_scene_radiance(image, transmission, atmospheric_light, t0=0.1):
    transmission = np.maximum(transmission, t0)
    J = (image - atmospheric_light) / transmission[:, :, None] + atmospheric_light
    return np.clip(J, 0, 255).astype(np.uint8)

def compute_metrics(dehazed, ground_truth):
    dehazed_gray = cv2.cvtColor(dehazed, cv2.COLOR_RGB2GRAY)
    ground_truth_gray = cv2.cvtColor(ground_truth, cv2.COLOR_RGB2GRAY)

    psnr_value = metrics.peak_signal_noise_ratio(ground_truth_gray, dehazed_gray)
    ssim_value = metrics.structural_similarity(ground_truth_gray, dehazed_gray)
    mse_value = metrics.mean_squared_error(ground_truth_gray, dehazed_gray)

    return psnr_value, ssim_value, mse_value

def dehaze_image(hazy_path, ground_truth_path=None):
    hazy = cv2.imread(hazy_path)
    hazy = cv2.cvtColor(hazy, cv2.COLOR_BGR2RGB)

    dark_channel = get_dark_channel(hazy)
    atmospheric_light = get_atmospheric_light(hazy, dark_channel)
    transmission = get_transmission_map(hazy, atmospheric_light)
    refined_transmission = refine_transmission(hazy, transmission)
    dehazed = recover_scene_radiance(hazy, refined_transmission, atmospheric_light)

    # Save Dehazed Image
    save_path = r"..\out_dehazed_image_save_path.jpg"
    cv2.imwrite(save_path, cv2.cvtColor(dehazed, cv2.COLOR_RGB2BGR))
    print(f"Dehazed image saved at: {save_path}")

    # Check and Compute Metrics
    print("Checking Ground Truth Path...")
    print("Path Provided:", ground_truth_path)
    print("Path Exists:", os.path.exists(ground_truth_path))

    if ground_truth_path is not None and os.path.exists(ground_truth_path):
        ground_truth = cv2.imread(ground_truth_path)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)

        print("Ground truth image loaded successfully.")

        psnr, ssim, mse = compute_metrics(dehazed, ground_truth)

        print("\nPerformance Metrics:")
        print(f"  PSNR : {psnr:.2f} dB")
        print(f"  SSIM : {ssim:.4f}")
        print(f"  MSE  : {mse:.2f}")

    else:
        print("Ground truth not provided or invalid path.")

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(hazy)
    plt.title("Hazy Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(refined_transmission, cmap="gray")
    plt.title("Transmission Map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(dehazed)
    plt.title("Dehazed Image")
    plt.axis("off")

    plt.show()

# Example Usage"C:\Users\malav\Downloads\HSTS_2.0\HSTS_2.0\test\hazy\HazeDr_Google_404.jpeg"
hazy_image_path = r"..\hazy_img_path.jpg"
ground_truth_path = r"..\out_dehazed_img_path.jpg"

dehaze_image(hazy_image_path, ground_truth_path)
