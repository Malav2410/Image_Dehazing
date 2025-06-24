import cv2
import image_dehazer


if __name__ == "__main__":
    
    HazeImg = cv2.imread(r"..\path_to_hazy_img.jpg")
    HazeCorrectedImg, haze_map = image_dehazer.remove_haze(HazeImg, showHazeTransmissionMap=False)		# Remove Haze
   #  Calculate PSNR & SSIM
    mse, psnr_val, ssim_val =image_dehazer.calculate_metrics(HazeCorrectedImg,HazeImg)
    print(f"PSNR : {psnr_val:.2f} dB")
    print(f"SSIM : {ssim_val:.4f}")   
    print(f"MSE  : {mse:.2f}")

    cv2.imshow('haze_map', haze_map);						# display the original hazy image
    cv2.imshow('enhanced_image', HazeCorrectedImg);			# display the result
    cv2.waitKey(0)
    cv2.imwrite(r"..\path_to_dehazed_img.jpg", HazeCorrectedImg)
   # Assuming you already have:
 # hazy_image -> Input hazy image
 # clear_image -> Ground truth clear image



   
