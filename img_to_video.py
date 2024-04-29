import os
import cv2

def images_to_video(image_folder, video_name, fps):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            images.append(cv2.imread(os.path.join(image_folder, filename)))

    # Get image dimensions
    height, width, _ = images[0].shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You may need to change the codec based on your system and needs
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Write images to video
    for img in images:
        video.write(img)

    # Release the video writer
    video.release()

# Example usage:
image_folder = '/home/zhuoran/5260Proj/exp_old/exp_hid_lay_3_nn_0.1_mask_iter/zju_386_mono-direct-mlp_field-mlp-shallow_mlp-default/predict-dance0/renders'
video_name = '386_psnr_31.mp4'
fps = 20  # Frames per second

images_to_video(image_folder, video_name, fps)
