"""
pip install moviepy if needed
"""

import os
import moviepy.editor as mpy

def create_output_dir_structure(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            os.makedirs(os.path.join(output_dir, os.path.relpath(os.path.join(root, dir_name), input_dir)), exist_ok=True)

def process_video(input_video_path, output_video_path, crop_region):
    # Load the video
    clip = mpy.VideoFileClip(input_video_path)

    # Crop the video
    x1, y1, x2, y2 = crop_region
    cropped_clip = clip.crop(x1=x1, y1=y1, x2=x2, y2=y2)

    # Rotate the video
    rotated_clip = cropped_clip.rotate(90)

    # Write the result to the output path
    rotated_clip.write_videofile(output_video_path)

def process_all_videos(input_dir, output_dir, crop_region):
    create_output_dir_structure(input_dir, output_dir)

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.mp4'):
                input_video_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_video_path, input_dir)
                output_video_path = os.path.join(output_dir, relative_path)

                # Process the video
                process_video(input_video_path, output_video_path, crop_region)

def main():
    input_dir = './TEST'
    output_dir = input_dir + '_cropped'

    """
    RELEVANT INFO:

    The top-left corner of the frame is the origin (0, 0).
    The x values increase as you move to the right.
    The y values increase as you move down.

    The crop_region is a tuple of 4 integers: (x1, y1, x2, y2)
    x1: The x-coordinate of the top-left corner of the crop rectangle.
    y1: The y-coordinate of the top-left corner of the crop rectangle.
    x2: The x-coordinate of the bottom-right corner of the crop rectangle.
    y2: The y-coordinate of the bottom-right corner of the crop rectangle.
    """

    crop_region = (50, 50, 400, 400)  # Change this to your desired crop region (x1, y1, x2, y2)

    os.makedirs(output_dir, exist_ok=True)
    process_all_videos(input_dir, output_dir, crop_region)

if __name__ == '__main__':
    main()
