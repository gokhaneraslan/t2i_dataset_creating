import cv2
import os
import argparse


def extract_frames(video_path, output_folder, interval_type, interval_value, image_format="jpg", prefix="frame"):
    """
    Extracts and saves frames from a video at specified intervals.

    Args:
        video_path (str): Path to the video file to be processed.
        output_folder (str): Folder where the frames will be saved.
        interval_type (str): Interval type ('s' for seconds, 'f' for frame count).
        interval_value (int/float): Value for seconds or frame count.
        image_format (str): Image format for saving (e.g., "jpg", "png").
        prefix (str): Prefix for the saved frame filenames.
    """
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found -> {video_path}")
        return 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    specific_output_folder = os.path.join(output_folder, video_name)

    if not os.path.exists(specific_output_folder):
        os.makedirs(specific_output_folder)
        print(f"Folder created: {specific_output_folder}")
        
    else:
        print(f"Frames will be saved to this folder (already exists): {specific_output_folder}")

    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video -> {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {video_path} - FPS: {fps:.2f}, Total Frames: {total_frames}")

    saved_frame_count = 0
    frame_id = 0
    next_capture_time = 0.0 
    next_capture_frame = 0  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_id / fps if fps > 0 else 0

        save_this_frame = False
        
        if interval_type == 's':
            if current_time >= next_capture_time:
                save_this_frame = True
                next_capture_time += interval_value
                
        elif interval_type == 'f':
            if frame_id >= next_capture_frame:
                save_this_frame = True
                next_capture_frame += interval_value
        
        if save_this_frame:

            frame_filename = f"{prefix}_{saved_frame_count:06d}.{image_format}"
            output_path = os.path.join(specific_output_folder, frame_filename)
            
            try:
                cv2.imwrite(output_path, frame)
                saved_frame_count += 1
                
                if saved_frame_count % 25 == 0 :
                    print(f"  {saved_frame_count} frames saved ({video_name})...")
                    
            except Exception as e:
                print(f"Error: Frame could not be saved {output_path} - {e}")
        
        frame_id += 1

    cap.release()
    
    print(f"A total of {saved_frame_count} frames from '{video_name}' video were saved to '{specific_output_folder}' folder.")
    
    return saved_frame_count


def process_all_videos_in_folder(input_folder, main_output_folder, interval_type, interval_value, image_format="jpg", prefix="frame"):
    """
    Processes all video files in the specified folder.
    """
    
    supported_formats = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    total_frames_extracted_all_videos = 0

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder not found -> {input_folder}")
        return

    if not os.path.exists(main_output_folder):
        os.makedirs(main_output_folder)
        print(f"Main output folder created: {main_output_folder}")

    for item in os.listdir(input_folder):
        item_path = os.path.join(input_folder, item)
        
        if os.path.isfile(item_path) and item.lower().endswith(supported_formats):
            print(f"\nProcessing: {item}")
            count = extract_frames(item_path, main_output_folder, interval_type, interval_value, image_format, prefix)
            total_frames_extracted_all_videos += count
        
        elif os.path.isdir(item_path):
            print(f"Subfolder found: {item_path}. Subfolders are not processed for now.")

    print(f"\n--- All Operations Completed ---")
    print(f"A total of {total_frames_extracted_all_videos} frames were extracted from all videos.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extracts frames from videos.")
    parser.add_argument("input_source", help="Video file to process or folder containing video files.")
    parser.add_argument("output_folder", help="Main folder where frames will be saved.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-s", "--seconds", type=float, help="Interval in seconds to capture a frame (e.g., 0.5).")
    group.add_argument("-f", "--frames", type=int, help="Interval in number of frames to capture a frame (e.g., 10).")
    
    parser.add_argument("--format", default="jpg", choices=["jpg", "png"], help="Image format for saving (default: jpg).")
    parser.add_argument("--prefix", default="frame", help="Prefix for saved frame filenames (default: frame).")

    args = parser.parse_args()

    interval_t = ""
    interval_v = 0

    if args.seconds is not None:
        interval_t = 's'
        interval_v = args.seconds
        if interval_v <= 0:
            print("Error: Seconds interval must be a positive value.")
            exit(1)
            
    elif args.frames is not None:
        interval_t = 'f'
        interval_v = args.frames
        if interval_v <= 0:
            print("Error: Frame interval must be a positive value.")
            exit(1)

    if os.path.isfile(args.input_source):
        extract_frames(
            args.input_source, 
            args.output_folder, 
            interval_t, interval_v, 
            args.format, 
            args.prefix
        )
    
    elif os.path.isdir(args.input_source):
        process_all_videos_in_folder(
            args.input_source, 
            args.output_folder, 
            interval_t, 
            interval_v, 
            args.format, 
            args.prefix
        )
    
    else:
        print(f"Error: Invalid input source -> {args.input_source}. Must be a file or a folder.")