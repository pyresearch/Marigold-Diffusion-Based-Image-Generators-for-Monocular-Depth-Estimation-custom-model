import os
import cv2
import imageio
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO
from diffusers import MarigoldDepthPipeline
from io import BytesIO
import zipfile

def process_video_with_yolo_ffmpeg(
    video_path: str,
    output_video_path: str,
    output_depth_zip_path: str,
    marigold_pipe: MarigoldDepthPipeline,
    yolo_model: YOLO,
    max_frames: int = 450,
    processing_resolution: int = 768,
    device: str = "cuda",
):
    """
    1. Opens the input video with ImageIO's 'ffmpeg' plugin.
    2. For each frame:
       - Run YOLO detection => bounding-box image.
       - Run Marigold => colorized depth image.
       - Stack YOLO + depth images side-by-side => combined frame.
       - Write the combined frame to `output_video_path`.
       - Save the 16-bit depth image in a ZIP (`output_depth_zip_path`).
    3. Stops after 'max_frames' or end of video.
    """

    # ---------------- 1) Read video with FFmpeg plugin ----------------
    # Requires `pip install imageio-ffmpeg`
    reader = imageio.get_reader(video_path, "ffmpeg")

    # Retrieve metadata
    meta_data = reader.get_meta_data()
    fps = meta_data["fps"]  # frames per second
    duration_sec = meta_data.get("duration", None)
    size = meta_data.get("size", (640, 480))

    # If we know total_frames from metadata, use that. Otherwise, large fallback.
    if duration_sec is not None:
        total_frames = int(duration_sec * fps)
    else:
        total_frames = 999999999

    # 2) Prepare a writer for the side-by-side video
    # Set 'fps=fps' to match input video
    writer = imageio.get_writer(output_video_path, fps=fps)

    # Prepare a .zip file for saving depth frames
    zipf = zipfile.ZipFile(output_depth_zip_path, "w", zipfile.ZIP_DEFLATED)

    # ---------------- 3) Frame-by-frame processing ----------------
    frame_count = 0
    for frame_id, frame in enumerate(reader):
        frame_count += 1
        if frame_count > max_frames:
            print("Reached max frames limit:", max_frames)
            break

        # Convert to PIL for YOLO & Marigold
        frame_pil = Image.fromarray(frame).convert("RGB")

        # ------------ YOLO detection ------------
        yolo_results = yolo_model(frame_pil, device=device)
        # .plot() returns a NumPy (H, W, 3) image with bounding boxes (RGB)
        yolo_detection_image = yolo_results[0].plot()

        # ------------ Marigold depth ------------
        pipe_out = marigold_pipe(
            frame_pil,
            num_inference_steps=1,        # change if you want more steps
            ensemble_size=1,              # or more for better results
            processing_resolution=processing_resolution,
            match_input_resolution=False,
            batch_size=1,
        )
        # Convert Marigold's depth to a color image (PIL -> NumPy)
        depth_colored_pil = marigold_pipe.image_processor.visualize_depth(
            pipe_out.prediction
        )[0]
        depth_colored_np = np.array(depth_colored_pil.convert("RGB"))  # shape (H2,W2,3)

        # ------------- Stack YOLO + Depth -------------
        det_h, det_w, _ = yolo_detection_image.shape
        dep_h, dep_w, _ = depth_colored_np.shape

        # Resize depth image to match YOLO detection image height
        if det_h != dep_h:
            depth_colored_np = cv2.resize(
                depth_colored_np,
                (dep_w * det_h // dep_h, det_h),  # keep aspect ratio
                interpolation=cv2.INTER_AREA,
            )

        # Combine horizontally
        combined = np.hstack([yolo_detection_image, depth_colored_np])

        # Convert RGB -> BGR for writing with imageio (depends on your preference)
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        # Write the combined frame
        writer.append_data(combined_bgr)

        # ------------ Save 16-bit depth frame to .zip ------------
        depth_16bit_pil = marigold_pipe.image_processor.export_depth_to_16bit_png(
            pipe_out.prediction
        )[0]
        # Save to in-memory buffer
        img_byte_arr = BytesIO()
        depth_16bit_pil.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        zipf.writestr(f"frame_{frame_count:05d}.png", img_byte_arr.read())

        print(f"Processed frame {frame_count}/{min(total_frames, max_frames)}")

    # 4) Close everything
    writer.close()
    zipf.close()
    reader.close()
    print(f"Completed video processing: wrote {frame_count} frames.")


def main():
    """
    Example usage script:
    1) Loads Marigold depth pipeline.
    2) Loads a YOLO model from ultralytics.
    3) Calls process_video_with_yolo_ffmpeg with user-specified paths.
    """
    # ---------------- 1) Device ----------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------- 2) Load Marigold pipeline ----------------
    marigold_pipe = MarigoldDepthPipeline.from_pretrained(
        "prs-eth/marigold-depth-lcm-v1-0",
        variant="fp16" if device == "cuda" else None,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    try:
        marigold_pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        pass  # xformers not installed, ignore

    # ---------------- 3) Load YOLO model ----------------
    yolo_model = YOLO("last.pt")  # Or your custom YOLO weights
    yolo_model.to(device)

    # ---------------- 4) Paths ----------------
    input_video_path = "5097500-hd_1920_1080_25fps.mp4"          # Your input video
    output_video_path = "output_side_by_side.mp4"   # The combined YOLO+Depth video
    output_depth_zip_path = "depth_frames_16bit.zip"

    # ---------------- 5) Run processing ----------------
    process_video_with_yolo_ffmpeg(
        video_path=input_video_path,
        output_video_path=output_video_path,
        output_depth_zip_path=output_depth_zip_path,
        marigold_pipe=marigold_pipe,
        yolo_model=yolo_model,
        max_frames=300,            # processes up to 300 frames
        processing_resolution=768, # typical for Marigold
        device=device,
    )
    print("Done! Check your output video and .zip of depth frames.")


if __name__ == "__main__":
    main()
