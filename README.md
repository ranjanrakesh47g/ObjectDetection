# Object Detection

This project compares vision-language models on a closed-vocabulary detection task. Each image or video clip receives one foreground, one background, and one relation, chosen from the labels present in the ground truth for that domain.

## Data

The collection covers two domains.

- The ecommerce split contains 38 product photographs from Getty Images, Pexels, and Unsplash. Every image uses the relation `object_with_background`. Foreground labels span 18 object classes, and background labels span 14 scene or context classes.
- The news split contains 15 images and 11 short clips of public figures, drawn from C-SPAN. Relations are `people_at_location` and `people_at_event`. The stills cover 6 people and 12 places or events. The clips cover 5 people and 10 places or events.

Ground truth is stored in `data/<split>/gt_images.json`, eg, for news, in `data/news/gt_videos.json`.

## Models

Six vision-language models were scored. Each response is constrained to the vocabulary of the split under evaluation. When two labels both apply, the prompt instructs the model to select the more specific one.

Open-source checkpoints:

- Qwen3-VL-2B, 4-bit, on a local GPU
- Qwen3-VL-4B, on a Colab GPU
- Qwen3-VL-8B, 4-bit, on a Colab GPU

Commercial APIs:

- Qwen3.8-27B and Qwen3-VL-30B-A3B through OpenRouter
- Gemini 3.1 Flash-Lite through GCP



## Evaluation

Each model labels every image in both splits. Accuracy, micro-F1, and macro-F1 are computed for foreground, background, and relation, then aggregated within each split and across the full image set. Mean latency is recorded for every model. Each prediction is restricted to the labels observed in that split. A field counts as correct only when the predicted string matches the ground truth exactly.

## Results

Commercial cost is reported per 1,000 images. Predictions and scores are written to `results/predictions.csv`, `results/metrics.csv`, and `results/overall_metrics.csv`.

Overall scores on the 53 stills:


| Model                         | Source                  | Accuracy | Micro-F1 | Macro-F1 | Latency (s) | Cost ($/1k images) |
| ----------------------------- | ----------------------- | -------- | -------- | -------- | ----------- | ------------------ |
| Qwen3-VL-2B, 4-bit            | open-weight, local      | 81.1%    | 83.0%    | 76.8%    | 1.164       | 0.000              |
| Qwen3-VL-4B                   | open-weight, colab      | 85.5%    | 86.3%    | 77.4%    | 3.477       | 0.000              |
| Qwen3-VL-8B, 4-bit            | open-weight, colab      | 85.5%    | 86.3%    | 79.8%    | 17.718      | 0.000              |
| Qwen3.8-27B (Openrouter)      | open-weight, commercial | 91.8%    | 91.8%    | 86.1%    | 2.342       | 0.160              |
| Qwen3-VL-30B-A3B (Openrouter) | open-weight, commercial | 89.9%    | 89.9%    | 84.3%    | 4.251       | 0.279              |
| Gemini 3.1 Flash-Lite         | commercial              | 93.7%    | 93.7%    | 89.1%    | 3.083       | 0.485              |




## Findings



### Open-source

- Qwen3-VL-4B has the highest accuracy among the self-hosted models (comparable to Qwen3-VL-8B), at a moderate latency of around 3 seconds.
- For low-latency applications, Qwen3-VL-2B is a suitable choice. It is the fastest among all the models and scores well above 81% in accuracy terms.



### Commercial

- Qwen3.8-27B (Openrouter) is the recommended commercial option. It has the lowest latency and the lowest price among them, with accuracy above 91%.



## Handling Video Clips

- Each clip is reduced to a few keyframes before inference.
- FFmpeg marks a shot boundary when the scene score exceeds 0.3. A shot is kept only when its duration is at least 0.5 seconds, else merged.
- Within each shot, frames are sampled at about five per second. YuNet selects the frame with the largest face. A shot with no face keeps its midpoint.
- Each keyframe is passed through the model to get foreground, background, relation predictions.
- The clip final prediction is the most frequent combination of foreground, background, and relation among the keyframes. A tie keeps the combination from the frame with the higher face score.
- Scores for the news clips are stored in `results/video_predictions.csv`, `results/video_metrics.csv`, and `results/video_overall_metrics.csv`.


| Model                    | Source                  | Accuracy | Micro-F1 | Macro-F1 | Latency (s) | Cost ($/1k keyframes) |
| ------------------------ | ----------------------- | -------- | -------- | -------- | ----------- | --------------------- |
| Qwen3-VL-2B, 4-bit       | open-weight, local      | 57.6%    | 57.6%    | 52.4%    | 1.409       | 0.000                 |
| Qwen3.8-27B (Openrouter) | open-weight, commercial | 84.8%    | 84.8%    | 79.3%    | 3.660       | 0.160                 |
| Gemini 3.1 Flash-Lite    | commercial              | 75.8%    | 75.8%    | 61.8%    | 4.176       | 0.485                 |


- On the 11 news clips, Qwen3.8-27B (Openrouter) scores the highest in terms of accuracy among all the models, and has the least values for latency and pricing among the commercial options.



## APIs

The local service listens on `127.0.0.1:8000` and labels an upload with Qwen3.8-27B.


| Method | Path            | Purpose                                                                                      |
| ------ | --------------- | -------------------------------------------------------------------------------------------- |
| GET    | `/health_check` | Returns `{"status": "ok"}`                                                                   |
| POST   | `/detect`       | Accepts an image or video file and returns the foreground, background, relation, and latency |




## Local setup

1. Install FFmpeg. The package includes `ffprobe`.
  ```bash
   sudo apt install ffmpeg
  ```
2. Download the YuNet weights. OpenCV does not ship this file.
  ```bash
   mkdir -p models
   curl -L -o models/face_detection_yunet_2023mar.onnx https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
  ```
3. Create a Python 3.12 environment and install the dependencies in `requirements.txt`.
  ```bash
   python3.12 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
  ```
4. Set `OPENROUTER_API_KEY` and `GEMINI_API_KEY` in the environment for the commercial models.
5. Start the local service. It labels an uploaded image or video with Qwen3.8-27B.
  ```bash
   python -m src.api.main
  ```

