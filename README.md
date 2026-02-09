# VideoRAG: Semantic Video Analysis & Keyframe Extraction

**VideoRAG** is a Python-based pipeline designed to facilitate Retrieval-Augmented Generation (RAG) on video content. It automates the process of downloading YouTube videos, extracting frames, generating semantic embeddings using state-of-the-art vision models (SigLIP), and selecting the most representative keyframes via clustering.

This tool is ideal for developers building "Chat with Video" applications, video summarization tools, or semantic video search engines.

##  Features

* **YouTube Integration**: Seamlessly download videos using `yt_dlp`.
* **State-of-the-Art Embeddings**: Uses **Google's SigLIP** (`siglip-so400m-patch14-384`) for high-quality, language-aligned image embeddings.
* **Smart Keyframe Selection**: Implements **K-Means clustering** to select semantically diverse keyframes, ensuring the selected frames represent distinct scenes rather than repetitive shots.
* **GPU Acceleration**: Automatically utilizes CUDA if available for faster processing.
* **Visualization**: Built-in support for displaying frames using `matplotlib` and `seaborn-image`.

##  Tech Stack

* **Python 3.8+**
* **PyTorch** (Deep Learning Framework)
* **Hugging Face Transformers** (Model loading)
* **Scikit-Learn** (K-Means Clustering)
* **OpenCV** (Video Processing)
* **yt-dlp** (Video Downloading)
 
##  How It Works

1.  **Frame Extraction**: The script reads the video file and samples frames at a specified interval (e.g., 1 frame per second).
2.  **Embedding Generation**: Each sampled frame is passed through the **SigLIP** model to create a dense vector representation.
3.  **Clustering**: The embeddings are grouped using **K-Means clustering**.
4.  **Selection**: The algorithm identifies the frame closest to the centroid of each cluster. This ensures that the final list of keyframes covers the "story" of the video without redundancy.


