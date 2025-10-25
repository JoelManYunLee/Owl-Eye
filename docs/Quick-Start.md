
You can download weights in https://drive.google.com/drive/folders/16mVjXrul3VaXKfHHYauY0QI-SG-JVLvL?usp=sharing

add the weight folder to the "src/model"

# Creating a Environment.

```
conda create --name SoloShuttlePose python=3.9
```

# Activate environment

```
conda activate SoloShuttlePose
```

# Install torch

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

# Install the other packages

```
pip install -r docs/requirements.txt
```

# Install yt-dlp

```
pip install yt-dlp
```

# Download the youtube BWF video for ShuttleSet

```
python src/tools/yt-dlp4ShuttleSet.py
```


# If you want to manually select the valid frames, you can run the following code.

```
python src/tools/FrameSelect.py --folder_path "videos"
```

# Run the following code for player, court ,net detect.

Process a single video file

```
python main.py --file_path "videos/<REPLACE WITH YOUR VIDEO>" --result_path "res" 
```

Process only unprocessed videos.

```
python main.py --folder_path "videos" --result_path "res" 
```

Force processing of all videos, including those that have already been processed.

```
python main.py --folder_path "videos" --result_path "res" --force
```

# Draw the court, net, and players

Process a single video file with all annotations enabled
```
python src/tools/VideoDraw.py --file_path "videos/<REPLACE WITH YOUR VIDEO>" --result_path "res" --court --net --players --ball --trajectory
```

Process only unprocessed videos.

```
python src/tools/VideoDraw.py --folder_path "videos" --result_path "res" --court --net --players --ball
```

Force processing of all videos, including those that have already been processed.

```
python src/tools/VideoDraw.py --folder_path "videos" --result_path "res" --force --court --net --players --ball --trajectory
```

