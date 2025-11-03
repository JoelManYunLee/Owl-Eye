## Usage: Landing Detection and Video Annotation

### 1) Generate landings JSON and annotated video

Single video:
```bash
python -m src.tools.LandingAnnotate --file_path path/to/video.mp4 --result_path res --force
```

Folder of videos:
```bash
python -m src.tools.LandingAnnotate --folder_path path/to/folder --result_path res
```

Artifacts:
- JSON: `res/landings/{video}_landings.json`
- Video: `res/videos/{video}/{video}_landing.mp4` (timeline overlay)

### 2) Timeline legend
- IN ground landing: green tick
- OUT ground landing: red tick
- Net hit: orange tick
- Current frame cursor: white line

### Notes
- Requires existing outputs: ball positions, player joints, and court/net references under `res/...` as produced by the repo's earlier stages.
- Homography is computed from `court_info` corners (indices [0,1,4,5]).


