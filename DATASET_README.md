Collecting landmarks and training model

1) Collect sequences:
- Run the dev server and open http://localhost:5000/collect
- Enter a label (e.g. "hello"), set frames (e.g. 30) and interval (e.g. 100 ms).
- Click "Record Sequence" and then "Save Last Sequence". Files are saved under `dataset/sequences/` as `label_TIMESTAMP.npy`.

2) Prepare dataset arrays:
```
python tools/prepare_dataset.py --seq-dir dataset/sequences --out dataset/prepared
```
This writes `X.npy`, `y.npy`, and `labels.txt` in the output folder.

3) Train model:
```
python tools/train_model.py --data dataset/prepared --epochs 20 --out static/models/sign_model.h5
```
After training, set `Config.MODEL_PATH` to the saved model path so the server uses the model for real-time detection.

Notes:
- Each saved sequence is an array of shape (T, D) where D is number of landmarks (usually 63).
- You can enable landmark logging during live detection by setting env `LOG_LANDMARKS=1` and inspecting `instance/landmarks/`.
