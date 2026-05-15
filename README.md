# CV Robot

CV Robot is a browser-based object detection app built for webcam and image uploads. It recognizes one main object at a time, focuses on small school items like notebooks, pens, pencils, markers, and erasers, and keeps a human-safety block in the response.

## Web App

This repo now includes a Render-ready Flask app:

- `app.py`: Flask server and detection API
- `detector.py`: YOLO-World detection logic
- `templates/index.html`: browser UI
- `static/`: frontend JS and CSS

## Local Run

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:10000`.

## Render Deploy

This repo includes both `render.yaml` and `.python-version`.

If you deploy manually on Render:

- Runtime: `Python`
- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn app:app`

If you use the Blueprint flow, Render can read `render.yaml` directly from the repo root.

## Notes

- Browser camera access usually requires HTTPS or localhost.
- The app returns one primary detection at a time.
- If an object is too close to a person, the object name can still be shown, but it remains safety-blocked.
