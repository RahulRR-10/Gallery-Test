# AI Photo Search Mobile App (React Native CLI)

Features
- Semantic photo search and gallery
- Full photo viewer with objects and faces
- People (face clusters), Groups, Relationships lists
- Settings with backend URL tester and in-app user guide

Backend
- Start: `python backend/start_api.py --host 0.0.0.0 --port 8000`
- Use your PC LAN IP in the app Settings (e.g., http://192.168.1.10:8000)

Run
```bash
cd frontend
npm install
npm run start # start Metro
npm run android # or npm run ios
```

Notes
- Android/iOS dev allow HTTP to LAN for convenience. Use HTTPS for production.

### User Guide

#### 1) Prerequisites
- Backend: Python 3.10+, install deps: `pip install -r backend/requirements.txt`
- Mobile: Node 20+, Android Studio (for Android) and/or Xcode (for iOS)

#### 2) Start the Backend
```bash
python backend/start_api.py --host 0.0.0.0 --port 8000
```
- Ensure your photos (or test images) are in `sample_photos/` so the app can load `/images/{filename}`.
- Find your PC’s IP: `ipconfig` (Windows). Example: `192.168.1.10`.
- Allow TCP port 8000 in your firewall for LAN access.

#### 3) Run the App
Android (device or emulator):
```bash
cd frontend
npm install
npm run start
npm run android
```

iOS (macOS, Simulator or device):
```bash
cd frontend
npm install
# First time iOS setup
cd ios && bundle install && bundle exec pod install && cd ..
npm run start
npm run ios
```

#### 4) Configure the Backend URL in the App
- Open the app → Settings → set Backend URL to `http://YOUR_PC_IP:8000` (e.g., `http://192.168.1.10:8000`).
- Tap “Test Connection” and verify status is healthy.

#### 5) Using the App
- Search: Enter a query (e.g., “sunset beach”) and optional time filter (e.g., “last month”), then Search.
- Gallery: Opens a grid of results; tap a photo to view details.
- Photo Viewer: Shows the full image, objects, and faces detected.
- People: Lists face clusters discovered by the backend.
- Groups: Lists people groups from the backend.
- Relationships: Lists inferred relationships with confidence scores.

Background tasks (run via backend):
- Indexing: POST `/api/index` with a directory then poll `/api/tasks/{id}`.
- Face clustering: POST `/api/faces/cluster`, then GET `/api/faces/clusters`.
- Build relationships: POST `/api/relationships/build`.

#### 6) Troubleshooting
- Connection fails: confirm phone and PC are on the same Wi‑Fi, firewall allows port 8000, URL uses your LAN IP.
- Images not loading: confirm files exist under `sample_photos/` and filenames from search results match `/images/{filename}`.
- iOS device HTTP issues: consider a tunnel (e.g., `npx localtunnel --port 8000`) and use the provided HTTPS URL temporarily.
- Metro issues: stop all node processes, then `npm run start -- --reset-cache`.

#### 7) Production Notes
- Replace HTTP with HTTPS; remove cleartext/ATS relaxations.
- Add auth if exposing the API outside your LAN.
