let video = document.getElementById('webcam');
let startBtn = document.getElementById('startBtn');
let snapBtn = document.getElementById('snapBtn');
let stopBtn = document.getElementById('stopBtn');
let predictionBox = document.getElementById('predictionBox');

let streamRef = null;

startBtn && startBtn.addEventListener('click', async () => {
  if (!navigator.mediaDevices) {
    alert("getUserMedia not supported");
    return;
  }
  try {
    streamRef = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = streamRef;
  } catch (err) {
    console.error(err);
    alert("Camera access denied or not available.");
  }
});

stopBtn && stopBtn.addEventListener('click', () => {
  if (streamRef) {
    streamRef.getTracks().forEach(t => t.stop());
    video.srcObject = null;
  }
});

snapBtn && snapBtn.addEventListener('click', async () => {
  if (!video || !video.srcObject) {
    alert("Start the camera first.");
    return;
  }
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async (blob) => {
    const fd = new FormData();
    fd.append('image', blob, 'snapshot.png');
    try{
      const langEl = document.getElementById('languageSelect');
      if(langEl && langEl.value) fd.append('lang', langEl.value);
    }catch(e){}
    try {
      predictionBox.innerText = "Sending...";
      const res = await fetch('/api/predict-image', {
        method: 'POST',
        body: fd
      });
      const data = await res.json();
      if (data.success) {
        predictionBox.innerText = JSON.stringify(data.prediction, null, 2);
      } else {
        predictionBox.innerText = "Error: " + (data.error || "Unknown");
      }
    } catch (err) {
      predictionBox.innerText = "Request failed: " + err;
    }
  }, 'image/png');
});
# Placeholder for static/js/detect.js
