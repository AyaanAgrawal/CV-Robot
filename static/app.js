const camera = document.getElementById("camera");
const captureCanvas = document.getElementById("captureCanvas");
const startCameraButton = document.getElementById("startCameraButton");
const captureButton = document.getElementById("captureButton");
const uploadInput = document.getElementById("uploadInput");
const statusCard = document.getElementById("statusCard");
const resultDetails = document.getElementById("resultDetails");
const resultLabel = document.getElementById("resultLabel");
const resultConfidence = document.getElementById("resultConfidence");
const resultSafety = document.getElementById("resultSafety");
const resultSummary = document.getElementById("resultSummary");
const annotatedImage = document.getElementById("annotatedImage");

let stream = null;

function setStatus(kind, message) {
  statusCard.className = `status ${kind}`;
  statusCard.textContent = message;
}

function updateResult(payload) {
  const detection = payload.detection;
  if (!detection) {
    resultDetails.hidden = true;
    annotatedImage.hidden = true;
    setStatus("idle", "No valid target found.");
    return;
  }

  resultLabel.textContent = detection.label;
  resultConfidence.textContent = `${detection.confidence_percent}%`;
  resultSafety.textContent = detection.blocked_by_human ? "Blocked by safety" : "Clear";
  resultSummary.textContent = detection.description;
  resultDetails.hidden = false;

  annotatedImage.src = payload.annotated_image;
  annotatedImage.hidden = false;
  setStatus(
    detection.blocked_by_human ? "blocked" : "clear",
    detection.blocked_by_human
      ? `Detected ${detection.label}, but safety is blocking action.`
      : `Detected ${detection.label} successfully.`
  );
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment" },
      audio: false,
    });
    camera.srcObject = stream;
    captureButton.disabled = false;
    setStatus("idle", "Camera is ready.");
  } catch (error) {
    setStatus("error", `Camera error: ${error.message}`);
  }
}

function captureFrameBlob() {
  const width = camera.videoWidth || 960;
  const height = camera.videoHeight || 720;
  captureCanvas.width = width;
  captureCanvas.height = height;
  const context = captureCanvas.getContext("2d");
  context.drawImage(camera, 0, 0, width, height);

  return new Promise((resolve, reject) => {
    captureCanvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error("Could not capture frame."));
    }, "image/jpeg", 0.9);
  });
}

async function sendImage(blob, filename) {
  setStatus("loading", "Analyzing image...");
  const formData = new FormData();
  formData.append("image", blob, filename);

  const response = await fetch("/api/detect", {
    method: "POST",
    body: formData,
  });

  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || "Detection failed.");
  }

  updateResult(payload);
}

startCameraButton.addEventListener("click", async () => {
  await startCamera();
});

captureButton.addEventListener("click", async () => {
  try {
    const blob = await captureFrameBlob();
    await sendImage(blob, "camera.jpg");
  } catch (error) {
    setStatus("error", error.message);
  }
});

uploadInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }

  try {
    await sendImage(file, file.name);
  } catch (error) {
    setStatus("error", error.message);
  }
});
