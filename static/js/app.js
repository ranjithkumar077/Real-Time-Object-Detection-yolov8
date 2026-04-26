const confidence = document.getElementById("confidence");
const confidenceValue = document.getElementById("confidenceValue");
const statusText = document.getElementById("statusText");
const totalObjects = document.getElementById("totalObjects");
const fpsValue = document.getElementById("fpsValue");
const classCount = document.getElementById("classCount");
const summaryList = document.getElementById("summaryList");
const loading = document.getElementById("loading");
const toast = document.getElementById("toast");
const qualityMode = document.getElementById("qualityMode");
const modelPreset = document.getElementById("modelPreset");
let activeModel = "yolov8s.pt";
let customModelActive = false;
let statusTimer = null;
let lastVoiceAlert = 0;

function showToast(message) {
    toast.textContent = message;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 3200);
}

function setLoading(isLoading, label = "Processing") {
    loading.hidden = !isLoading;
    statusText.textContent = isLoading ? label : "Ready";
}

function updateStats(summary = {}) {
    const counts = summary.counts || {};
    totalObjects.textContent = summary.total || 0;
    fpsValue.textContent = summary.fps || 0;
    classCount.textContent = Object.keys(counts).length;
    summaryList.innerHTML = "";

    if (!Object.keys(counts).length) {
        summaryList.innerHTML = "<p>No detections yet.</p>";
        return;
    }

    Object.entries(counts).sort((a, b) => b[1] - a[1]).forEach(([name, count]) => {
        const row = document.createElement("div");
        row.className = "summary-item";
        row.innerHTML = `<strong>${name}</strong><span>${count}</span>`;
        summaryList.appendChild(row);
    });

    maybeVoiceAlert(counts);
}

function maybeVoiceAlert(counts) {
    const alertClasses = ["person", "car", "cell phone", "mobile"];
    const matched = alertClasses.find((item) => counts[item]);
    const now = Date.now();
    if (!matched || now - lastVoiceAlert < 7000 || !("speechSynthesis" in window)) return;

    const speech = new SpeechSynthesisUtterance(`${matched} detected`);
    speech.rate = 1;
    speech.pitch = 1;
    window.speechSynthesis.speak(speech);
    lastVoiceAlert = now;
}

function renderHistory(rows = []) {
    const body = document.getElementById("historyBody");
    body.innerHTML = "";
    if (!rows.length) {
        body.innerHTML = `<tr><td colspan="6">No detection history yet.</td></tr>`;
        return;
    }
    rows.forEach((row) => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${row.timestamp}</td>
            <td>${row.source}</td>
            <td>${row.file_name}</td>
            <td>${row.total_objects}</td>
            <td>${row.class_counts}</td>
            <td>${row.output_file}</td>
        `;
        body.appendChild(tr);
    });
}

async function refreshHistory() {
    const res = await fetch("/history");
    const data = await res.json();
    renderHistory(data.history || []);
}

async function postJson(url, payload) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Request failed");
    return data;
}

confidence.addEventListener("input", () => {
    confidenceValue.textContent = `${Math.round(Number(confidence.value) * 100)}%`;
});

modelPreset.addEventListener("change", () => {
    activeModel = modelPreset.value;
    customModelActive = false;
    showToast(`${modelPreset.options[modelPreset.selectedIndex].text} selected.`);
});

document.querySelectorAll(".nav-btn").forEach((button) => {
    button.addEventListener("click", () => {
        document.querySelectorAll(".nav-btn").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(button.dataset.target).classList.add("active");
        if (button.dataset.target === "historyPanel") refreshHistory();
    });
});

document.getElementById("fullscreenBtn").addEventListener("click", () => {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen();
    else document.exitFullscreen();
});

document.getElementById("uploadModelBtn").addEventListener("click", async () => {
    const input = document.getElementById("customModel");
    if (!input.files.length) return showToast("Choose a .pt model first.");
    const form = new FormData();
    form.append("model", input.files[0]);
    setLoading(true, "Uploading");
    try {
        const res = await fetch("/upload_model", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        activeModel = data.model;
        customModelActive = true;
        showToast("Custom model loaded successfully.");
    } catch (error) {
        showToast(error.message);
    } finally {
        setLoading(false);
    }
});

document.getElementById("startWebcamBtn").addEventListener("click", async () => {
    setLoading(true, "Starting");
    try {
        await postJson("/webcam/start", {
            camera: Number(document.getElementById("cameraIndex").value || 0),
            confidence: Number(confidence.value),
            model: activeModel,
            quality: qualityMode.value,
        });
        const feed = document.getElementById("webcamFeed");
        feed.src = `/video_feed?ts=${Date.now()}`;
        feed.style.display = "block";
        document.getElementById("webcamEmpty").style.display = "none";
        statusText.textContent = "Live";
        statusTimer = setInterval(async () => {
            const res = await fetch("/webcam/status");
            const data = await res.json();
            updateStats(data.summary);
        }, 1200);
        showToast("Webcam detection started.");
    } catch (error) {
        showToast(error.message);
    } finally {
        loading.hidden = true;
    }
});

document.getElementById("stopWebcamBtn").addEventListener("click", async () => {
    try {
        const data = await postJson("/webcam/stop", {});
        clearInterval(statusTimer);
        statusText.textContent = "Stopped";
        renderHistory(data.history || []);
        showToast("Webcam detection stopped.");
    } catch (error) {
        showToast(error.message);
    }
});

document.getElementById("screenshotBtn").addEventListener("click", async () => {
    try {
        const data = await postJson("/screenshot", {});
        showToast("Screenshot captured.");
        window.open(data.download_url, "_blank");
    } catch (error) {
        showToast(error.message);
    }
});

document.getElementById("imageForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    form.append("confidence", confidence.value);
    form.append("model", activeModel);
    form.append("quality", qualityMode.value);
    setLoading(true, "Detecting");
    try {
        const res = await fetch("/detect_image", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        const image = document.getElementById("imageResult");
        image.src = `${data.output_url}?ts=${Date.now()}`;
        image.style.display = "block";
        image.nextElementSibling.style.display = "none";
        const link = document.getElementById("imageDownload");
        link.href = data.download_url;
        link.hidden = false;
        updateStats(data.summary);
        renderHistory(data.history);
        showToast(data.message);
    } catch (error) {
        showToast(error.message);
    } finally {
        setLoading(false);
    }
});

document.getElementById("videoForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    form.append("confidence", confidence.value);
    form.append("model", activeModel);
    form.append("quality", qualityMode.value);
    setLoading(true, "Processing video");
    try {
        const res = await fetch("/detect_video", { method: "POST", body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error);
        const video = document.getElementById("videoResult");
        video.src = `${data.output_url}?ts=${Date.now()}`;
        video.style.display = "block";
        video.nextElementSibling.style.display = "none";
        const link = document.getElementById("videoDownload");
        link.href = data.download_url;
        link.hidden = false;
        updateStats(data.summary);
        renderHistory(data.history);
        showToast(data.message);
    } catch (error) {
        showToast(error.message);
    } finally {
        setLoading(false);
    }
});

document.getElementById("refreshHistoryBtn").addEventListener("click", refreshHistory);
refreshHistory();
