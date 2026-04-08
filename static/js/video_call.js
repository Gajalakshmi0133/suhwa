const socket = io();
const localVideo = document.getElementById('localVideo');
const remoteVideo = document.getElementById('remoteVideo');
const speechSubtitles = document.getElementById('speechSubtitles');
const signSubtitles = document.getElementById('signSubtitles');
const captionsLog = document.getElementById('captionsLog');

let localStream;
let peerConnection;
const config = {
    iceServers: [
        { urls: 'stun:stun.l.google.com:19302' },
        { urls: 'stun:stun1.l.google.com:19302' }
    ]
};

// --- WebRTC Logic ---

async function startCall() {
    try {
        localStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
        localVideo.srcObject = localStream;

        socket.emit('join', { room: ROOM_ID });

        setupSignDetection();
        setupSpeechToText();
    } catch (err) {
        console.error('Error accessing media devices:', err);
    }
}

socket.on('user-joined', async (data) => {
    console.log('User joined:', data.userId);
    createPeerConnection(data.userId);
    const offer = await peerConnection.createOffer();
    await peerConnection.setLocalDescription(offer);
    socket.emit('offer', { room: ROOM_ID, offer, to: data.userId });
});

socket.on('offer', async (data) => {
    console.log('Received offer from:', data.userId);
    createPeerConnection(data.userId);
    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.offer));
    const answer = await peerConnection.createAnswer();
    await peerConnection.setLocalDescription(answer);
    socket.emit('answer', { room: ROOM_ID, answer, to: data.userId });
});

socket.on('answer', async (data) => {
    console.log('Received answer');
    await peerConnection.setRemoteDescription(new RTCSessionDescription(data.answer));
});

socket.on('ice-candidate', async (data) => {
    try {
        await peerConnection.addIceCandidate(new RTCIceCandidate(data.candidate));
    } catch (e) {
        console.error('Error adding ice candidate', e);
    }
});

function createPeerConnection(userId) {
    peerConnection = new RTCPeerConnection(config);
    
    localStream.getTracks().forEach(track => {
        peerConnection.addTrack(track, localStream);
    });

    peerConnection.ontrack = (event) => {
        remoteVideo.srcObject = event.streams[0];
    };

    peerConnection.onicecandidate = (event) => {
        if (event.candidate) {
            socket.emit('ice-candidate', { room: ROOM_ID, candidate: event.candidate, to: userId });
        }
    };
}

// --- Sign Detection (MediaPipe + Server API) ---

let hands;
let camera;
let lastSignText = "";

function setupSignDetection() {
    hands = new Hands({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    });

    hands.setOptions({
        maxNumHands: 2,
        modelComplexity: 1,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    hands.onResults(onHandResults);

    camera = new Camera(localVideo, {
        onFrame: async () => {
            await hands.send({ image: localVideo });
        },
        width: 640,
        height: 480
    });
    camera.start();
}

async function onHandResults(results) {
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        try {
            const lang = document.getElementById('languageSelect')?.value || 'asl';
            const response = await fetch('/api/detect-stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    landmarks: results.multiHandLandmarks,
                    lang: lang
                })
            });
            const data = await response.json();
            if (data.success && data.subtitle && data.subtitle !== lastSignText) {
                lastSignText = data.subtitle;
                displaySubtitle(data.subtitle, 'sign', 'You');
                socket.emit('subtitle', { room: ROOM_ID, text: data.subtitle, type: 'sign', user: 'Partner' });
            }
        } catch (err) {
            console.error('Sign detection API error:', err);
        }
    }
}

// --- Speech to Text (Web Speech API) ---

function setupSpeechToText() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('Web Speech API not supported in this browser.');
        return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
            if (event.results[i].isFinal) {
                finalTranscript += event.results[i][0].transcript;
            } else {
                interimTranscript += event.results[i][0].transcript;
            }
        }

        const text = finalTranscript || interimTranscript;
        if (text) {
            displaySubtitle(text, 'speech', 'You');
            socket.emit('subtitle', { room: ROOM_ID, text: text, type: 'speech', user: 'Partner' });
        }
    };

    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
    };

    recognition.onend = () => {
        recognition.start(); // Keep it running
    };

    recognition.start();
}

// --- Subtitle Management ---

function displaySubtitle(text, type, sender) {
    const el = type === 'speech' ? speechSubtitles : signSubtitles;
    el.innerText = `${sender}: ${text}`;
    el.style.display = 'block';

    // Log the caption
    const logItem = document.createElement('div');
    logItem.className = `mb-1 ${type === 'speech' ? 'text-primary' : 'text-success'}`;
    logItem.innerHTML = `<strong>${sender} (${type}):</strong> ${text}`;
    captionsLog.prepend(logItem);

    // Hide after some time if no new results
    clearTimeout(el.timeout);
    el.timeout = setTimeout(() => {
        el.style.display = 'none';
    }, 5000);
}

socket.on('subtitle', (data) => {
    displaySubtitle(data.text, data.type, 'Partner');
});

// --- Controls ---

document.getElementById('toggleMic').addEventListener('click', () => {
    const audioTrack = localStream.getAudioTracks()[0];
    audioTrack.enabled = !audioTrack.enabled;
    document.getElementById('toggleMic').classList.toggle('btn-danger', !audioTrack.enabled);
    document.getElementById('toggleMic').innerHTML = audioTrack.enabled ? '<i class="fas fa-microphone"></i>' : '<i class="fas fa-microphone-slash"></i>';
});

document.getElementById('toggleVideo').addEventListener('click', () => {
    const videoTrack = localStream.getVideoTracks()[0];
    videoTrack.enabled = !videoTrack.enabled;
    document.getElementById('toggleVideo').classList.toggle('btn-danger', !videoTrack.enabled);
    document.getElementById('toggleVideo').innerHTML = videoTrack.enabled ? '<i class="fas fa-video"></i>' : '<i class="fas fa-video-slash"></i>';
});

document.getElementById('endCall').addEventListener('click', () => {
    window.location.href = '/dashboard';
});

// Initialize
startCall();
