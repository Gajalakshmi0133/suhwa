import io
import cv2
import numpy as np
from .model_loader import load_model
from PIL import Image
import os
from ..utils.nlp_utils import translate_glosses_to_english

# Lazy-mediapipe loader: mediapipe/TensorFlow can have incompatible protobuf
# requirements with other packages. Import mediapipe only when needed so the
# Flask app can still run in environments without TF installed.
_MP = None
_mp_hands = None
_PERSISTENT_HANDS = None
Hands = None
_LAST_HAND_COUNT = 0

def _ensure_mediapipe():
    global _MP, _mp_hands, _PERSISTENT_HANDS, Hands
    if _MP is not None:
        return True
    try:
        import mediapipe as mp
        _MP = mp
        _mp_hands = mp.solutions.hands
        Hands = _mp_hands.Hands
        # create persistent hands detector for stream usage
        _PERSISTENT_HANDS = _mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        return True
    except Exception as e:
        # mediapipe (or its dependencies) unavailable; log once and continue
        print('Mediapipe not available:', e)
        _MP = None
        _mp_hands = None
        _PERSISTENT_HANDS = None
        Hands = None
        return False

# Default class names for individual alphabets (0-9, A-Z, del, space)
DEFAULT_CLASS_NAMES = [str(i) for i in range(10)] + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["del", "space"]

def _landmarks_from_results(results):
    if not results or not getattr(results, 'multi_hand_landmarks', None):
        return None
    
    lm = []
    # Extract landmarks for all detected hands (up to 2)
    for hand_landmarks in results.multi_hand_landmarks:
        for p in hand_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    
    # If only one hand detected, we might want to pad with zeros or just return the 63 landmarks
    # depending on what the model expects. For now, we return what we found.
    return np.array(lm, dtype=np.float32)


def _is_image_model(model):
    """Heuristic: determine whether a model (Keras or YOLO) expects image-shaped input."""
    # Check for YOLO predictor wrapper
    if hasattr(model, '__class__') and model.__class__.__name__ == 'YOLOPredictor':
        return True

    try:
        shape = getattr(model, 'input_shape', None)
        if not shape:
            return False
        if isinstance(shape, list):
            shape = shape[0]
        if len(shape) == 4:
            return True
        if len(shape) == 2 and (shape[1] in (28*28, 784)):
            return True
    except Exception:
        pass
    return False


def extract_hand_landmarks_from_image_bgr(bgr_image):
    """
    Given an OpenCV BGR image (single static image), return normalized
    landmark array or None. This keeps backward compatibility but uses a
    fresh Hands instance optimized for static images.
    """
    if not _ensure_mediapipe():
        return None
    try:
        with Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5) as hands:
            image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)
            return _landmarks_from_results(results)
    except Exception as e:
        print('Error extracting hand landmarks (static):', e)
        return None


def extract_hand_landmarks_from_frame(bgr_image):
    """
    Given an OpenCV BGR image captured from a video stream, use a
    persistent Hands detector for better real-time performance and
    tracking. Returns normalized landmark array or None.
    """
    if not _ensure_mediapipe() or _PERSISTENT_HANDS is None:
        return None
    try:
        image_rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        results = _PERSISTENT_HANDS.process(image_rgb)
        # record last seen hand count for callers that may use it
        global _LAST_HAND_COUNT
        _LAST_HAND_COUNT = 0
        if results and getattr(results, 'multi_hand_landmarks', None):
            _LAST_HAND_COUNT = len(results.multi_hand_landmarks)
        return _landmarks_from_results(results)
    except Exception as e:
        print('Error extracting hand landmarks (stream):', e)
        return None

def _count_fingers(lm):
    """
    lm: flattened landmark array [x0, y0, z0, x1, y1, z1, ...]
    Returns string representation of number (1-5) or None.
    This is now less aggressive to avoid false positives during general sign recognition.
    """
    if lm is None or len(lm) < 63:
        return None
    
    # helper to get x,y for a landmark index
    def get_pt(idx):
        return lm[idx*3], lm[idx*3+1]

    # Tips and corresponding joints
    tips = [8, 12, 16, 20] # Index, Middle, Ring, Pinky
    count = 0
    for tip in tips:
        tip_x, tip_y = get_pt(tip)
        pip_x, pip_y = get_pt(tip-2)
        # Check if finger is significantly extended
        if tip_y < (pip_y - 0.05):
            count += 1
    
    # Thumb: compare X relative to IP joint (3) and base (2)
    thumb_tip_x, thumb_tip_y = get_pt(4)
    thumb_ip_x, thumb_ip_y = get_pt(3)
    wrist_x, _ = get_pt(0)
    
    if abs(thumb_tip_x - wrist_x) > abs(thumb_ip_x - wrist_x) + 0.05:
        count += 1
        
    # Only return a number if it looks like a clear number gesture (very high confidence)
    # and we don't have a better model prediction.
    if count > 0:
        return str(count)
    return None

def predict_from_image_bytes(image_bytes, model_path=None):
    """
    Input: bytes of an image (PNG/JPEG)
    Output: a dictionary or string prediction
    """
    # Try using image-based CNN when appropriate.
    # If the configured model is an image model, prefer the image predictor.
    try:
        model = load_model(model_path) if model_path else None
    except Exception:
        model = None

    # If a loaded model looks like an image model, delegate to image predictor
    if model is not None and _is_image_model(model):
        try:
            from . import predict_image as _imgpred
            return _imgpred.predict_from_image_bytes(image_bytes, model_path=model_path)
        except Exception as e:
            # fall through to landmark-based approach if image predictor unavailable
            print('Image predictor failed:', e)

    # Otherwise attempt landmarks-based prediction first
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img_np = np.array(image)[:, :, ::-1]  # RGB -> BGR for OpenCV
    lm = extract_hand_landmarks_from_image_bgr(img_np)
    if lm is None:
        # If landmark extraction failed, try image predictor as a fallback
        try:
            from . import predict_image as _imgpred
            return _imgpred.predict_from_image_bytes(image_bytes, model_path=model_path)
        except Exception:
            return {'label': None, 'confidence': 0.0, 'note': 'No hand detected', 'landmarks': None}

    # If we have landmarks, and a model exists, use it for landmark-based prediction
    if model is None:
        # Try counting fingers first
        num_label = _count_fingers(lm)
        if num_label:
            return {'label': num_label, 'sentence': num_label + '.', 'confidence': 0.9, 'note': 'Rule-based number detection'}

        # Fallback: simple classification based on hand position
        wrist_x, wrist_y = lm[0], lm[1]  # Assuming normalized 0-1
        if wrist_x < 0.4:
            label = 'hello'
        elif wrist_x > 0.6:
            label = 'thank you'
        elif wrist_y < 0.4:
            label = 'help'
        else:
            label = 'please'
        confidence = 0.8
        sentence = label.capitalize() + '.'
        return {'label': label, 'sentence': sentence, 'confidence': confidence, 'note': 'Using fallback classification (no model loaded)'}

    # Preprocess for model: depends on your training. Example: reshape
    x = lm.reshape(1, -1)  # (1, N)
    preds = model.predict(x)
    idx = int(np.argmax(preds, axis=1)[0])
    confidence = float(np.max(preds))
    class_names = getattr(model, 'class_names', DEFAULT_CLASS_NAMES)
    if idx < len(class_names):
        label = class_names[idx]
    else:
        label = 'unknown'
    sentence = label.capitalize() + '.'
    return {'label': label, 'sentence': sentence, 'confidence': confidence}

def predict_from_video_file(video_path, model_path=None, max_frames=500, prediction_interval=5):
    """
    Process video file, predicts words at intervals and combines into a sentence.
    Tries both landmark and image-based prediction for robustness.
    """
    cap = cv2.VideoCapture(video_path)
    words = []
    confidences = []
    timeline = [] # Store {timestamp, word}
    frame_count = 0
    model = load_model(model_path) if model_path else None
    
    # Ensure labels are loaded for the model
    if model and model_path:
        labels_path = os.path.join(os.path.dirname(model_path), 'labels.txt')
        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                model.class_names = [line.strip() for line in f.readlines() if line.strip()]

    # Get FPS for timestamp calculation
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30 # Fallback
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        
        # Predict every prediction_interval frames
        if frame_count % prediction_interval != 0:
            continue
            
        # Use auto-predictor which tries landmarks FIRST, then CNN if landmarks fail
        res = predict_auto_from_bgr_frame(frame, model=model)
        
        label = res.get('label')
        conf = res.get('confidence', 0.0)
        
        # Only add if we actually detected something with reasonable confidence
        if label and label != 'unknown' and conf > 0.2:
            # Avoid adding the same word repeatedly if it's detected in consecutive samples
            if not words or words[-1] != label:
                words.append(label)
                confidences.append(conf)
                # Calculate timestamp in seconds
                timestamp = round(frame_count / fps, 2)
                timeline.append({'time': timestamp, 'word': label})
                
    cap.release()
    
    if not words:
        return {
            'label': None, 
            'words': [], 
            'sentence': None, 
            'confidence': 0.0, 
            'confidences': [], 
            'timeline': [],
            'note': 'No hand action or signs detected in video',
            'frames_analyzed': frame_count
        }
        
    # Combine words into sentence
    sentence = ' '.join(words).capitalize() + '.'
    # Translate to proper English
    english_translation = translate_glosses_to_english(words)
    
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0
    
    return {
        'label': sentence, 
        'words': words, 
        'sentence': sentence, 
        'english_translation': english_translation,
        'confidence': avg_confidence, 
        'confidences': confidences, 
        'timeline': timeline,
        'predictions_count': len(words), 
        'frames_analyzed': frame_count
    }


def predict_from_landmarks(lm, model=None):
    """Predict from a flattened landmark array [x,y,z, ...]."""
    if lm is None:
        return {'label': None, 'confidence': 0.0, 'note': 'No landmarks'}

    # Try counting fingers for numbers first
    num_label = _count_fingers(lm)
    
    # Heuristic fallback logic
    def get_heuristic_label(lm_array):
        wrist_x, wrist_y = lm_array[0], lm_array[1]
        thumb_tip_y = lm_array[4*3+1]
        index_tip_y = lm_array[8*3+1]
        
        if wrist_y < 0.3: return 'hello'
        if thumb_tip_y > wrist_y: return 'please'
        if index_tip_y < 0.2: return 'help'
        if wrist_x < 0.3: return 'thank you'
        if wrist_x > 0.7: return 'sorry'
        return 'yes'

    if model is None:
        if num_label:
            return {'label': num_label, 'sentence': num_label + '.', 'confidence': 0.9, 'note': 'Rule-based number'}
        
        label = get_heuristic_label(lm)
        return {'label': label, 'sentence': label.capitalize() + '.', 'confidence': 0.5, 'note': 'Heuristic fallback (no model)'}

    try:
        # Check if model is likely an image model
        if _is_image_model(model):
            # Cannot use landmark array directly on image model
            if num_label:
                return {'label': num_label, 'sentence': num_label + '.', 'confidence': 0.9, 'note': 'Rule-based number (image model fallback)'}
            label = get_heuristic_label(lm)
            return {'label': label, 'sentence': label.capitalize() + '.', 'confidence': 0.5, 'note': 'Heuristic fallback (image model incompatible)'}

        # Prepare input shape based on model expectations
        try:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            
            # Expected input feature dimension
            expected_dim = input_shape[-1]
            
            # Adjust lm to match expected_dim
            current_lm = lm
            if len(lm) > expected_dim:
                current_lm = lm[:expected_dim]
            elif len(lm) < expected_dim:
                padded = np.zeros(expected_dim, dtype=np.float32)
                padded[:len(lm)] = lm
                current_lm = padded

            if len(input_shape) == 3:
                # Sequence model (batch, T, D)
                T = input_shape[1]
                D = input_shape[2]
                x = current_lm.reshape(1, 1, -1)
                # If model expects more than 1 frame, we pad to T
                if T > 1:
                    padded_x = np.zeros((1, T, D), dtype=np.float32)
                    padded_x[0, 0, :] = current_lm
                    x = padded_x
            else:
                x = current_lm.reshape(1, -1)
        except Exception:
            # Fallback if shape inspection fails
            x = lm[:63].reshape(1, -1) if len(lm) >= 63 else lm.reshape(1, -1)

        preds = model.predict(x)
        idx = int(np.argmax(preds, axis=1)[0])
        confidence = float(np.max(preds))
        
        class_names = getattr(model, 'class_names', DEFAULT_CLASS_NAMES)
        label = class_names[idx] if idx < len(class_names) else 'unknown'
        
        # If we have a model, trust its prediction more unless it's extremely low confidence
        if confidence < 0.2 and num_label:
            label = num_label
            confidence = 0.8
            
        return {'label': label, 'sentence': label.capitalize() + '.', 'confidence': confidence}
    except Exception as e:
        print(f"DEBUG: predict_from_landmarks error: {e}")
        # Fallback to heuristic on error
        if num_label:
            return {'label': num_label, 'sentence': num_label + '.', 'confidence': 0.9, 'note': 'Rule-based number (error fallback)'}
        label = get_heuristic_label(lm)
        return {'label': label, 'sentence': label.capitalize() + '.', 'confidence': 0.5, 'note': f'Heuristic fallback (error: {str(e)[:30]})'}

def predict_from_bgr_frame(frame_bgr, model=None):
    """
    Predict from a single OpenCV BGR frame using the stream-optimized
    extractor. `model` should be a loaded Keras model (or None to use
    fallback heuristics). Returns a dict: {label, confidence, sentence, note}.
    """
    # If model provided appears to be an image model (CNN), try it FIRST before landmarks
    # as landmark extraction (Mediapipe) might be failing due to environment issues.
    if model is not None and _is_image_model(model):
        try:
            from . import predict_image as _imgpred
            res = _imgpred.predict_from_bgr_frame(frame_bgr, model=model)
            if res.get('label') is not None and res.get('confidence', 0) > 0.4:
                res['note'] = 'CNN Prediction'
                res['hand_count'] = _LAST_HAND_COUNT
                return res
        except Exception as e:
            print('Image-predictor prioritized attempt failed:', e)

    lm = extract_hand_landmarks_from_frame(frame_bgr)
    
    # If landmarks could not be found, try using the image-based predictor (as fallback)
    if lm is None:
        print("DEBUG: No landmarks found, falling back to image-based predictor")
        try:
            from . import predict_image as _imgpred
            return _imgpred.predict_from_bgr_frame(frame_bgr, model=model)
        except Exception as e:
            print(f"DEBUG: Image predictor fallback error: {e}")
            return {'label': None, 'confidence': 0.0, 'note': 'No hand detected', 'hand_count': _LAST_HAND_COUNT}

    # Landmark-based model prediction (default path)
    res = predict_from_landmarks(lm, model=model)
    res['hand_count'] = _LAST_HAND_COUNT
    return res


def predict_auto_from_bgr_frame(frame_bgr, model=None, model_path=None):
    """Convenience wrapper used by the app: tries landmark-based prediction
    first, and falls back to the image CNN if landmarks are not detected or
    if the provided model appears to expect image input."""
    # Prefer landmark flow; predict_from_bgr_frame already tries image fallback
    return predict_from_bgr_frame(frame_bgr, model=model)
