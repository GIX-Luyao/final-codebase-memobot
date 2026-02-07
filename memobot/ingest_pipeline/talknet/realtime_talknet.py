import torch
import numpy as np
import cv2
import python_speech_features
from talkNet import talkNet
import sys
import os

class RealtimeTalkNet:
    def __init__(self, model_path, device='cpu'):
        self.device = device
        # Ensure correct working directory for imports if needed
        # Assuming talkNet is in the same directory or path is set up correctly
        self.s = talkNet()
        self.s.loadParameters(model_path)
        self.s.eval()
        # Optimize model for inference
        if device == 'cuda':
            self.s.cuda()
        elif device == 'mps':
            self.s.to('mps')
        
    def preprocess_audio(self, audio_int16_chunk):
        """
        Input: numpy array of int16 audio (expecting ~1 second / 16000 samples)
        Output: torch tensor for TalkNet
        """
        # MFCC parameters as defined in TalkNet demo
        mfcc = python_speech_features.mfcc(audio_int16_chunk, 16000, numcep=13, winlen=0.025, winstep=0.010)
        
        # TalkNet expects specific input dimensions. 
        # For 1 second, we usually get ~100 MFCC frames.
        # Ensure we match the model's expected input size
        metrics = mfcc.shape[0]
        length = min(metrics, 100) # Cap at 100 frames (1 second)
        
        mfcc_feature = mfcc[:length, :]
        return torch.FloatTensor(mfcc_feature).unsqueeze(0).to(self.device)

    def preprocess_video_buffer(self, video_frames, face_bbox):
        """
        Input: 
            video_frames: List of last 25 full frames (numpy arrays)
            face_bbox: (x1, y1, x2, y2) of the face in the CURRENT frame
        Output: 
            torch tensor of grayscale face crops (1, 25, 112, 112)
        """
        x1, y1, x2, y2 = face_bbox
        w = x2 - x1
        h = y2 - y1
        
        # Calculate center
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Define a square crop size, adding some margin (TalkNet training setup)
        size = int(max(w, h) * 1.4) 
        half_size = size // 2
        
        # ROI coordinates
        roi_x1 = cx - half_size
        roi_y1 = cy - half_size
        roi_x2 = roi_x1 + size
        roi_y2 = roi_y1 + size

        prepared_faces = []
        
        for frame in video_frames:
            # Grayscale required by TalkNet
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Safe crop with padding if out of bounds
            h_img, w_img = gray.shape
            
            # Pad logic to prevent crash at edges
            top = max(0, -roi_y1)
            bottom = max(0, roi_y2 - h_img)
            left = max(0, -roi_x1)
            right = max(0, roi_x2 - w_img)
            
            if top > 0 or bottom > 0 or left > 0 or right > 0:
                gray = cv2.copyMakeBorder(gray, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
                crop = gray[roi_y1+top:roi_y2+bottom, roi_x1+left:roi_x2+right]
            else:
                crop = gray[roi_y1:roi_y2, roi_x1:roi_x2]

            try:
                # TalkNet expects 224x224 input, then it centers crops to 112x112 internally usually,
                # but let's stick to the demo logic: Resize to 224x224
                # Note: If the corp is empty or invalid, this will throw
                if crop.size == 0:
                     prepared_faces.append(np.zeros((112, 112), dtype=np.uint8))
                     continue

                resized = cv2.resize(crop, (224, 224))
                # Center crop to 112x112 as per evaluation logic in demo
                center_crop = resized[56:168, 56:168] 
                prepared_faces.append(center_crop)
            except Exception:
                # Fallback blank frame if crop failed
                prepared_faces.append(np.zeros((112, 112), dtype=np.uint8))

        if not prepared_faces:
             return None

        # Stack and conversions
        video_tensor = np.array(prepared_faces) # (25, 112, 112)
        return torch.FloatTensor(video_tensor).unsqueeze(0).to(self.device)

    def predict_active_speaker(self, video_frames, audio_chunk, face_bboxes):
        """
        Main entry point.
        video_frames: List of 25 numpy frames.
        audio_chunk: Numpy array of int16 audio samples (approx 16000 long).
        face_bboxes: List of [x1, y1, x2, y2] for all faces detected in the current frame.
        
        Returns:
            The face_bbox of the ACTIVE speaker, or None if no one is speaking.
        """
        
        if len(video_frames) < 20: 
            return None # Not enough history

        audio_input = self.preprocess_audio(audio_chunk)
        
        best_score = -100
        best_bbox = None

        with torch.no_grad():
            for bbox in face_bboxes:
                video_input = self.preprocess_video_buffer(video_frames, bbox)
                if video_input is None: continue

                # Inference
                # Look at demoTalkNet logic: forward_audio_frontend -> visual -> cross -> backend
                embedA = self.s.model.forward_audio_frontend(audio_input)
                embedV = self.s.model.forward_visual_frontend(video_input)
                embedA, embedV = self.s.model.forward_cross_attention(embedA, embedV)
                out = self.s.model.forward_audio_visual_backend(embedA, embedV)
                
                # Out is usually (1, 2) active/inactive or score
                # TalkNet "score" in demo is lossAV.forward(out), which returns confidence
                # Here we simplify: The output 0 is inactive, 1 is active usually.
                # Let's assume out[0][1] is the "speaking" score.
                
                # Check output shape of your specific TalkNet model version
                if out is None or out.numel() == 0:
                    continue

                # Handle different output shapes safely
                # Case 1: (Batch, Time, Classes) - Take mean over time for class 1
                if len(out.shape) == 3 and out.shape[-1] >= 2:
                    # avg_score = out[0, :, 1].mean().item()
                    # Or max score? Active bits might be short. 
                    # Let's take the max of the active class probability
                    score = torch.max(out[0, :, 1]).item()
                
                # Case 2: (Batch, Classes) - Pooled output
                elif len(out.shape) == 2 and out.shape[-1] >= 2:
                    score = out[0][1].item()
                    
                # Case 3: (Batch, Time, 1) or (Batch, 1) - Binary score
                else: 
                    # Flatten and take max
                    score = torch.max(out).item()
                
                # Debug print to help tune
                if score > 0.0:
                     print(f"[TalkNet] Face {bbox} Score: {score:.4f} Shape: {out.shape}")
                
                if score > best_score:
                    best_score = score
                    best_bbox = bbox

        # Threshold (tune as needed, 0.0 or 0.5 depending on loss type)
        if best_score > 0.1: # Try low positive threshold for logits
            return best_bbox
            
        return None
