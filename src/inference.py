import torch
import torch.nn as nn
import cv2
import os
import numpy as np
import pandas as pd
from PIL import Image
import time
import logging
import toml

# Set logging level to suppress warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import necessary modules
from data.face_detection.ibug.face_detection import RetinaFacePredictor
from data.face_detection.ibug.face_detection.utils import SimpleFaceTracker
from ultralytics import YOLO
from data.dataset import convert_mp4_to_mp3, img_processing, pad_wav
from models.architectures import ResNet50, AudioModel, AVTmodel
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

class EmotionRecognition:
    def __init__(self, config):
        self.config = config
        self.device = config['general']['device']
        self.detector = config['general']['detector']
        self.load_models()

    def predict_emotion(self, path, text):
        self.dict_time = {}
        self.text = text
        v_fss = self.load_video_frames(path)
        if v_fss is None:
            print(f"Skipping video {path} due to no faces detected.")
            return None, None, None

        wav, a_fss = self.load_audio_features(path)
        t_fss = self.load_text_features(wav)
        pred_sc = self.predict_single_corpus(a_fss, v_fss, t_fss)
        pred_mc = self.predict_multi_corpus(a_fss, v_fss, t_fss)
        pred_afew = self.predict_afew_corpus(a_fss, v_fss, t_fss)

        top_emotions_sc = self.get_top_emotions(pred_sc)
        top_emotions_mc = self.get_top_emotions(pred_mc)
        top_emotions_afew = self.get_top_emotions(pred_afew)

        return top_emotions_sc, top_emotions_mc, top_emotions_afew

    def get_top_emotions(self, predictions):
        EMOTIONS = self.config['emotion_labels']['labels']
        top_indices = np.argsort(predictions)[0][-2:][::-1]
        top_emotions = [f"{EMOTIONS[index]}: {predictions[0][index]:.2f}" for index in top_indices]
        return ", ".join(top_emotions)

    def load_models(self):
        self.load_video_model()
        self.load_audio_model()
        self.load_text_model()
        self.load_avt_models()
        self.load_data_processor()

    def load_video_model(self):
        self.video_model = ResNet50(num_classes=7, channels=3)
        self.video_model.load_state_dict(torch.load(self.config['models']['video_model_path'], map_location=self.device))
        self.video_model.to(self.device).eval()
        if self.detector == 'retinaface':
            self.face_detector = RetinaFacePredictor(
                threshold=self.config['face_detection']['threshold'],
                device=self.device,
                model=RetinaFacePredictor.get_model('resnet50')
            )
            self.face_tracker = SimpleFaceTracker(
                iou_threshold=self.config['face_detection']['iou_threshold'],
                minimum_face_size=self.config['face_detection']['minimum_face_size']
            )

    def load_audio_model(self):
        path_audio_model = self.config['models']['audio_model_path']
        self.processor = AutoFeatureExtractor.from_pretrained(path_audio_model)
        config = AutoConfig.from_pretrained(path_audio_model)
        config.num_labels = self.config['parameters']['encoder_num_classes']
        self.audio_model = AudioModel.from_pretrained(
            path_audio_model, config=config, ignore_mismatched_sizes=True
        )
        self.audio_model.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)
        self.audio_model.load_state_dict(torch.load(self.config['models']['audio_model_path_2'], map_location=self.device))
        self.audio_model.to(self.device).eval()

    def load_text_model(self):
        path_text_model = self.config['models']['text_model_path']
        self.tokenizer = AutoTokenizer.from_pretrained(path_text_model)
        config = AutoConfig.from_pretrained(path_text_model)
        config.num_labels = self.config['parameters']['encoder_num_classes']
        self.text_model = AutoModelForSequenceClassification.from_pretrained(
            path_text_model, config=config, ignore_mismatched_sizes=True
        )
        self.text_model.load_state_dict(torch.load(self.config['models']['text_model_path_2'], map_location=self.device))
        self.text_model.to(self.device).eval()
        self.s2t = pipeline(
            "automatic-speech-recognition",
            model=self.config['models']['whisper_model'],
            chunk_length_s=self.config['speech_recognition']['chunk_length_s'],
            device=self.device
        )
        self.features = {}
        self.text_model.classifier.dense.register_forward_hook(self.get_activations('features'))

    def load_avt_models(self):
        self.avt_sc_model = AVTmodel(
            512, 1024, 768, gated_dim=32, n_classes=self.config['parameters']['encoder_num_classes'], drop=0
        )
        self.avt_sc_model.load_state_dict(torch.load(self.config['models']['avt_sc_model_path'], map_location=self.device))
        self.avt_sc_model.to(self.device).eval()

        self.avt_mc_model = AVTmodel(
            512, 1024, 768, gated_dim=64, n_classes=self.config['parameters']['mc_num_classes'], drop=0
        )
        self.avt_mc_model.load_state_dict(torch.load(self.config['models']['avt_mc_model_path'], map_location=self.device))
        self.avt_mc_model.to(self.device).eval()

        self.avt_model = AVTmodel(
            512, 1024, 768, gated_dim=64, n_classes=self.config['parameters']['encoder_num_classes'], drop=0
        )
        self.avt_model.load_state_dict(torch.load(self.config['models']['avt_model_path'], map_location=self.device))
        self.avt_model.to(self.device).eval()

    def load_data_processor(self):
        self.step = self.config['parameters']['step']  # sec
        self.window = self.config['parameters']['window']  # sec
        self.need_frames = self.config['parameters']['need_frames']
        self.sr = self.config['parameters']['sampling_rate']

    def get_activations(self, name):
        def hook(model, input, output):
            self.features[name] = output.detach()
        return hook

    def load_video_frames(self, path):
        if self.detector == 'yolo':
            self.face_detector = YOLO(self.config['models']['yolo_weights_path'])
        start_time = time.time()
        window_v = self.window * self.need_frames
        step_v = self.step * self.need_frames
        video_stream = cv2.VideoCapture(path)
        w = int(video_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(video_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
        sec = frame_count / fps

        step = int(fps * 5 / 25)
        count_frame = 0
        faces_images = []

        while True:
            ret, fr = video_stream.read()
            if not ret:
                break
            if count_frame % step == 0:
                faces = torch.zeros((1, 3, 224, 224))
                count_face = 0

                if self.detector == 'retinaface':
                    results = self.face_detector(fr, rgb=False)

                    for face in results:
                        startX, startY, endX, endY = face[:4].astype(int)
                        startX, startY = max(0, startX), max(0, startY)
                        endX, endY = min(w - 1, endX), min(h - 1, endY)
                        curr_fr = fr[startY:endY, startX:endX]
                        curr_fr = cv2.cvtColor(curr_fr, cv2.COLOR_BGR2RGB)
                        cur_face_copy = img_processing(Image.fromarray(curr_fr))
                        faces += cur_face_copy
                        count_face += 1

                    if count_face > 0:
                        faces /= count_face
                        faces_images.append(faces)
                elif self.detector == 'yolo':
                    curr_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
                    results = self.face_detector.track(
                        curr_fr,
                        persist=True,
                        imgsz=640,
                        conf=0.01,
                        iou=0.5,
                        augment=False,
                        device=self.device,
                        verbose=False,
                    )

                    if results[0].boxes.xyxy.cpu().tolist() != []:
                        for i in results[0].boxes:
                            box = i.xyxy.int().cpu().tolist()[0]
                            startX, startY = max(0, box[0]), max(0, box[1])
                            endX, endY = min(w - 1, box[2]), min(h - 1, box[3])

                            face_region = curr_fr[startY:endY, startX:endX]
                            cur_face_copy = img_processing(Image.fromarray(face_region))

                            faces += cur_face_copy
                            count_face += 1

                        if count_face > 0:
                            faces /= count_face
                            faces_images.append(faces)
                else:
                    raise ValueError(f"Unknown detector: {self.detector}")
            count_frame += 1

        video_stream.release()
        if self.detector == 'retinaface':
            self.face_tracker.reset()

        if len(faces_images) == 0:
            # Handle the case where no faces were detected in any frame
            print("No faces detected in the video.")
            return None

        v_fss = torch.cat(faces_images, dim=0)

        batch_size_limit = self.config['parameters']['batch_size_limit']
        num_images = len(v_fss)

        with torch.no_grad():

            if num_images > batch_size_limit:
                all_fss_subbatch = []

                for i in range(0, num_images, batch_size_limit):
                    v_fss_subbatch = v_fss[i : i + batch_size_limit]
                    v_fss_subbatch = self.video_model.extract_features(v_fss_subbatch.to(self.device))
                    all_fss_subbatch.append(v_fss_subbatch)
                v_fss = torch.cat(all_fss_subbatch, dim=0).detach().cpu().numpy()
            else:
                v_fss = self.video_model.extract_features(v_fss.to(self.device)).detach().cpu().numpy()

        segments_v = []
        for start_v in range(0, v_fss.shape[0] + 1, step_v):
            end_v = min(start_v + window_v, v_fss.shape[0])
            segment = v_fss[start_v:end_v, :]
            if end_v - start_v < step_v and start_v != 0:
                break
            segments_v.append(np.mean(segment, axis=0))

        segments_v = np.array(segments_v)

        v_fss = np.hstack((np.mean(segments_v, axis=0), np.std(segments_v, axis=0)))
        v_fss = torch.from_numpy(v_fss)
        v_fss = torch.unsqueeze(v_fss, 0)

        time_video = time.time() - start_time
        self.dict_time['time_video'] = sec % 60
        self.dict_time['time_feature_video'] = time_video % 60

        return v_fss

    def load_audio_features(self, path):
        start_time = time.time()
        window_a = self.window * self.sr
        step_a = self.step * self.sr

        wav = convert_mp4_to_mp3(path, self.sr)

        segments_a = []

        for start_a in range(0, len(wav) + 1, step_a):
            end_a = min(start_a + window_a, len(wav))
            if end_a - start_a < step_a and start_a != 0:
                break
            a_fss_chunk = wav[start_a:end_a]
            a_fss = pad_wav(a_fss_chunk, window_a)
            a_fss = torch.unsqueeze(a_fss, 0)
            a_fss = self.processor(a_fss, sampling_rate=self.sr)
            a_fss = a_fss['input_values'][0]
            segments_a.append(torch.from_numpy(a_fss))

        a_fss = torch.cat(segments_a)
        with torch.no_grad():
            a_fss = self.audio_model.extract_features(a_fss.to(self.device))
            a_fss = a_fss.cpu().numpy()

        a_fss = np.mean(a_fss, axis=1)
        a_fss = np.hstack((np.mean(a_fss, axis=0), np.std(a_fss, axis=0)))
        a_fss = torch.from_numpy(a_fss)
        a_fss = torch.unsqueeze(a_fss, 0)

        time_audio = time.time() - start_time
        self.dict_time['time_feature_audio'] = time_audio % 60
        return wav, a_fss

    def load_text_features(self, wav):
        start_time = time.time()
        if self.text:
            text = self.text
        else:
            text = self.s2t(wav.numpy(), batch_size=8)["text"]

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            outputs = self.text_model(**inputs.to(self.device))
            t_fss = self.features['features']
            self.features.clear()
        time_text = time.time() - start_time
        self.dict_time['time_feature_text'] = time_text % 60
        return t_fss

    def predict_single_corpus(self, a_fss, v_fss, t_fss):
        start_time = time.time()
        with torch.no_grad():
            pred_sc = self.avt_sc_model(
                a_fss.to(self.device), v_fss.to(self.device), t_fss.to(self.device)
            )
        pred_sc = nn.functional.softmax(pred_sc, dim=1).cpu().detach().numpy()
        time_sc = time.time() - start_time
        self.dict_time['time_pred_sc'] = time_sc % 60
        return pred_sc

    def predict_multi_corpus(self, a_fss, v_fss, t_fss):
        start_time = time.time()
        with torch.no_grad():
            pred_mc = self.avt_mc_model(
                a_fss.to(self.device), v_fss.to(self.device), t_fss.to(self.device)
            )
        pred_mc = nn.functional.softmax(pred_mc, dim=1).cpu().detach().numpy()
        time_mc = time.time() - start_time
        self.dict_time['time_pred_mc'] = time_mc % 60
        return pred_mc

    def predict_afew_corpus(self, a_fss, v_fss, t_fss):
        start_time = time.time()
        with torch.no_grad():
            pred_afew = self.avt_model(
                a_fss.to(self.device), v_fss.to(self.device), t_fss.to(self.device)
            )
        pred_afew = nn.functional.softmax(pred_afew, dim=1).cpu().detach().numpy()
        time_afew = time.time() - start_time
        self.dict_time['time_pred_afew'] = time_afew % 60
        return pred_afew


# Example usage:
if __name__ == "__main__":
    # Load configuration from TOML file
    # This part of the code must be customized for each corpora separately
    config = toml.load('src/config.toml')

    path_to_video = config['paths']['video_path']
    name_videos = [
        i for i in os.listdir(os.path.join(path_to_video)) if i.endswith('.avi')
    ]

    emotion_recognition = EmotionRecognition(config=config)
    path_true_data = config['paths']['true_data_path']
    df_AFEW = pd.read_csv(path_true_data)

    for idx, path in enumerate(name_videos):
        print(f'{idx + 1} / {len(name_videos)}')
        true_emo = df_AFEW[df_AFEW.name_file == path].emo.tolist()[0]
        # For MELD, EIMOCAP and MOSEI corpora, use the text prepared by the corpus authors. text='transcription by authors'
        pred_emotion_sc, pred_emotion_mc, pred_emotion_afew = emotion_recognition.predict_emotion(
            os.path.join(path_to_video, path),
            text=None
        )
        if pred_emotion_sc is None:
            continue
        print('Name video: ', path)
        print('Video duration sec: {:.2f}'.format(emotion_recognition.dict_time['time_video']))
        print('Recognition time sec: {:.2f}'.format(
            sum(list(emotion_recognition.dict_time.values())[1:])
        ))
        print('Results with MELD encoders:')
        print('True emotion:', true_emo)
        print('Two max predicted emotions of the Single-Corpus model:', pred_emotion_sc)
        print('Two max predicted emotions of the Multi-Corpus model:', pred_emotion_mc)
        print('Two max predicted emotions of the AFEW model:', pred_emotion_afew)
        print()
