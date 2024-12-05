import cv2
import mediapipe as mp
import itertools
import numpy as np
from scipy.interpolate import splev, splprep
from threading import Lock

class MakeupApplication:
    def __init__(self):
        self._face_mesh = None
        self._face_mesh_lock = Lock()
        self.LEFT_EYE_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_LEFT_EYE)))
        self.RIGHT_EYE_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)))
        self.LIPS_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_LIPS)))
        self.LEFT_EYEBROW_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)))
        self.RIGHT_EYEBROW_INDEXES = list(set(itertools.chain(*mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)))
        self.EYELINER_INDEXES = [33, 133, 157, 158, 159, 160, 161, 246, 362, 398, 384, 385, 386, 387, 388, 466]

    @property
    def face_mesh(self):
        if self._face_mesh is None:
            with self._face_mesh_lock:
                if self._face_mesh is None:  # Double-check pattern
                    mp_face_mesh = mp.solutions.face_mesh
                    self._face_mesh = mp_face_mesh.FaceMesh(
                        static_image_mode=True,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5
                    )
        return self._face_mesh

    def get_upper_side_coordinates(self, eye_landmarks):
        sorted_landmarks = sorted(eye_landmarks, key=lambda coord: coord.y)
        half_length = len(sorted_landmarks) // 2
        return sorted_landmarks[:half_length]

    def get_lower_side_coordinates(self, eye_landmarks):
        sorted_landmarks = sorted(eye_landmarks, key=lambda coord: coord.y)
        half_length = len(sorted_landmarks) // 2
        return sorted_landmarks[half_length:]

    def apply_lipstick(self, frame, landmarks, lips_indices, color=(0, 0, 255), alpha=0.4, blur_radius=5):
        try:
            mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            points = []
            for idx in lips_indices:
                landmark = landmarks[idx]
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                points.append([x, y])
            
            points = np.array(points, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
            
            # Apply Gaussian blur to create smooth edges
            mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
            
            # Create colored overlay
            colored_mask = frame.copy()
            colored_mask[mask > 0] = color
            
            # Blend the original frame with the colored mask
            mask = mask.astype(float) / 255
            mask = np.stack([mask, mask, mask], axis=-1)
            frame = frame.astype(float) * (1 - mask * alpha) + colored_mask.astype(float) * (mask * alpha)
            
            return frame.astype(np.uint8)
        except Exception as e:
            print(f"Error in apply_lipstick: {str(e)}")
            return frame

    def draw_eyeliner(self, image, upper_eye_coordinates, color=(14, 14, 18), thickness=2, fade_factor=0.6):
        result_image = image.copy()
        
        eyeliner_points = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in upper_eye_coordinates]
        eyeliner_points.sort(key=lambda x: x[0])
        eyeliner_points = np.array(eyeliner_points, dtype=np.int32)

        if len(eyeliner_points) >= 4:
            spline_curve = cv2.polylines(np.zeros_like(image), [eyeliner_points], isClosed=False, color=(255, 255, 255), thickness=thickness)

        for i in range(len(eyeliner_points) - 1):
            start_point = tuple(eyeliner_points[i])
            end_point = tuple(eyeliner_points[i + 1])

            relative_pos = i / (len(eyeliner_points) - 1)
            dynamic_thickness = int(thickness * (1 - fade_factor * abs(relative_pos - 0.5)))

            cv2.line(result_image, start_point, end_point, color, dynamic_thickness)

        blurred_result = cv2.GaussianBlur(result_image, (3, 3), 1)
        final_result = cv2.addWeighted(result_image, 0.9, blurred_result, 0.1, 0)

        return final_result

    def apply_eyeshadow(self, frame, left_eye_indices, right_eye_indices, color=(130, 50, 200), alpha=0.25, blur_radius=25):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame) 

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            mask = np.zeros_like(frame, dtype=np.uint8)

            for face_landmarks in result.multi_face_landmarks:
                def create_eyeshadow(eye_indices):
                    eye_points = np.array([
                        [int(face_landmarks.landmark[idx].x * w),
                         int(face_landmarks.landmark[idx].y * h)]
                        for idx in eye_indices
                    ], np.int32)

                    eye_mask = np.zeros_like(frame, dtype=np.uint8)
                    cv2.fillPoly(eye_mask, [eye_points], color)

                    mask[:, :, 0] = np.where(eye_mask[:, :, 0] > 0, color[0], mask[:, :, 0])
                    mask[:, :, 1] = np.where(eye_mask[:, :, 1] > 0, color[1], mask[:, :, 1])
                    mask[:, :, 2] = np.where(eye_mask[:, :, 2] > 0, color[2], mask[:, :, 2])

                create_eyeshadow(left_eye_indices)
                create_eyeshadow(right_eye_indices)

            blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
            frame = cv2.addWeighted(blurred_mask, alpha, frame, 1 - alpha, 0)

        return frame

    def apply_blush(self, frame, left_cheek_indices, right_cheek_indices, color=(130, 119, 255), alpha=0.20, blur_radius=45):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape
            mask = np.zeros_like(frame, dtype=np.uint8)

            for face_landmarks in result.multi_face_landmarks:
                def create_gradient_blush(cheek_indices):
                    cheek_points = np.array([
                        [int(face_landmarks.landmark[idx].x * w),
                         int(face_landmarks.landmark[idx].y * h)]
                        for idx in cheek_indices
                    ], np.int32)
                    
                    cheek_center = np.mean(cheek_points, axis=0).astype(int)
                    max_distance = np.max(np.linalg.norm(cheek_points - cheek_center, axis=1))

                    cheek_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.fillPoly(cheek_mask, [cheek_points], 1.0)

                    Y, X = np.ogrid[:h, :w]
                    distances = np.sqrt((X - cheek_center[0]) ** 2 + (Y - cheek_center[1]) ** 2)
                    gradient_alpha = alpha * (1 - (distances / max_distance))
                    gradient_alpha = np.clip(gradient_alpha, 0, alpha)

                    for i in range(3):
                        mask[:, :, i] = np.where(cheek_mask, color[i] * gradient_alpha, mask[:, :, i])

                create_gradient_blush(left_cheek_indices)
                create_gradient_blush(right_cheek_indices)

            blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)
            frame = cv2.addWeighted(blurred_mask, 1, frame, 1 - alpha, 0)

        return frame

    def process_frame(self, frame, makeup_options=None):
        if makeup_options is None:
            makeup_options = {
                'lipstick': {'enabled': True, 'color': (0, 0, 255)},
                'eyeliner': {'enabled': True, 'color': (14, 14, 18)},
                'eyeshadow': {'enabled': True, 'color': (91, 123, 195)},
                'blush': {'enabled': True, 'color': (130, 119, 255)}
            }

        try:
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame
            result = self.face_mesh.process(rgb_frame)

            # If no face is detected, return the original frame
            if not result.multi_face_landmarks:
                return frame

            # Process detected face
            for face_landmarks in result.multi_face_landmarks:
                # Create a copy of the frame for modifications
                processed_frame = frame.copy()

                if makeup_options.get('eyeliner', {}).get('enabled', False):
                    left_eye_landmarks = [face_landmarks.landmark[idx] for idx in self.LEFT_EYE_INDEXES]
                    right_eye_landmarks = [face_landmarks.landmark[idx] for idx in self.RIGHT_EYE_INDEXES]
                    
                    upper_left_eye = self.get_upper_side_coordinates(left_eye_landmarks)
                    upper_right_eye = self.get_upper_side_coordinates(right_eye_landmarks)
                    lower_left_eye = self.get_lower_side_coordinates(left_eye_landmarks)
                    lower_right_eye = self.get_lower_side_coordinates(right_eye_landmarks)

                    eyeliner_color = makeup_options['eyeliner'].get('color', (14, 14, 18))
                    processed_frame = self.draw_eyeliner(processed_frame, upper_left_eye, color=eyeliner_color)
                    processed_frame = self.draw_eyeliner(processed_frame, upper_right_eye, color=eyeliner_color)
                    processed_frame = self.draw_eyeliner(processed_frame, lower_left_eye, color=eyeliner_color)
                    processed_frame = self.draw_eyeliner(processed_frame, lower_right_eye, color=eyeliner_color)

                if makeup_options.get('lipstick', {}).get('enabled', False):
                    lipstick_color = makeup_options['lipstick'].get('color', (0, 0, 255))
                    processed_frame = self.apply_lipstick(processed_frame, face_landmarks.landmark, self.LIPS_INDEXES, lipstick_color)

                if makeup_options.get('blush', {}).get('enabled', False):
                    left_cheek_indices = [449,450,348,330,266,425,411,352,345,346]
                    right_cheek_indices = [31,111,123,187,205,36,101,119,230,229,228]
                    blush_color = makeup_options['blush'].get('color', (130, 119, 255))
                    processed_frame = self.apply_blush(processed_frame, left_cheek_indices, right_cheek_indices, color=blush_color)

                if makeup_options.get('eyeshadow', {}).get('enabled', False):
                    left_eye_shadow_indices = [157,56,222,223,224,225,113,247,30,29,27,28]
                    right_eye_shadow_indices = [384,286,258,442,443,444,445,342,388,387,386,385]
                    eyeshadow_color = makeup_options['eyeshadow'].get('color', (91, 123, 195))
                    processed_frame = self.apply_eyeshadow(processed_frame, left_eye_shadow_indices, right_eye_shadow_indices, color=eyeshadow_color)

                return processed_frame

        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return frame

        return frame

    def process_image(self, image_path, makeup_options=None):
        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Could not read image from path")
        return self.process_frame(frame, makeup_options)
