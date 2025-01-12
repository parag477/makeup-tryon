import cv2
import mediapipe as mp
import itertools
import numpy as np
from scipy.interpolate import splev, splprep
from threading import Lock
import base64

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

    def apply_lipstick(self, image, landmarks, indexes, color, blur_kernel_size=(7, 7), blur_sigma=3, color_intensity=0.2):
        points = np.array([(int(landmarks[idx].x * image.shape[1]), int(landmarks[idx].y * image.shape[0])) for idx in indexes])

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [cv2.convexHull(points)], 255)

        boundary_mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)

        colored_image = np.zeros_like(image)
        colored_image[:] = color

        lipstick_image = cv2.bitwise_and(colored_image, colored_image, mask=boundary_mask)
        lips_colored = cv2.addWeighted(image, 1, lipstick_image, color_intensity, 0)

        blurred = cv2.GaussianBlur(lips_colored, blur_kernel_size, blur_sigma)

        gradient_mask = cv2.GaussianBlur(boundary_mask, (15, 15), 0)
        gradient_mask = gradient_mask / 255.0
        lips_with_gradient = (blurred * gradient_mask[..., np.newaxis] + image * (1 - gradient_mask[..., np.newaxis])).astype(np.uint8)

        final_image = np.where(boundary_mask[..., np.newaxis] == 0, image, lips_with_gradient)
        return final_image

    def apply_eyeliner(self, frame, left_eye_indices, right_eye_indices, color=(0, 0, 0), thickness=2, alpha=0.2):
        # Convert the frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Create a transparent overlay for blending
            overlay = frame.copy()

            for face_landmarks in result.multi_face_landmarks:
                # Function to draw smooth eyeliner along the eye contour
                def draw_smooth_eyeliner(eye_indices):
                    # Extract eye region coordinates
                    eye_points = np.array([
                        [
                            int(face_landmarks.landmark[idx].x * w),
                            int(face_landmarks.landmark[idx].y * h)
                        ]
                        for idx in eye_indices
                    ], np.int32)

                    # Smooth the eyeliner path using polylines
                    cv2.polylines(
                        overlay, [eye_points], isClosed=False, color=tuple(map(int, color)), thickness=thickness, lineType=cv2.LINE_AA
                    )

                    # Add a tapered effect: thinning at the inner and outer corners
                    for i in range(len(eye_points) - 1):
                        start_point = tuple(eye_points[i])
                        end_point = tuple(eye_points[i + 1])
                        current_thickness = max(1, thickness - i)  # Taper the thickness
                        cv2.line(overlay, start_point, end_point, tuple(map(int, color)), current_thickness, lineType=cv2.LINE_AA)

                # Apply the eyeliner to both eyes
                draw_smooth_eyeliner(left_eye_indices)
                draw_smooth_eyeliner(right_eye_indices)

            # Blend the overlay with the original frame for transparency
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        return frame

    def apply_eyeshadow(self, frame, left_eye_indices, right_eye_indices, color=(130, 50, 200), alpha=0.17, blur_radius=35):
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

    def apply_blush(self, frame, left_cheek_indices, right_cheek_indices, color=(130, 119, 255), alpha=0.12, blur_radius=45, sparsity=0.8):
    
        # Convert the frame to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            h, w, _ = frame.shape

            # Create a base mask to hold the blush effect
            mask = np.zeros_like(frame, dtype=np.uint8)

            for face_landmarks in result.multi_face_landmarks:
                def create_gradient_blush(cheek_indices):
                    # Get cheek coordinates and determine center
                    cheek_points = np.array([
                        [
                            int(face_landmarks.landmark[idx].x * w),
                            int(face_landmarks.landmark[idx].y * h)
                        ]
                        for idx in cheek_indices
                    ], np.int32)

                    # Calculate center of cheek
                    cheek_center = np.mean(cheek_points, axis=0).astype(int)
                    max_distance = np.max(np.linalg.norm(cheek_points - cheek_center, axis=1))

                    # Create a local cheek mask
                    cheek_mask = np.zeros((h, w), dtype=np.float32)
                    cv2.fillPoly(cheek_mask, [cheek_points], 1.0)  # Fill the cheek area

                    # Calculate distance from center for each pixel in cheek region
                    Y, X = np.ogrid[:h, :w]
                    distances = np.sqrt((X - cheek_center[0]) ** 2 + (Y - cheek_center[1]) ** 2)
                    gradient_alpha = alpha * (1 - (distances / max_distance)) * sparsity
                    gradient_alpha = np.clip(gradient_alpha, 0, alpha)  # Limit alpha range

                    # Apply gradient to color and add to main mask
                    for i in range(3):  # Apply color gradient on each channel
                        mask[:, :, i] = np.where(cheek_mask, color[i] * gradient_alpha, mask[:, :, i])

                # Apply blush to each cheek
                create_gradient_blush(left_cheek_indices)
                create_gradient_blush(right_cheek_indices)

            # Apply a slight blur to soften the edges
            blurred_mask = cv2.GaussianBlur(mask, (blur_radius, blur_radius), 0)

            # Blend the blurred mask with the original frame
            frame = cv2.addWeighted(blurred_mask, 1, frame, 1 - alpha, 0)

        return frame

    def apply_makeup(self, image_data, makeup_options):
        """Apply makeup to the image based on the provided options."""
        try:
            # Decode base64 image
            image_data = image_data.split(',')[1] if ',' in image_data else image_data
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Get face landmarks
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                return None, "No face detected"

            # Create a copy of the image for makeup application
            result_image = image.copy()

            # Apply each type of makeup if enabled
            for face_landmarks in results.multi_face_landmarks:
                if makeup_options.get('lipstick_enabled', False):
                    result_image = self.apply_lipstick(
                        result_image, 
                        face_landmarks, 
                        self.LIPS_INDEXES, 
                        makeup_options.get('lipstick_color', [0, 0, 255])
                    )
                
                if makeup_options.get('eyeliner_enabled', False):
                    EYELINER_LEFT = [359,263,466,388,387,386,385,384,398]
                    EYELINER_RIGHT = [130,33,246,161,160,159,158,157,173]
                    result_image = self.apply_eyeliner(
                        result_image, 
                        EYELINER_LEFT, 
                        EYELINER_RIGHT, 
                        makeup_options.get('eyeliner_color', [14, 14, 18])
                    )
                
                if makeup_options.get('eyeshadow_enabled', False):
                    left_eye_shadow_indices = [157,56,222,223,224,225,113,247,160,159,158]
                    right_eye_shadow_indices = [384,286,258,442,443,444,445,342,388,387,386,385]
                    result_image = self.apply_eyeshadow(
                        result_image, 
                        left_eye_shadow_indices, 
                        right_eye_shadow_indices, 
                        makeup_options.get('eyeshadow_color', [91, 123, 195])
                    )
                
                if makeup_options.get('blush_enabled', False):
                    left_cheek_indices = [449,450,348,330,266,425,411,352,345,346]
                    right_cheek_indices = [31,111,123,187,205,36,101,119,230,229,228]
                    result_image = self.apply_blush(
                        result_image, 
                        left_cheek_indices, 
                        right_cheek_indices, 
                        makeup_options.get('blush_color', [130, 119, 255])
                    )

            # Convert the result image to base64
            _, buffer = cv2.imencode('.jpg', result_image)
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            return image_base64, None

        except Exception as e:
            return None, str(e)

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
                    EYELINER_LEFT = [359,263,466,388,387,386,385,384,398]
                    EYELINER_RIGHT = [130,33,246,161,160,159,158,157,173]
                    eyeliner_color = makeup_options['eyeliner'].get('color', (14, 14, 18))
                    processed_frame = self.apply_eyeliner(processed_frame, EYELINER_LEFT, EYELINER_RIGHT, color=eyeliner_color)

                if makeup_options.get('lipstick', {}).get('enabled', False):
                    lipstick_color = makeup_options['lipstick'].get('color', (0, 0, 255))
                    processed_frame = self.apply_lipstick(processed_frame, face_landmarks.landmark, self.LIPS_INDEXES, lipstick_color)

                if makeup_options.get('blush', {}).get('enabled', False):
                    left_cheek_indices = [449,450,348,330,266,425,411,352,345,346]
                    right_cheek_indices = [31,111,123,187,205,36,101,119,230,229,228]
                    blush_color = makeup_options['blush'].get('color', (130, 119, 255))
                    processed_frame = self.apply_blush(processed_frame, left_cheek_indices, right_cheek_indices, color=blush_color)

                if makeup_options.get('eyeshadow', {}).get('enabled', False):
                    left_eye_shadow_indices = [157,56,222,223,224,225,113,247,160,159,158]
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
