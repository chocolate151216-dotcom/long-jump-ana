import os
import math
import cv2
import numpy as np
from ultralytics import YOLO
import torch

class PoseEstimator:
    def __init__(self, show_keypoint_labels=False, show_ground_lines=False, show_analysis=True):
        self.keypoint_names = [
            'nose', 'l_eye', 'r_eye', 'l_ear', 'r_ear',
            'l_sho', 'r_sho', 'l_elb', 'r_elb',
            'l_wri', 'r_wri', 'l_hip', 'r_hip', 'l_kne', 'r_kne', 'l_ank', 'r_ank'
        ]
        self.skeleton = [
            (0,1),(0,2),(1,3),(2,4),
            (5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),
            (11,13),(13,15),(12,14),(14,16)
        ]
        self.keypoint_colors = [(0,0,255)]*5 + [(0,255,0)]*2 + [(255,0,0)]*4 + [(255,0,255)]*6
        self.skeleton_colors = [(0,0,255)]*4 + [(0,255,0)]*5 + [(255,0,255)]*7
        # 動的スケルトン（モデルから取得できる場合に使用）
        self.dynamic_skeleton = None

        self.ground_angle_deg = 0.0  # 右向き水平を0度とする
        self.model = None
        self.device = 0
        self.conf = 0.25
        # ログ出力用コールバック（任意）
        self._log_cb = None

        self.show_keypoint_labels = show_keypoint_labels
        self.show_ground_lines = show_ground_lines
        self.show_analysis = show_analysis

    def _idx(self, short_name: str) -> int:
        return self.keypoint_names.index(short_name)

    def set_logger(self, log_callback):
        """診断ログの出力先を設定"""
        self._log_cb = log_callback

    def _log(self, msg):
        try:
            if self._log_cb:
                self._log_cb(str(msg))
        except Exception:
            pass

    def load_model(self, model_path, device=0, conf=0.25):
        """再学習モデルをロード"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
        self.model = YOLO(model_path)  # 再学習モデルをロード
        self.device = device
        self.conf = conf

    def _prepare_source(self, image):
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"failed to read image: {image}")
        else:
            img = image
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        return img

    def predict_keypoints(self, image, imgsz=640):
        if self.model is None:
            return None
        source = self._prepare_source(image)
        results = None
        try:
            # 1回目: predict API
            results = self.model.predict(source=source, device=self.device, conf=self.conf, imgsz=imgsz, verbose=False)
            if not results or getattr(results[0], "keypoints", None) is None:
                # 2回目: __call__ 経路
                results = self.model(source, device=self.device, conf=self.conf, imgsz=imgsz, verbose=False)
            # 3回目: 明示的なタスク上書きで再試行
            if (not results or getattr(results[0], "keypoints", None) is None) and hasattr(self.model, "overrides"):
                try:
                    self.model.overrides["task"] = "pose"
                    results = self.model.predict(source=source, device=self.device, conf=self.conf, imgsz=imgsz, verbose=False)
                except Exception:
                    pass
        except Exception as e:
            # デバイスエラー時は CPU で一度だけ再試行
            if 'CUDA' in str(e) or 'device' in str(e):
                self._log(f"PoseEstimator: 推論デバイスエラーによりCPU再試行: {e}")
                try:
                    results = self.model.predict(source=source, device='cpu', conf=self.conf, imgsz=imgsz, verbose=False)
                    self.device = 'cpu'
                except Exception as e2:
                    self._log(f"PoseEstimator: CPU再試行も失敗: {e2}")
                    raise
            else:
                raise

        if not results:
            self._log("PoseEstimator: 推論結果が空です")
            return None

        res = results[0]
        if getattr(res, "keypoints", None) is None:
            # 最終的に keypoints が無い場合
            self._log("PoseEstimator: このフレームで keypoints が返りませんでした（モデルが pose でない可能性）")
            return None

        kps = None
        try:
            # 動的スケルトンの取得（あれば使用）
            try:
                skel = getattr(res.keypoints, "skeleton", None)
                if skel:
                    self.dynamic_skeleton = [(int(a), int(b)) for a, b in skel]
            except Exception:
                self.dynamic_skeleton = None

            xy = getattr(res.keypoints, "xy", None)   # (n,K,2)
            conf = getattr(res.keypoints, "conf", None)  # (n,K)
            if xy is not None and conf is not None:
                xy = xy.cpu().numpy()
                conf = conf.cpu().numpy()
                kps = np.concatenate([xy, conf[..., None]], axis=-1)  # (n,K,3)
            else:
                data = getattr(res.keypoints, "data", None)  # (n,K,3)
                if data is not None:
                    kps = data.cpu().numpy()
            if kps is None:
                self._log("PoseEstimator: keypoints 配列の組み立てに失敗しました")
        except Exception as e:
            self._log(f"PoseEstimator: keypoints 取得時に例外: {e}")
            kps = None
        return kps

    def draw_pose(self, image, keypoints, confidence_threshold=0.5, color=None, radius=4):
        out = image.copy()
        # 可視性配列の安全化（任意長対応）
        vis = (keypoints[:, 2] >= confidence_threshold) if keypoints.shape[1] >= 3 else np.ones((keypoints.shape[0],), dtype=bool)

        # キーポイント描画（任意長対応）
        for i in range(keypoints.shape[0]):
            if not vis[i]:
                continue
            x, y = float(keypoints[i, 0]), float(keypoints[i, 1])
            col = color if color is not None else (0, 255, 0)
            cv2.circle(out, (int(x), int(y)), radius, col, -1)
            if self.show_keypoint_labels and i < len(self.keypoint_names):
                cv2.putText(out, self.keypoint_names[i], (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)

        # スケルトン選択（モデル提供を優先、無ければ既定）
        skel = self.dynamic_skeleton if self.dynamic_skeleton else self.skeleton
        # スケルトン描画（インデックス範囲ガード）
        for (s, e) in skel:
            if s < 0 or e < 0 or s >= keypoints.shape[0] or e >= keypoints.shape[0]:
                continue
            if vis[s] and vis[e]:
                scol = (0, 255, 255) if color is None else color
                p1 = (int(keypoints[s, 0]), int(keypoints[s, 1]))
                p2 = (int(keypoints[e, 0]), int(keypoints[e, 1]))
                cv2.line(out, p1, p2, scol, 2)

        if self.show_ground_lines:
            h, w = out.shape[:2]
            ang = math.radians(self.ground_angle_deg)
            cx, cy = w//2, h//2
            dx = int(math.cos(ang) * w)
            dy = int(math.sin(ang) * w)
            cv2.line(out, (cx - dx, cy - dy), (cx + dx, cy + dy), (255, 0, 0), 1)
        return out

    def _angle_between_vectors(self, v1, v2) -> float:
        a = np.array(v1, dtype=np.float32)
        b = np.array(v2, dtype=np.float32)
        na = np.linalg.norm(a) + 1e-6
        nb = np.linalg.norm(b) + 1e-6
        cosang = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        return math.degrees(math.acos(cosang))

    def calculate_knee_angle(self, keypoints, side='left', confidence_threshold=0.5):
        hip_i  = self._idx('l_hip' if side=='left' else 'r_hip')
        knee_i = self._idx('l_kne' if side=='left' else 'r_kne')
        ank_i  = self._idx('l_ank' if side=='left' else 'r_ank')
        pts = keypoints[[hip_i, knee_i, ank_i], :]
        if np.any(pts[:,2] < confidence_threshold):
            return None
        hip = pts[0,:2]; knee = pts[1,:2]; ank = pts[2,:2]
        v1 = hip - knee
        v2 = ank - knee
        return self._angle_between_vectors(v1, v2)

    def segment_angle_with_ground(self, keypoints, segment='shin', side='left', confidence_threshold=0.5):
        ground_rad = math.radians(self.ground_angle_deg)
        ground_vec = np.array([math.cos(ground_rad), math.sin(ground_rad)], dtype=np.float32)
        if segment == 'shin':
            p1_i = self._idx('l_kne' if side=='left' else 'r_kne')
            p2_i = self._idx('l_ank' if side=='left' else 'r_ank')
        else:
            p1_i = self._idx('l_hip' if side=='left' else 'r_hip')
            p2_i = self._idx('l_kne' if side=='left' else 'r_kne')
        pts = keypoints[[p1_i, p2_i], :]
        if np.any(pts[:,2] < confidence_threshold):
            return None
        v_seg = pts[1,:2] - pts[0,:2]
        if np.linalg.norm(v_seg) < 1e-3:
            return None
        return self._angle_between_vectors(v_seg, ground_vec)

    def infer_takeoff_side(self, keypoints, confidence_threshold=0.2):
        la = self._idx('l_ank'); ra = self._idx('r_ank')
        if keypoints[la,2] < confidence_threshold and keypoints[ra,2] < confidence_threshold:
            return 'left'
        if keypoints[la,2] >= confidence_threshold and keypoints[ra,2] >= confidence_threshold:
            return 'left' if keypoints[la,1] > keypoints[ra,1] else 'right'
        return 'left' if keypoints[la,2] >= confidence_threshold else 'right'

    def lead_knee_lift_is_above_hip(self, keypoints, side='left', confidence_threshold=0.5):
        knee_i = self._idx('l_kne' if side=='left' else 'r_kne')
        hip_i  = self._idx('l_hip' if side=='left' else 'r_hip')
        pts = keypoints[[knee_i, hip_i], :]
        if np.any(pts[:,2] < confidence_threshold):
            return None
        knee_y = float(pts[0,1]); hip_y = float(pts[1,1])
        return knee_y < hip_y

    def get_pose_info(self, keypoints, confidence_threshold=0.5):
        return {'jump_analysis': {}}

    def set_ground_angle(self, angle_deg):
        self.ground_angle_deg = float(angle_deg)

    def estimate_ground_angle(self, p1, p2):
        x1, y1 = p1; x2, y2 = p2
        ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
        self.ground_angle_deg = float(ang)
        return self.ground_angle_deg