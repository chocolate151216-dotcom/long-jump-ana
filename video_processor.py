import os
import cv2
import numpy as np
import collections
import torch
import subprocess  
import tempfile    
import shutil
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from pose_estimator import PoseEstimator

class VideoProcessor:
    def __init__(self, config, jump_analyzer=None):
        self.config = config
        self.jump_analyzer = jump_analyzer
        self.pose_estimator = getattr(self, "pose_estimator", PoseEstimator())
        self._jp_font_cache = {}  # 日本語フォントキャッシュ
        self.detection_model = None  # 追加
        self.tracker = None          # 追加
        self._device = 'cpu'         # 追加

    def _select_device(self, prefer=None):
        try:
            import torch
            if prefer and str(prefer).lower() == 'cpu':
                return 'cpu'
            if torch.cuda.is_available():
                return '0'  # CUDA:0 を使用
        except Exception:
            pass
        return 'cpu'

    def _initialize_models(self):
        """モデルとトラッカーを初期化"""
        # PoseEstimator を初期化
        self.pose_estimator = PoseEstimator()
        if hasattr(self.config, "pose_model_path"):
            self.pose_estimator.load_model(
                self.config.pose_model_path,
                device=getattr(self.config, "device", 0),
                conf=getattr(self.config, "pose_conf", 0.25)
            )

        # YOLO検出モデルとDeepSort初期化
        if self.detection_model is None:
            det_path = getattr(self.config, "det_model_path", "yolov8n.pt")
            self._device = self._select_device(getattr(self.config, "device", None))
            try:
                self.detection_model = YOLO(det_path)
            except Exception as e:
                raise RuntimeError(f"検出モデル読み込み失敗: {det_path} ({e})")
        if self.tracker is None:
            self.tracker = DeepSort(max_age=30, n_init=3, max_iou_distance=0.7)

    # 検出・追跡（簡易スタブ：既存の処理が別にある場合はそこで self.detection_model を使う）
    def detect_and_track(self, video_path, progress_callback=None, log_callback=None):
        self._initialize_models()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 一時ディレクトリを作成
        self.temp_dir = tempfile.mkdtemp(prefix="long_jump_tracking_")
        
        detection_results = {}
        frame_idx = 0
        
        # 走り幅跳び用の関心領域を設定
        roi_y_start = max(0, self.config.takeoff_line_y - 200)
        roi_y_end = min(height, self.config.takeoff_line_y + 300)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # プログレス更新
                if progress_callback:
                    progress = (frame_idx / total_frames) * 100
                    progress_callback(progress, f"フレーム処理中 {frame_idx + 1}/{total_frames}")
                    
                # 関心領域での人物検出
                roi_frame = frame[roi_y_start:roi_y_end, :]
                results = self.detection_model(roi_frame, classes=[0], verbose=False)  # クラス0は'person'
                
                if results and len(results[0].boxes) > 0:
                    # 検出結果を抽出
                    detections = []
                    boxes = results[0].boxes
                    
                    for box in boxes:
                        # バウンディングボックス座標を取得
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        
                        if conf > self.config.detection_threshold:
                            x1, y1, x2, y2 = xyxy
                            # ROI座標を元のフレーム座標に変換
                            y1 += roi_y_start
                            y2 += roi_y_start
                            
                            w, h = x2 - x1, y2 - y1
                            
                            # 走り幅跳び選手らしいサイズフィルタリング
                            if w > 30 and h > 60 and h/w > 1.2:  # 縦長の人物
                                detections.append([[x1, y1, w, h], conf])
                                
                    # トラッカーを更新
                    if detections:
                        tracks = self.tracker.update_tracks(detections, frame=frame)
                        
                        # 追跡結果を保存
                        frame_filename = os.path.join(self.temp_dir, f"frame_{frame_idx:06d}.jpg")
                        cv2.imwrite(frame_filename, frame)
                        
                        for track in tracks:
                            if track.is_confirmed():
                                track_id = track.track_id
                                bbox = track.to_ltrb()  # left, top, right, bottom
                                
                                if track_id not in detection_results:
                                    detection_results[track_id] = []
                                    
                                detection_results[track_id].append({
                                    'frame_idx': frame_idx,
                                    'frame_path': frame_filename,
                                    'bbox': bbox,
                                    'timestamp': frame_idx / fps,
                                    'center_x': (bbox[0] + bbox[2]) / 2,
                                    'center_y': (bbox[1] + bbox[3]) / 2
                                })
                                
                frame_idx += 1
                
                # 定期的にログ出力
                if log_callback and frame_idx % 30 == 0:
                    log_callback(f"処理済み {frame_idx}/{total_frames} フレーム, 検出選手数: {len(detection_results)}")
                    
        finally:
            cap.release()
            
        # 十分な検出数を持つトラックのみをフィルタリング
        min_detections = max(5, total_frames // 50)  # 最低2%のフレーム
        filtered_results = {
            track_id: frames for track_id, frames in detection_results.items()
            if len(frames) >= min_detections
        }
        
        # 走り幅跳びの軌跡分析による追加フィルタリング
        final_results = self._filter_jump_trajectories(filtered_results)
        
        if log_callback:
            log_callback(f"検出完了。{len(final_results)}人の選手を特定しました")
            
        return final_results
        
    def _filter_jump_trajectories(self, detection_results):
        """走り幅跳びの軌跡に基づいてフィルタリング"""
        filtered_results = {}
        
        for track_id, frames in detection_results.items():
            if len(frames) < 10:
                continue
                
            # X座標の変化を分析
            x_positions = [f['center_x'] for f in frames]
            x_movement = max(x_positions) - min(x_positions)
            
            # 十分な水平移動があるかチェック（走り幅跳びの特徴）
            if x_movement > 200:  # ピクセル単位での最小移動距離
                # Y座標の変化パターンをチェック
                y_positions = [f['center_y'] for f in frames]
                y_variation = np.std(y_positions)
                
                # 適度なY座標の変化があるかチェック（跳躍の特徴）
                if y_variation > 20:
                    filtered_results[track_id] = frames
                    
        return filtered_results
        
    # 日本語フォント取得（Meiryo/游ゴシック/MSゴシック等を探索）
    def _get_jp_font(self, size=28):
        fs = int(size)
        if fs in self._jp_font_cache:
            return self._jp_font_cache[fs]
        candidates = []
        cfg_path = getattr(self.config, "jp_font_path", None)
        if cfg_path:
            candidates.append(cfg_path)
        candidates += [
            r"C:\Windows\Fonts\meiryo.ttc",
            r"C:\Windows\Fonts\Meiryo.ttc",
            r"C:\Windows\Fonts\msgothic.ttc",
            r"C:\Windows\Fonts\YuGothM.ttc",
            r"C:\Windows\Fonts\YuGothB.ttc",
        ]
        font = None
        for fp in candidates:
            try:
                if os.path.exists(fp):
                    font = ImageFont.truetype(fp, fs)
                    break
            except Exception:
                continue
        if font is None:
            font = ImageFont.load_default()  # ASCIIのみ
        self._jp_font_cache[fs] = font
        return font

    # 日本語対応テキスト描画（複数行）
    def _draw_texts(self, frame, texts, size=28, stroke_width=2, stroke=(0,0,0)):
        """
        texts: list of (text, (x,y), (b,g,r) or None)
        """
        if not texts:
            return frame
        font = self._get_jp_font(size)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for txt, pos, color in texts:
            col = tuple(color) if color is not None else (255, 255, 255)
            # PillowはRGB、OpenCVはBGRなのでカラー順を直す
            col_rgb = (col[2], col[1], col[0])
            try:
                draw.text(pos, txt, font=font, fill=col, stroke_width=stroke_width, stroke_fill=stroke)
            except Exception:
                # フォールバック（不可視文字除去）
                safe = txt.encode("utf-8", "ignore").decode("utf-8", "ignore")
                draw.text(pos, safe, font=font, fill=col, stroke_width=stroke_width, stroke_fill=stroke)
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    def _find_ffmpeg(self):
        cfg = getattr(self.config, "ffmpeg_path", None)
        if cfg and os.path.isfile(cfg):
            return cfg
        env_ffmpeg = os.environ.get("FFMPEG_PATH")
        if env_ffmpeg and os.path.isfile(env_ffmpeg):
            return env_ffmpeg
        for name in ("ffmpeg.exe", "ffmpeg"):
            path = shutil.which(name)
            if path:
                return path
        candidates = [
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
            r"C:\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def _start_ffmpeg_writer(self, output_path, fps, width, height, crf=18, preset='slow'):
        ffmpeg = self._find_ffmpeg()
        self._ffmpeg_proc = None
        self._video_writer = None
        if ffmpeg:
            cmd = [
                ffmpeg, "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", f"{fps}",
                "-i", "-",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", preset,
                "-crf", str(crf),
                output_path
            ]
            self._ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def _write_frame_to_ffmpeg(self, frame):
        if self._ffmpeg_proc:
            self._ffmpeg_proc.stdin.write(frame.tobytes())
        elif self._video_writer:
            self._video_writer.write(frame)

    def _close_ffmpeg_writer(self):
        if self._ffmpeg_proc:
            try:
                self._ffmpeg_proc.stdin.close()
                self._ffmpeg_proc.wait()
            except Exception:
                pass
            self._ffmpeg_proc = None
        if self._video_writer:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None

    def generate_pose_video_impl(
            
        self,
        video_path,
        output_path,
        athlete_id=None,
        athlete_frames=None,
        progress_callback=None,
        log_callback=None,
        selected_indices=None,
        takeoff_start_frame=None,
        takeoff_end_frame=None,
        pose_model_path=None,
        device=None,
        conf=None,
        imgsz=640
    ):
        # ログ関数
        def _log(msg):
            if log_callback:
                try:
                    log_callback(str(msg))
                except Exception:
                    pass

        # モデル準備
        pose_model_path = pose_model_path or getattr(self.config, "pose_model_path", None) or r"C:\Users\choco\LongJ\yolov8x-pose.pt"
        if not pose_model_path or not os.path.exists(pose_model_path):
            _log(f"ポーズモデルが見つかりません: {pose_model_path}")
            return False

        dev = self._select_device(getattr(self.config, "device", device))
        self.pose_estimator.set_logger(_log)
        try:
            self.pose_estimator.load_model(
                pose_model_path,
                device=dev,
                conf=(conf if conf is not None else 0.18)
            )
        except Exception as e:
            _log(f"ポーズモデルの読み込みに失敗: {e}")
            return False

        # 動画準備
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            _log("動画のオープンに失敗しました")
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 範囲決定（未指定なら全体）
        start = 0 if takeoff_start_frame is None else max(0, int(takeoff_start_frame))
        end = (total_frames - 1) if takeoff_end_frame is None else min(total_frames - 1, int(takeoff_end_frame))
        if end < start:
            end = start

        # 選手フレームのマップ（bboxがあればクロップして推論）
        frame_map = {}
        if athlete_frames is not None and len(athlete_frames) > 0:
            for fi in athlete_frames:
                # フレーム番号（None 以外の最初のキーを採用）
                idx = None
                for k in ('frame_index', 'frame_idx', 'frame', 'index'):
                    v = fi.get(k, None)
                    if v is not None:
                        idx = v
                        break
                # bbox（None 以外の最初のキーを採用）※ np.ndarray を or で評価しない
                bbox = None
                for k in ('bbox', 'xyxy', 'tlbr'):
                    vb = fi.get(k, None)
                    if vb is not None:
                        bbox = vb
                        break
                if idx is not None:
                    frame_map[int(idx)] = bbox

        # ライター開始
        try:
            self._start_ffmpeg_writer(output_path, fps, width, height, crf=18, preset='medium')
        except Exception as e:
            _log(f"動画書き出し開始に失敗（cv2へフォールバック）: {e}")
            self._ffmpeg_proc = None
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        total_to_process = (end - start + 1)
        processed = 0

        try:
            for fidx in range(start, end + 1):
                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    # 読み取れない場合でもそのまま書き出し
                    if ok and frame is not None:
                        self._write_frame_to_ffmpeg(frame)
                    processed += 1
                    if progress_callback:
                        progress_callback(int(100 * processed / max(1, total_to_process)), f"処理中 {processed}/{total_to_process}")
                    continue

                # クロップ領域（あれば使用）
                crop = frame
                offset = (0, 0)
                bbox = frame_map.get(fidx)
                if (bbox is not None) and isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
                     x1, y1, x2, y2 = map(int, bbox[:4])  # xyxy/tlbrを想定
                     x1 = max(0, min(width - 1, x1))
                     y1 = max(0, min(height - 1, y1))
                     x2 = max(0, min(width - 1, x2))
                     y2 = max(0, min(height - 1, y2))
                     if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2].copy()
                        offset = (x1, y1)

                # ポーズ推論
                kps_batch = self.pose_estimator.predict_keypoints(crop, imgsz=imgsz)
                if kps_batch is None or (isinstance(kps_batch, np.ndarray) and kps_batch.size == 0):
                    # 推論失敗時は元フレームを書き出し
                    self._write_frame_to_ffmpeg(frame)
                    processed += 1
                    if progress_callback:
                        progress_callback(int(100 * processed / max(1, total_to_process)), f"処理中 {processed}/{total_to_process}")
                    continue

                # 複数人時は平均confが最大の人物を選択
                try:
                    conf_mean = np.nanmean(kps_batch[:, :, 2], axis=1)
                    sel = int(np.argmax(conf_mean))
                except Exception:
                    sel = 0
                kps = kps_batch[sel]  # (K,3)

                # キーポイント選択フィルタ（残像用）
                if selected_indices is not None and len(selected_indices) > 0:
                    try:
                        kps = kps[np.array(selected_indices, dtype=int)]
                    except Exception:
                        pass

                # オフセット復元（クロップした場合）
                if offset != (0, 0):
                    kps[:, 0] = kps[:, 0] + offset[0]
                    kps[:, 1] = kps[:, 1] + offset[1]

                # 描画
                drawn = self.pose_estimator.draw_pose(frame, kps, confidence_threshold=0.5, color=None, radius=4)

                # 改善案の計算とオーバーレイ（jump_analyzerがあれば）
                if self.jump_analyzer:
                    # フェーズ判定（簡易）
                    phase = 'flight'  # デフォルト
                    if takeoff_start_frame is not None and takeoff_end_frame is not None:
                        if fidx < takeoff_start_frame:
                            phase = 'approach'
                        elif takeoff_start_frame <= fidx <= takeoff_end_frame:
                            phase = 'takeoff'
                        else:
                            phase = 'flight'  # 着地は区別しにくいのでflight扱い
                    suggestions = self.jump_analyzer._generate_frame_suggestions(kps, fidx, takeoff_start_frame or 0, takeoff_end_frame or total_frames)
                    if suggestions:
                        # テキストオーバーレイ
                        texts = [(s, (10, 30 + i*30), (255, 255, 255)) for i, s in enumerate(suggestions)]
                        drawn = self._draw_texts(drawn, texts, size=24, stroke_width=2, stroke=(0,0,0))

                # 書き出し
                self._write_frame_to_ffmpeg(drawn)

                processed += 1
                if progress_callback and processed % 5 == 0:
                    progress_callback(int(100 * processed / max(1, total_to_process)), f"処理中 {processed}/{total_to_process}")
        finally:
            cap.release()
            self._close_ffmpeg_writer()

        _log(f"ポーズ動画生成完了: {output_path}")
        if progress_callback:
            progress_callback(100, "完了")
        return True
    
    def generate_pose_video(self, *args, **kwargs):
        """
        互換ラッパー: すべての引数を本実装 generate_pose_video_impl にフォワードします。
        """
        return self.generate_pose_video_impl(*args, **kwargs)

