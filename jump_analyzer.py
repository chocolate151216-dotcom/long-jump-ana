import cv2
import numpy as np
import math
from datetime import datetime
from scipy import signal
from typing import Dict, List, Tuple, Any

class JumpAnalyzer:
    """走り幅跳び専用の分析クラス"""
    
    def __init__(self, config):
        self.config = config
        
        # 跳躍フェーズの定義
        self.phases = {
            'approach': 'アプローチ（助走）',
            'takeoff': '踏切',
            'flight': '飛行',
            'landing': '着地'
        }
        
        # キーポイントのインデックス（COCO形式）
        self.keypoint_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 3, 'right_ear': 4,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }
        
    def analyze_jump(self, video_path, athlete_id, athlete_frames, 
                    progress_callback=None, log_callback=None):
        """走り幅跳びの包括的分析を実行"""
        
        if log_callback:
            log_callback("跳躍分析を開始しています...")
            
        try:
            # 動画を開く
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"動画ファイルを開けません: {video_path}")
                
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # フレームデータを時系列順にソート
            sorted_frames = sorted(athlete_frames, key=lambda x: x['frame_idx'])
            
            # 各フレームでポーズ推定を実行
            pose_data = self._extract_pose_data(cap, sorted_frames, progress_callback, log_callback)
            
            # 跳躍フェーズを検出
            phases = self._detect_jump_phases(pose_data, fps, log_callback)
            
            # 各種分析を実行
            distance_analysis = self._analyze_jump_distance(pose_data, phases)
            speed_analysis = self._analyze_speed(pose_data, phases, fps)
            technique_analysis = self._analyze_technique(pose_data, phases)
            pose_analysis = self._analyze_pose_quality(pose_data, phases)
            
            # フレームごとの改善提案を収集
            detailed_suggestions = self._collect_frame_suggestions(pose_data, phases)
            
            # 結果をまとめる
            analysis_results = {
                'athlete_id': athlete_id,
                'analysis_time': datetime.now().isoformat(),
                'video_fps': fps,
                'total_frames': len(pose_data),
                
                # 跳躍距離
                'jump_distance': distance_analysis.get('distance', 0),
                'takeoff_position': distance_analysis.get('takeoff_pos', 0),
                'landing_position': distance_analysis.get('landing_pos', 0),
                
                # フェーズ分析
                'approach_duration': phases.get('approach_duration', 0),
                'takeoff_duration': phases.get('takeoff_duration', 0),
                'flight_duration': phases.get('flight_duration', 0),
                'landing_duration': phases.get('landing_duration', 0),
                
                # 速度分析
                'max_approach_speed': speed_analysis.get('max_approach_speed', 0),
                'takeoff_speed': speed_analysis.get('takeoff_speed', 0),
                'takeoff_angle': speed_analysis.get('takeoff_angle', 0),
                
                # 技術分析
                'takeoff_foot': technique_analysis.get('takeoff_foot', 'N/A'),
                'max_height': technique_analysis.get('max_height', 0),
                'landing_angle': technique_analysis.get('landing_angle', 0),
                
                # ポーズ分析
                'takeoff_pose_score': pose_analysis.get('takeoff_score', 0),
                'flight_pose_score': pose_analysis.get('flight_score', 0),
                'landing_pose_score': pose_analysis.get('landing_score', 0),
                
                # 改善提案
                'improvement_suggestions': self._generate_suggestions(
                    distance_analysis, speed_analysis, technique_analysis, pose_analysis
                ),
                
                # 詳細改善提案（フレームごと）
                'detailed_suggestions': detailed_suggestions,
                
                # 詳細データ
                'phases': phases,
                'pose_data': pose_data[:10]  # 最初の10フレームのみ保存（容量削減）
            }
            
            cap.release()
            
            if log_callback:
                log_callback(f"跳躍分析が完了しました。記録: {analysis_results['jump_distance']:.2f}m")
                
            return analysis_results
            
        except Exception as e:
            if log_callback:
                log_callback(f"分析中にエラーが発生しました: {str(e)}")
            raise e
            
    def _extract_pose_data(self, cap, sorted_frames, progress_callback, log_callback):
        """各フレームからポーズデータを抽出"""
        from ultralytics import YOLO
        
        pose_model = YOLO('yolov8n-pose.pt')
        pose_data = []
        
        total_frames = len(sorted_frames)
        
        for i, frame_info in enumerate(sorted_frames):
            if progress_callback:
                progress = (i / total_frames) * 50  # 分析の前半50%
                progress_callback(progress, f"ポーズ抽出中 {i+1}/{total_frames}")
                
            frame_idx = frame_info['frame_idx']
            bbox = frame_info['bbox']
            
            # フレームを読み込み
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            # バウンディングボックス領域を抽出
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            person_crop = frame[y1:y2, x1:x2]
            
            # ポーズ推定を実行
            results = pose_model(person_crop, verbose=False)
            
            if results and len(results[0].keypoints) > 0:
                keypoints_obj = results[0].keypoints[0]
                # Keypointsオブジェクト → Tensor → NumPy配列
                if hasattr(keypoints_obj, "data"):
                    keypoints = keypoints_obj.data.cpu().numpy()
                else:
                    keypoints = keypoints_obj  # すでにNumPy配列ならそのまま

                import numpy as np
                if not isinstance(keypoints, np.ndarray):
                    raise TypeError(f"keypoints is not a numpy array: {type(keypoints)}")

                adjusted_keypoints = keypoints.copy()
                if len(adjusted_keypoints.shape) == 2:
                    adjusted_keypoints[:, 0] += x1
                    adjusted_keypoints[:, 1] += y1
                    
                pose_data.append({
                    'frame_idx': frame_idx,
                    'timestamp': frame_idx / cap.get(cv2.CAP_PROP_FPS),
                    'keypoints': adjusted_keypoints,
                    'bbox': bbox,
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2
                })
                
        return pose_data
        
    def _detect_jump_phases(self, pose_data, fps, log_callback):
        """跳躍の各フェーズを検出"""
        if len(pose_data) < 10:
            return {}
            
        # 重心の軌跡を計算
        center_positions = []
        timestamps = []
        
        for data in pose_data:
            center_positions.append([data['center_x'], data['center_y']])
            timestamps.append(data['timestamp'])
            
        center_positions = np.array(center_positions)
        timestamps = np.array(timestamps)
        
        # 速度を計算
        velocities = np.diff(center_positions, axis=0) * fps
        speeds = np.linalg.norm(velocities, axis=1)
        
        # フェーズ検出
        phases = {}
        
        try:
            # 踏切線との交差点を検出
            takeoff_line_y = self.config.takeoff_line_y
            takeoff_idx = self._find_takeoff_point(center_positions, takeoff_line_y)
            
            # 着地点を検出（Y座標の急激な変化）
            landing_idx = self._find_landing_point(center_positions, takeoff_idx)
            
            if takeoff_idx > 0 and landing_idx > takeoff_idx:
                # 各フェーズの時間を計算
                phases['approach_duration'] = timestamps[takeoff_idx] - timestamps[0]
                phases['takeoff_duration'] = 0.2  # 踏切は約0.2秒と仮定
                phases['flight_duration'] = timestamps[landing_idx] - timestamps[takeoff_idx]
                phases['landing_duration'] = timestamps[-1] - timestamps[landing_idx]
                
                phases['takeoff_frame'] = takeoff_idx
                phases['landing_frame'] = landing_idx
                
                if log_callback:
                    log_callback(f"フェーズ検出完了: 助走{phases['approach_duration']:.1f}s, "
                               f"飛行{phases['flight_duration']:.1f}s")
                               
        except Exception as e:
            if log_callback:
                log_callback(f"フェーズ検出でエラー: {str(e)}")
                
        return phases
        
    def _find_takeoff_point(self, positions, takeoff_line_y):
        """踏切点を検出"""
        for i in range(1, len(positions)):
            if positions[i-1, 1] > takeoff_line_y and positions[i, 1] <= takeoff_line_y:
                return i
        return len(positions) // 3  # 見つからない場合は1/3地点を仮定
        
    def _find_landing_point(self, positions, takeoff_idx):
        """着地点を検出"""
        if takeoff_idx >= len(positions) - 5:
            return len(positions) - 1
            
        # 踏切後のY座標の変化を分析
        y_positions = positions[takeoff_idx:, 1]
        
        # 最高点を見つける
        min_y_idx = np.argmin(y_positions)
        
        # 最高点以降で急激な下降を検出
        for i in range(min_y_idx + 1, len(y_positions) - 1):
            if y_positions[i+1] - y_positions[i] > 20:  # 急激な下降
                return takeoff_idx + i
                
        return len(positions) - 1
        
    def _analyze_jump_distance(self, pose_data, phases):
        """跳躍距離を分析"""
        if not pose_data or not phases:
            return {'distance': 0, 'takeoff_pos': 0, 'landing_pos': 0}
            
        try:
            takeoff_idx = phases.get('takeoff_frame', 0)
            landing_idx = phases.get('landing_frame', len(pose_data) - 1)
            
            takeoff_pos = pose_data[takeoff_idx]['center_x']
            landing_pos = pose_data[landing_idx]['center_x']
            
            # ピクセル距離をメートルに変換
            pixel_distance = abs(landing_pos - takeoff_pos)
            distance_meters = pixel_distance / self.config.pixel_to_meter_ratio
            
            return {
                'distance': distance_meters,
                'takeoff_pos': takeoff_pos,
                'landing_pos': landing_pos
            }
            
        except Exception:
            return {'distance': 0, 'takeoff_pos': 0, 'landing_pos': 0}
            
    def _analyze_speed(self, pose_data, phases, fps):
        """速度分析を実行"""
        if len(pose_data) < 5:
            return {'max_approach_speed': 0, 'takeoff_speed': 0, 'takeoff_angle': 0}
            
        try:
            # 位置データを抽出
            positions = np.array([[d['center_x'], d['center_y']] for d in pose_data])
            
            # 速度を計算
            velocities = np.diff(positions, axis=0) * fps / self.config.pixel_to_meter_ratio
            speeds = np.linalg.norm(velocities, axis=1)
            
            takeoff_idx = phases.get('takeoff_frame', len(pose_data) // 2)
            
            # 助走最高速度
            approach_speeds = speeds[:min(takeoff_idx, len(speeds))]
            max_approach_speed = np.max(approach_speeds) if len(approach_speeds) > 0 else 0
            
            # 踏切時速度
            takeoff_speed = speeds[min(takeoff_idx, len(speeds) - 1)] if len(speeds) > 0 else 0
            
            # 踏切角度
            takeoff_angle = 0
            if takeoff_idx < len(velocities):
                velocity_vector = velocities[takeoff_idx]
                takeoff_angle = math.degrees(math.atan2(-velocity_vector[1], velocity_vector[0]))
                
            return {
                'max_approach_speed': max_approach_speed,
                'takeoff_speed': takeoff_speed,
                'takeoff_angle': takeoff_angle
            }
            
        except Exception:
            return {'max_approach_speed': 0, 'takeoff_speed': 0, 'takeoff_angle': 0}
            
    def _analyze_technique(self, pose_data, phases):
        """技術分析を実行"""
        if not pose_data or not phases:
            return {'takeoff_foot': 'N/A', 'max_height': 0, 'landing_angle': 0}
            
        try:
            takeoff_idx = phases.get('takeoff_frame', 0)
            landing_idx = phases.get('landing_frame', len(pose_data) - 1)
            
            # 踏切足の判定
            takeoff_foot = self._determine_takeoff_foot(pose_data, takeoff_idx)
            
            # 最高到達点
            max_height = self._calculate_max_height(pose_data, takeoff_idx, landing_idx)
            
            # 着地角度
            landing_angle = self._calculate_landing_angle(pose_data, landing_idx)
            
            return {
                'takeoff_foot': takeoff_foot,
                'max_height': max_height,
                'landing_angle': landing_angle
            }
            
        except Exception:
            return {'takeoff_foot': 'N/A', 'max_height': 0, 'landing_angle': 0}
            
    def _determine_takeoff_foot(self, pose_data, takeoff_idx):
        """踏切足を判定"""
        if takeoff_idx >= len(pose_data):
            return 'N/A'
            
        try:
            keypoints = pose_data[takeoff_idx]['keypoints']
            
            left_ankle = keypoints[self.keypoint_indices['left_ankle']]
            right_ankle = keypoints[self.keypoint_indices['right_ankle']]
            
            # 信頼度チェック
            if left_ankle[2] < 0.5 or right_ankle[2] < 0.5:
                return 'N/A'
                
            # より低い位置（地面に近い）足が踏切足
            if left_ankle[1] > right_ankle[1]:
                return '左足'
            else:
                return '右足'
                
        except Exception:
            return 'N/A'
            
    def _calculate_max_height(self, pose_data, takeoff_idx, landing_idx):
        """最高到達点を計算"""
        try:
            flight_data = pose_data[takeoff_idx:landing_idx+1]
            if not flight_data:
                return 0
                
            min_y = min(d['center_y'] for d in flight_data)
            takeoff_y = pose_data[takeoff_idx]['center_y']
            
            height_pixels = takeoff_y - min_y
            height_meters = height_pixels / self.config.pixel_to_meter_ratio
            
            return max(0, height_meters)
            
        except Exception:
            return 0
            
    def _calculate_landing_angle(self, pose_data, landing_idx):
        """着地角度を計算"""
        try:
            if landing_idx < 2 or landing_idx >= len(pose_data):
                return 0
                
            # 着地前後の位置変化から角度を計算
            pre_pos = pose_data[landing_idx - 2]['center_y']
            post_pos = pose_data[landing_idx]['center_y']
            
            angle = math.degrees(math.atan2(post_pos - pre_pos, 2))
            return abs(angle)
            
        except Exception:
            return 0
            
    def _analyze_pose_quality(self, pose_data, phases):
        """ポーズ品質を分析"""
        if not pose_data or not phases:
            return {'takeoff_score': 0, 'flight_score': 0, 'landing_score': 0}
            
        try:
            takeoff_idx = phases.get('takeoff_frame', 0)
            landing_idx = phases.get('landing_frame', len(pose_data) - 1)
            
            # 各フェーズのポーズスコアを計算
            takeoff_score = self._calculate_pose_score(pose_data, takeoff_idx, 'takeoff')
            
            flight_start = takeoff_idx + 1
            flight_end = landing_idx
            flight_score = self._calculate_phase_pose_score(pose_data, flight_start, flight_end, 'flight')
            
            landing_score = self._calculate_pose_score(pose_data, landing_idx, 'landing')
            
            return {
                'takeoff_score': takeoff_score,
                'flight_score': flight_score,
                'landing_score': landing_score
            }
            
        except Exception:
            return {'takeoff_score': 0, 'flight_score': 0, 'landing_score': 0}
            
    def _calculate_pose_score(self, pose_data, frame_idx, phase):
        """特定フレームのポーズスコアを計算"""
        if frame_idx >= len(pose_data):
            return 0
            
        try:
            keypoints = pose_data[frame_idx]['keypoints']
            score = 0
            
            # 基本的な姿勢評価
            if phase == 'takeoff':
                score = self._evaluate_takeoff_pose(keypoints)
            elif phase == 'flight':
                score = self._evaluate_flight_pose(keypoints)
            elif phase == 'landing':
                score = self._evaluate_landing_pose(keypoints)
                
            return min(10, max(0, score))
            
        except Exception:
            return 0
            
    def _calculate_phase_pose_score(self, pose_data, start_idx, end_idx, phase):
        """フェーズ全体のポーズスコアを計算"""
        if start_idx >= end_idx or end_idx > len(pose_data):
            return 0
            
        scores = []
        for i in range(start_idx, min(end_idx, len(pose_data))):
            score = self._calculate_pose_score(pose_data, i, phase)
            scores.append(score)
            
        return np.mean(scores) if scores else 0
        
    def _evaluate_takeoff_pose(self, keypoints):
        """踏切時のポーズを評価"""
        score = 5.0  # 基本スコア
        
        try:
            # 体の直立性をチェック
            left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
            right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
            left_hip = keypoints[self.keypoint_indices['left_hip']]
            right_hip = keypoints[self.keypoint_indices['right_hip']]
            
            # 肩と腰のバランス
            if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_balance = abs(left_shoulder[1] - right_shoulder[1])
                hip_balance = abs(left_hip[1] - right_hip[1])
                
                if shoulder_balance < 20 and hip_balance < 20:
                    score += 2.0
                    
                # 前傾角度の評価
                torso_angle = self._calculate_torso_angle(keypoints)
                if 10 <= torso_angle <= 30:  # 理想的な前傾角度
                    score += 2.0
                elif torso_angle < 45:
                    score += 1.0
                    
        except Exception:
            pass
            
        return score
        
    def _evaluate_flight_pose(self, keypoints):
        """飛行時のポーズを評価"""
        score = 5.0
        
        try:
            # 空中での姿勢の安定性
            left_knee = keypoints[self.keypoint_indices['left_knee']]
            right_knee = keypoints[self.keypoint_indices['right_knee']]
            
            if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                knee_height_diff = abs(left_knee[1] - right_knee[1])
                if knee_height_diff < 30:  # 膝の高さが揃っている
                    score += 2.0
                    
            # 腕の位置
            left_wrist = keypoints[self.keypoint_indices['left_wrist']]
            right_wrist = keypoints[self.keypoint_indices['right_wrist']]
            
            if left_wrist[2] > 0.5 and right_wrist[2] > 0.5:
                # 腕が適切に上がっているか
                left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
                right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
                
                if (left_shoulder[2] > 0.5 and left_wrist[1] < left_shoulder[1] and
                    right_shoulder[2] > 0.5 and right_wrist[1] < right_shoulder[1]):
                    score += 2.0
                    
        except Exception:
            pass
            
        return score
        
    def _evaluate_landing_pose(self, keypoints):
        """着地時のポーズを評価"""
        score = 5.0
        
        try:
            # 着地時の足の位置
            left_ankle = keypoints[self.keypoint_indices['left_ankle']]
            right_ankle = keypoints[self.keypoint_indices['right_ankle']]
            
            if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
                ankle_distance = abs(left_ankle[0] - right_ankle[0])
                if 20 <= ankle_distance <= 60:  # 適切な足幅
                    score += 2.0
                    
            # 前方への体重移動
            nose = keypoints[self.keypoint_indices['nose']]
            left_hip = keypoints[self.keypoint_indices['left_hip']]
            right_hip = keypoints[self.keypoint_indices['right_hip']]
            
            if all(kp[2] > 0.5 for kp in [nose, left_hip, right_hip]):
                hip_center_x = (left_hip[0] + right_hip[0]) / 2
                if nose[0] > hip_center_x:  # 頭が腰より前
                    score += 2.0
                    
        except Exception:
            pass
            
        return score
        
    def _calculate_torso_angle(self, keypoints):
        """体幹の角度を計算"""
        try:
            left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
            right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
            left_hip = keypoints[self.keypoint_indices['left_hip']]
            right_hip = keypoints[self.keypoint_indices['right_hip']]
            
            if all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_center = [(left_shoulder[0] + right_shoulder[0]) / 2,
                                 (left_shoulder[1] + right_shoulder[1]) / 2]
                hip_center = [(left_hip[0] + right_hip[0]) / 2,
                             (left_hip[1] + right_hip[1]) / 2]
                
                angle = math.degrees(math.atan2(
                    hip_center[1] - shoulder_center[1],
                    hip_center[0] - shoulder_center[0]
                ))
                
                return abs(90 - abs(angle))  # 垂直からの角度
                
        except Exception:
            pass
            
        return 0
        
    def _collect_frame_suggestions(self, pose_data, phases):
        """フレームごとの改善提案を収集"""
        suggestions = []
        
        takeoff_idx = phases.get('takeoff_frame', 0)
        landing_idx = phases.get('landing_frame', len(pose_data) - 1)
        
        for i, data in enumerate(pose_data):
            frame_suggestions = self._generate_frame_suggestions(data['keypoints'], i, takeoff_idx, landing_idx)
            if frame_suggestions:
                suggestions.extend(frame_suggestions)
                
        # 重複を除去
        return list(set(suggestions))
        
    def _generate_frame_suggestions(self, keypoints, frame_idx, takeoff_idx, landing_idx):
        """フレームごとの改善提案を生成"""
        suggestions = []
        
        try:
            # フェーズ判定
            if frame_idx == takeoff_idx:
                phase = 'takeoff'
            elif takeoff_idx < frame_idx < landing_idx:
                phase = 'flight'
            elif frame_idx == landing_idx:
                phase = 'landing'
            else:
                phase = 'approach'
                
            # キーポイントの位置関係に基づく提案
            if phase == 'takeoff':
                torso_angle = self._calculate_torso_angle(keypoints)
                if torso_angle < 10:
                    suggestions.append("踏切時: 体をより前傾させてください")
                elif torso_angle > 45:
                    suggestions.append("踏切時: 前傾しすぎないように注意してください")
                    
                # 膝の位置
                left_knee = keypoints[self.keypoint_indices['left_knee']]
                right_knee = keypoints[self.keypoint_indices['right_knee']]
                if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                    if left_knee[1] > right_knee[1] + 20:
                        suggestions.append("踏切時: 右膝を高く上げてください")
                    elif right_knee[1] > left_knee[1] + 20:
                        suggestions.append("踏切時: 左膝を高く上げてください")
                        
            elif phase == 'flight':
                # 腕の位置
                left_wrist = keypoints[self.keypoint_indices['left_wrist']]
                right_wrist = keypoints[self.keypoint_indices['right_wrist']]
                left_shoulder = keypoints[self.keypoint_indices['left_shoulder']]
                right_shoulder = keypoints[self.keypoint_indices['right_shoulder']]
                
                if (left_wrist[2] > 0.5 and left_shoulder[2] > 0.5 and left_wrist[1] > left_shoulder[1]):
                    suggestions.append("飛行時: 左腕をより高く上げてください")
                if (right_wrist[2] > 0.5 and right_shoulder[2] > 0.5 and right_wrist[1] > right_shoulder[1]):
                    suggestions.append("飛行時: 右腕をより高く上げてください")
                    
                # 膝の揃い
                left_knee = keypoints[self.keypoint_indices['left_knee']]
                right_knee = keypoints[self.keypoint_indices['right_knee']]
                if left_knee[2] > 0.5 and right_knee[2] > 0.5:
                    knee_diff = abs(left_knee[1] - right_knee[1])
                    if knee_diff > 50:
                        suggestions.append("飛行時: 膝の高さを揃えてください")
                        
            elif phase == 'landing':
                # 着地時の足幅
                left_ankle = keypoints[self.keypoint_indices['left_ankle']]
                right_ankle = keypoints[self.keypoint_indices['right_ankle']]
                if left_ankle[2] > 0.5 and right_ankle[2] > 0.5:
                    ankle_dist = abs(left_ankle[0] - right_ankle[0])
                    if ankle_dist < 20:
                        suggestions.append("着地時: 足幅を広げてください")
                    elif ankle_dist > 80:
                        suggestions.append("着地時: 足幅を狭くしてください")
                        
                # 前方への体重移動
                nose = keypoints[self.keypoint_indices['nose']]
                left_hip = keypoints[self.keypoint_indices['left_hip']]
                right_hip = keypoints[self.keypoint_indices['right_hip']]
                if all(kp[2] > 0.5 for kp in [nose, left_hip, right_hip]):
                    hip_center_x = (left_hip[0] + right_hip[0]) / 2
                    if nose[0] < hip_center_x:
                        suggestions.append("着地時: 体重をより前方に移動させてください")
                        
        except Exception:
            pass
            
        return suggestions
        
    def _generate_suggestions(self, distance_analysis, speed_analysis, technique_analysis, pose_analysis):
        """改善提案を生成"""
        suggestions = []
        
        # 距離に基づく提案
        distance = distance_analysis.get('distance', 0)
        if distance < 4.0:
            suggestions.append("• 助走速度を上げて、より力強い踏切を心がけましょう")
        elif distance > 7.0:
            suggestions.append("• 素晴らしい記録です！技術の安定性を重視しましょう")
            
        # 速度に基づく提案
        max_speed = speed_analysis.get('max_approach_speed', 0)
        if max_speed < 8.0:
            suggestions.append("• 助走速度が不足しています。スプリント練習を強化しましょう")
            
        takeoff_angle = speed_analysis.get('takeoff_angle', 0)
        if takeoff_angle < 15:
            suggestions.append("• 踏切角度が浅すぎます。より高く跳び上がることを意識しましょう")
        elif takeoff_angle > 25:
            suggestions.append("• 踏切角度が急すぎます。前方への推進力を重視しましょう")
            
        # ポーズに基づく提案
        takeoff_score = pose_analysis.get('takeoff_score', 0)
        if takeoff_score < 6:
            suggestions.append("• 踏切時の姿勢を改善しましょう。体幹を安定させることが重要です")
            
        flight_score = pose_analysis.get('flight_score', 0)
        if flight_score < 6:
            suggestions.append("• 空中姿勢を改善しましょう。膝を高く上げ、腕を効果的に使いましょう")
            
        landing_score = pose_analysis.get('landing_score', 0)
        if landing_score < 6:
            suggestions.append("• 着地技術を向上させましょう。前方への体重移動を意識しましょう")
            
        if not suggestions:
            suggestions.append("• 全体的に良好な技術です。継続的な練習で更なる向上を目指しましょう")
            
        return "\n".join(suggestions)