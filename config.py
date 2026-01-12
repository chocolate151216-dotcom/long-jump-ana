import os

class Config:
    """走り幅跳び分析アプリケーションの設定クラス"""
    
    def __init__(self):
        # 検出パラメータ
        self.detection_threshold = 0.6  # 走り幅跳びでは高い精度が必要
        self.pose_confidence_threshold = 0.5
        
        # DeepSORT パラメータ
        self.max_age = 50          # 最大フレーム数（短距離なので短く）
        self.n_init = 2            # 確認に必要な連続検出数
        self.nn_budget = 100       # 外観記述子ギャラリーの最大サイズ
        self.max_cosine_distance = 0.2  # マッチングの最大コサイン距離（厳しく）
        
        # 動画処理パラメータ
        self.output_fps = 30
        self.output_quality = 95
        
        # UI パラメータ
        self.thumbnail_size = (120, 180)
        self.preview_cols = 4
        
        # モデルパラメータ
        self.detection_model_name = 'yolov8n.pt'
        self.pose_model_name = 'yolov8n-pose.pt'
        self.pose_model_path = os.path.join(os.getcwd(), "yolov8x-pose.pt")  # 再学習モデルのパス
        
        # 走り幅跳び特有のパラメータ
        self.takeoff_line_y = 400          # 踏切線のY座標
        self.landing_area_start_x = 500    # 砂場開始位置のX座標
        self.pixel_to_meter_ratio = 100.0  # ピクセル/メートル比
        
        # 跳躍フェーズ検出パラメータ
        self.approach_speed_threshold = 2.0    # 助走速度閾値 (m/s)
        self.takeoff_duration_max = 0.3        # 踏切フェーズ最大時間 (秒)
        self.landing_impact_threshold = 50     # 着地衝撃検出閾値
        
        # ポーズ分析パラメータ
        self.pose_analysis_keypoints = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle'
        ]
        
        # ポーズ描画間隔
        self.pose_draw_interval = 1  # 1/3/5フレームごとに描画を切り替え
        
    def update_detection_threshold(self, threshold):
        """検出信頼度閾値を更新"""
        self.detection_threshold = max(0.1, min(1.0, threshold))
        
    def update_pose_threshold(self, threshold):
        """ポーズ信頼度閾値を更新"""
        self.pose_confidence_threshold = max(0.1, min(1.0, threshold))
        
    def update_calibration(self, takeoff_line_y, landing_area_start_x, pixel_to_meter_ratio):
        """キャリブレーション設定を更新"""
        self.takeoff_line_y = takeoff_line_y
        self.landing_area_start_x = landing_area_start_x
        self.pixel_to_meter_ratio = pixel_to_meter_ratio
        
    def get_tracker_config(self):
        """DeepSORT トラッカーの設定を取得"""
        return {
            'max_age': self.max_age,
            'n_init': self.n_init,
            'nn_budget': self.nn_budget,
            'max_cosine_distance': self.max_cosine_distance
        }
        
    def get_jump_analysis_config(self):
        """跳躍分析の設定を取得"""
        return {
            'takeoff_line_y': self.takeoff_line_y,
            'landing_area_start_x': self.landing_area_start_x,
            'pixel_to_meter_ratio': self.pixel_to_meter_ratio,
            'approach_speed_threshold': self.approach_speed_threshold,
            'takeoff_duration_max': self.takeoff_duration_max,
            'landing_impact_threshold': self.landing_impact_threshold
        }