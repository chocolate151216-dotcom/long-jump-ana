import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import cv2
from PIL import Image, ImageTk
import os
import json
from video_processor import VideoProcessor
from config import Config
from jump_analyzer import JumpAnalyzer
from pose_estimator import PoseEstimator  
import subprocess
import shutil

class LongJumpApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("走り幅跳び分析システム - Long Jump Analysis System")
        self.geometry("1400x900")
        self.configure(bg="#f0f8ff")

        self.config = Config()
        # CUDA 非対応でも落ちないよう既定は CPU
        setattr(self.config, "device", "cpu")
        self.jump_analyzer = JumpAnalyzer(self.config)  # ← 順序を前に移動
        self.video_processor = VideoProcessor(self.config, self.jump_analyzer)
        self.pose_estimator = PoseEstimator()  # ← これを追加
        
        # Variables
        self.input_video_path = tk.StringVar()
        self.output_video_path = tk.StringVar()
        self.selected_athlete_id = tk.IntVar(value=-1)
        self.processing = False
        self.detection_results = {}
        self.jump_analysis_results = {}
        
        # Jump analysis parameters
        self.takeoff_line_y = tk.IntVar(value=400)  # 踏切線のY座標
        self.landing_area_start_x = tk.IntVar(value=500)  # 砂場開始位置
        self.pixel_to_meter_ratio = tk.DoubleVar(value=self.config.pixel_to_meter_ratio)
        self.keypoint_interval = tk.IntVar(value=getattr(self.config, "keypoint_interval", 1))
        self.pose_draw_interval = tk.StringVar(value="1")
        
        self.setup_ui()
        self.log_message("ログ欄テスト表示")
        
        
    def setup_ui(self):
        """ユーザーインターフェースの設定"""
        # メインコンテナ
        # スクロール可能なキャンバスを作成
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 以降、main_frameの代わりにscrollable_frameを使ってウィジェットを配置
        # 例: title_label = ttk.Label(scrollable_frame, text="走り幅跳び分析システム", 
        #                           font=("Arial", 18, "bold"), foreground="#2c3e50")
        # title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # グリッド重み設定
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        scrollable_frame.columnconfigure(1, weight=1)
        
        # タイトル
        title_label = ttk.Label(scrollable_frame, text="走り幅跳び分析システム", 
                               font=("Arial", 18, "bold"), foreground="#2c3e50")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        subtitle_label = ttk.Label(scrollable_frame, text="Long Jump Analysis System", 
                                 font=("Arial", 12), foreground="#7f8c8d")
        subtitle_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # 動画入力セクション
        input_frame = ttk.LabelFrame(scrollable_frame, text="動画入力 (Video Input)", padding="10")
        input_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        input_frame.columnconfigure(1, weight=1)
        
        ttk.Label(input_frame, text="入力動画:").grid(row=0, column=0, padx=(0, 10))
        ttk.Entry(input_frame, textvariable=self.input_video_path, state="readonly").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(input_frame, text="参照", command=self.browse_input_video).grid(
            row=0, column=2)
       # 追加: 前処理済み動画をワンクリックでセットするボタン
        ttk.Button(input_frame, text="前処理動画を読み込む", command=self.load_preprocessed_video).grid(
            row=1, column=0, columnspan=3, pady=(8, 0))
        # 動画前処理セクション
        preprocess_frame = ttk.LabelFrame(scrollable_frame, text="動画前処理 (Preprocessing)", padding="10")
        preprocess_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        ttk.Button(preprocess_frame, text="動画を前処理", command=self.start_preprocessing).grid(row=0, column=0, padx=(0, 10))
        # キャリブレーション設定
        calib_frame = ttk.LabelFrame(scrollable_frame, text="キャリブレーション設定 (Calibration)", padding="10")
        calib_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        ground_calib_btn = ttk.Button(calib_frame, text="地面角度キャリブレーション", command=self.calibrate_ground_angle)
        ground_calib_btn.grid(row=2, column=3, columnspan=2, pady=(10, 0))
        ttk.Button(calib_frame, text="フレーム上で指定", command=self.calibrate_from_frame).grid(
            row=2, column=0, columnspan=6, pady=(10, 0)
)

        ttk.Label(calib_frame, text="踏切線Y座標:").grid(row=0, column=0, padx=(0, 10))
        ttk.Scale(calib_frame, from_=0, to=1000, variable=self.takeoff_line_y, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, padx=(0, 10))
        ttk.Label(calib_frame, textvariable=self.takeoff_line_y).grid(row=0, column=2, padx=(0, 20))
        
        ttk.Label(calib_frame, text="砂場開始X座標:").grid(row=0, column=3, padx=(0, 10))
        ttk.Scale(calib_frame, from_=0, to=1500, variable=self.landing_area_start_x, 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=4, padx=(0, 10))
        ttk.Label(calib_frame, textvariable=self.landing_area_start_x).grid(row=0, column=5, padx=(0, 20))
        
        ttk.Label(calib_frame, text="ピクセル/メートル比:").grid(row=1, column=0, padx=(0, 10))
        ttk.Scale(calib_frame, from_=50.0, to=200.0, variable=self.pixel_to_meter_ratio, 
                 orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, padx=(0, 10))
        ttk.Label(calib_frame, textvariable=self.pixel_to_meter_ratio).grid(row=1, column=2)
        
        # 基準距離を指定ボタン
        ttk.Button(calib_frame, text="基準距離を指定", command=self.set_pixel_to_meter_by_two_points).grid(
            row=2, column=1, columnspan=2, pady=(10, 0)
)
    
        # 処理セクション
        process_frame = ttk.LabelFrame(scrollable_frame, text="処理 (Processing)", padding="10")
        process_frame.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.detect_button = ttk.Button(process_frame, text="選手検出・追跡", 
                                       command=self.start_detection)
        self.detect_button.grid(row=0, column=0, padx=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var, 
                                          maximum=100, length=400)
        self.progress_bar.grid(row=0, column=1, padx=(0, 10))
        
        self.status_label = ttk.Label(process_frame, text="準備完了")
        self.status_label.grid(row=0, column=2)

        # 踏切フェーズフレーム指定
        self.takeoff_start_frame = tk.IntVar(value=0)
        self.takeoff_end_frame = tk.IntVar(value=0)

        ttk.Label(process_frame, text="踏切開始フレーム:").grid(row=1, column=0, padx=(0, 10))
        ttk.Entry(process_frame, textvariable=self.takeoff_start_frame, width=8).grid(row=1, column=1, padx=(0, 10))
        ttk.Label(process_frame, text="踏切終了フレーム:").grid(row=1, column=2, padx=(0, 10))
        ttk.Entry(process_frame, textvariable=self.takeoff_end_frame, width=8).grid(row=1, column=3, padx=(0, 10))
        
        # 選手選択セクション
        self.selection_frame = ttk.LabelFrame(scrollable_frame, text="選手選択 (Athlete Selection)", padding="10")
        self.selection_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # プレビューエリア
        self.preview_frame = ttk.Frame(self.selection_frame)
        self.preview_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        # 選手選択コントロール
        selection_controls = ttk.Frame(self.selection_frame)
        selection_controls.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Label(selection_controls, text="選手ID選択:").grid(row=0, column=0, padx=(0, 10))
        self.athlete_combo = ttk.Combobox(selection_controls, textvariable=self.selected_athlete_id, 
                                        state="readonly", width=10)
        self.athlete_combo.grid(row=0, column=1, padx=(0, 10))
        
        self.analyze_button = ttk.Button(selection_controls, text="跳躍分析実行", 
                                       command=self.analyze_jump, state="disabled")
        self.analyze_button.grid(row=0, column=2, padx=(0, 10))
        
        self.pose_button = ttk.Button(selection_controls, text="ポーズ動画生成", 
                                     command=self.generate_pose_video, state="disabled")
        self.pose_button.grid(row=0, column=3)
        
        # 分析結果表示
        results_frame = ttk.LabelFrame(scrollable_frame, text="分析結果 (Analysis Results)", padding="10")
        results_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        
        # 結果表示用のテキストウィジェット
        self.results_text = tk.Text(results_frame, height=8, wrap=tk.WORD, font=("Courier", 10))
        results_scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=results_scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        results_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # 出力セクション
        output_frame = ttk.LabelFrame(scrollable_frame, text="出力 (Output)", padding="10")
        output_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)
        
        ttk.Label(output_frame, text="出力動画:").grid(row=0, column=0, padx=(0, 10))
        ttk.Entry(output_frame, textvariable=self.output_video_path, state="readonly").grid(
            row=0, column=1, sticky=(tk.W, tk.E), padx=(0, 10))
        ttk.Button(output_frame, text="参照", command=self.browse_output_video).grid(
            row=0, column=2, padx=(0, 10))
        
        ttk.Button(output_frame, text="結果をJSONで保存", command=self.save_results_json).grid(
            row=0, column=3)
        
        # ログセクション
        log_frame = ttk.LabelFrame(scrollable_frame, text="ログ (Log)", padding="10")
        log_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=12, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        for i in range(8):
            scrollable_frame.rowconfigure(i, weight=0)
        scrollable_frame.rowconfigure(9, weight=10)  # ログ欄の行だけ大きなweight

        # 残像に残す関節選択
        self.keypoint_vars = [tk.BooleanVar(value=False) for _ in range(17)]
        keypoint_names = [
            "鼻", "左目", "右目", "左耳", "右耳", "左肩", "右肩", "左肘", "右肘",
            "左手首", "右手首", "左股関節", "右股関節", "左膝", "右膝", "左足首", "右足首"
        ]
        keypoint_frame = ttk.LabelFrame(scrollable_frame, text="残像に残す関節選択", padding="10")
        keypoint_frame.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        for i, name in enumerate(keypoint_names):
            ttk.Checkbutton(keypoint_frame, text=name, variable=self.keypoint_vars[i]).grid(row=i//6, column=i%6)

        # 設定セクション
        settings_frame = ttk.LabelFrame(scrollable_frame, text="設定 (Settings)", padding="10")
        settings_frame.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        settings_frame.columnconfigure(1, weight=1)
        
        ttk.Label(settings_frame, text="キャリブレーション間隔:").grid(row=0, column=0, padx=(0, 10))
        ttk.Combobox(
            settings_frame,
            textvariable=self.keypoint_interval,
            values=("1", "3", "5"),
            state="readonly",
            width=6
        ).grid(row=0, column=1, padx=(0, 10))
        
        ttk.Label(settings_frame, text="描画キーポイント:").grid(row=1, column=0, padx=(0, 10))
        ttk.Combobox(
            settings_frame,
            textvariable=self.pose_draw_interval,
            values=("1", "3", "5"),
            state="readonly",
            width=6
        ).grid(row=1, column=1, padx=(0, 10))
        
        ttk.Button(settings_frame, text="設定を適用", command=self.apply_settings).grid(row=2, column=0, columnspan=2, pady=(10, 0))

        # ← ここにポーズモデル選択UIを追加
        ttk.Label(settings_frame, text="ポーズモデル:").grid(row=3, column=0, padx=(0, 10))
        self.pose_model_path_var = tk.StringVar(value=os.path.join(os.getcwd(), "yolov8x-pose.pt"))
        ttk.Entry(settings_frame, textvariable=self.pose_model_path_var, width=40).grid(row=3, column=1, padx=(0, 10))
        ttk.Button(settings_frame, text="参照", command=self.browse_pose_model).grid(row=3, column=2, padx=(0, 10))
        
        # 残像表示設定
        self.show_trail = tk.BooleanVar(value=True)
        ttk.Checkbutton(scrollable_frame, text="残像を表示", variable=self.show_trail).grid(row=12, column=0, columnspan=3, pady=(10, 0))
        
        # 動画処理時のスレッド数
        self.thread_count = tk.IntVar(value=4)
        ttk.Label(scrollable_frame, text="動画処理スレッド数:").grid(row=13, column=0, padx=(0, 10))
        ttk.Spinbox(
            scrollable_frame,
            from_=1, to=8,
            textvariable=self.thread_count,
            width=5
        ).grid(row=13, column=1, padx=(0, 10))
        
        ttk.Button(scrollable_frame, text="スレッド数を設定", command=self.set_thread_count).grid(row=13, column=2, pady=(10, 0))
        
        # ログウィンドウ
        ttk.Button(scrollable_frame, text="ログウィンドウ", command=self.show_log_window).grid(row=14, column=0, columnspan=3, pady=(10, 0))
        
        # ショートカット説明
        shortcut_info = """
ショートカットキー:
- スペース: 一時停止 / 再生
- c: 現在フレームをキャリブレーションポイントに設定
- d: 次のフレームにスキップ
- a: 前のフレームにスキップ
- q, Esc: 終了
"""
        ttk.Label(scrollable_frame, text=shortcut_info, font=("Courier", 10), justify=tk.LEFT).grid(
            row=14, column=0, columnspan=3, pady=(10, 0))
        
        # デフォルトボタン
        ttk.Button(scrollable_frame, text="デフォルト設定を読み込む", command=self.load_default_settings).grid(
            row=15, column=0, columnspan=3, pady=(10, 0))
        
        # ウィンドウサイズリセットボタン
        ttk.Button(scrollable_frame, text="ウィンドウサイズをリセット", command=self.reset_window_size).grid(
            row=16, column=0, columnspan=3, pady=(10, 0))
        
        # ヘルプボタン
        ttk.Button(scrollable_frame, text="ヘルプ", command=self.show_help).grid(
            row=17, column=0, columnspan=3, pady=(10, 0))

        for i in range(18):
            scrollable_frame.rowconfigure(i, weight=0)
        scrollable_frame.rowconfigure(9, weight=10)  # ログ欄の行だけ大きなweight

    def get_point_from_frame(self, frame, message="位置をクリックしてください"):
        import cv2
        clone = frame.copy()
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((x, y))
                cv2.destroyAllWindows()

        self.withdraw()

        window_name = "Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # ★ウィンドウを明示的に作成
        cv2.imshow(window_name, clone)
        cv2.setMouseCallback(window_name, click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        self.deiconify()

        return points[0] if points else None
    
    def select_frame_and_get_point(self, video_path, window_name="Calibration"):
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        point = None

        self.withdraw()  # Tkinterウィンドウを隠す

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 先にウィンドウを作成

        click_mode = False
        pts = []

        def click_event(event, x, y, flags, param):
            if click_mode and event == cv2.EVENT_LBUTTONDOWN:
                pts.append((x, y))

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            disp = frame.copy()
            cv2.putText(disp, f"Frame: {idx+1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(window_name, disp)

            # ★ここでウィンドウが表示された後にsetMouseCallbackを呼ぶ
            if click_mode:
                cv2.setMouseCallback(window_name, click_event)
            else:
                cv2.setMouseCallback(window_name, lambda *a: None)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('d') or key == 83:  # → or d
                idx = min(idx+1, total-1)
            elif key == ord('a') or key == 81:  # ← or a
                idx = max(idx-1, 0)
            elif key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                click_mode = True
                pts.clear()
                while True:
                    cv2.imshow(window_name, disp)
                    cv2.setMouseCallback(window_name, click_event)  # ★ここで再度セット
                    k = cv2.waitKey(20) & 0xFF
                    if k == 27:  # ESCでキャンセル
                        click_mode = False
                        break
                    if len(pts) > 0:
                        point = (idx, pts[0])
                        click_mode = False
                        break
                cv2.setMouseCallback(window_name, lambda *a: None)
                if point:
                    break

        cap.release()
        cv2.destroyAllWindows()
        self.deiconify()
        return point  # (フレーム番号, (x, y))

    def calibrate_from_frame(self):
        """動画の任意フレームでクリックしてキャリブレーション"""
        video_path = self.input_video_path.get()
        if not video_path:
            messagebox.showerror("エラー", "入力動画ファイルを選択してください")
            return

        # 踏切線Y座標
        messagebox.showinfo("キャリブレーション", "踏切線の位置を選択してください（フレーム移動→cでクリック）")
        result1 = self.select_frame_and_get_point(video_path, window_name="takeoff_line_select")
        if result1 is None:
            return
        frame_idx1, (x1, y1) = result1
        self.takeoff_line_y.set(y1)
        self.log_message(f"踏切線Y座標: {y1}（フレーム: {frame_idx1}）")

        # 砂場開始X座標
        messagebox.showinfo("キャリブレーション", "砂場開始位置を選択してください（フレーム移動→cでクリック）")
        result2 = self.select_frame_and_get_point(video_path, window_name="landing_area_select")
        if result2 is None:
            return
        frame_idx2, (x2, y2) = result2
        self.landing_area_start_x.set(x2)
        self.log_message(f"砂場開始X座標: {x2}（フレーム: {frame_idx2}）")
        
        result = self.select_frame_and_get_point(self.input_video_path.get())
        if result:
            frame_idx, (x, y) = result
            # ここでx, yを踏切線や砂場開始位置に使う
            self.calibrate_takeoff_line()
            
    def calibrate_takeoff_line(self):
        result = self.select_frame_and_get_point(self.input_video_path.get())
        if result:
            frame_idx, (x, y) = result
            self.takeoff_line_y.set(y)
            self.log_message(f"踏切線Y座標: {y}（フレーム: {frame_idx}）")

    def calibrate_landing_area(self):
        result = self.select_frame_and_get_point(self.input_video_path.get())
        if result:
            frame_idx, (x, y) = result
            self.landing_area_start_x.set(x)
            self.log_message(f"砂場開始X座標: {x}（フレーム: {frame_idx}）")

    def browse_input_video(self):
        """入力動画ファイルの参照"""
        filename = filedialog.askopenfilename(
            title="入力動画を選択",
            filetypes=[("動画ファイル", "*.mp4 *.avi *.mov *.mkv"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.input_video_path.set(filename)
            self.log_message(f"入力動画が選択されました: {os.path.basename(filename)}")
            # 動画変更時に検出結果をリセット
            self.detection_results = {}
            self.athlete_combo['values'] = []
            self.athlete_combo.set('')
            
    def browse_output_video(self):
        """出力動画の保存場所を参照"""
        filename = filedialog.asksaveasfilename(
            title="出力動画を保存",
            defaultextension=".mp4",
            filetypes=[("MP4ファイル", "*.mp4"), ("AVIファイル", "*.avi"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.output_video_path.set(filename)
            self.log_message(f"出力動画パスが設定されました: {os.path.basename(filename)}")

    def load_preprocessed_video(self):
        """ユーザーが前処理済み動画を選択して読み込む"""
        initial_default = r"C:\Users\choco\OneDrive\Documents\f6621c3c-c557-4220-9a36-dd6008d10695.mp4"
        initialdir = os.path.dirname(initial_default) if os.path.exists(initial_default) else os.path.expanduser("~")
        filename = filedialog.askopenfilename(
            title="前処理済み動画を選択",
            initialdir=initialdir,
            filetypes=[("動画ファイル", "*.mp4 *.avi *.mov *.mkv"), ("すべてのファイル", "*.*")]
        )
        if not filename:
            return
        self.input_video_path.set(filename)
        self.log_message(f"前処理済み動画をセットしました: {os.path.basename(filename)}")
        messagebox.showinfo("セット完了", f"前処理済み動画をセットしました:\n{filename}")
        # 動画変更時に検出結果をリセット
        self.detection_results = {}
        self.athlete_combo['values'] = []
        self.athlete_combo.set('')
 
         # キャリブレーション設定
        self.config.update_calibration(
             takeoff_line_y=self.takeoff_line_y.get(),
             landing_area_start_x=self.landing_area_start_x.get(),
             pixel_to_meter_ratio=self.pixel_to_meter_ratio.get()
         )
        self.config.keypoint_interval = self.keypoint_interval.get()
        
    def log_message(self, message):
        """ログにメッセージを追加"""
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.update_idletasks()
        
    def update_progress(self, value, status="処理中..."):
        """プログレスバーとステータスを更新"""
        self.progress_var.set(value)
        self.status_label.config(text=status)
        self.update_idletasks()
        
    def start_detection(self):
        """検出・追跡処理を開始"""
        if not self.input_video_path.get():
            messagebox.showerror("エラー", "入力動画ファイルを選択してください")
            return
            
        if self.processing:
            return
            
        self.processing = True
        self.detect_button.config(state="disabled")
        self.analyze_button.config(state="disabled")
        self.pose_button.config(state="disabled")
        
        # 別スレッドで検出処理を開始
        thread = threading.Thread(target=self._detection_worker)
        thread.daemon = True
        thread.start()
        
    def _detection_worker(self):
        """検出・追跡のワーカースレッド"""
        try:
            self.log_message("選手の検出・追跡を開始しています...")
            self.update_progress(0, "初期化中...")
            
            # キャリブレーション設定を更新
            self.config.update_calibration(
                takeoff_line_y=self.takeoff_line_y.get(),
                landing_area_start_x=self.landing_area_start_x.get(),
                pixel_to_meter_ratio=self.pixel_to_meter_ratio.get()
            )
            
            # 動画処理で検出・追跡を実行
            results = self.video_processor.detect_and_track(
                self.input_video_path.get(),
                progress_callback=self.update_progress,
                log_callback=self.log_message
            )
            
            if results:
                self.detection_results = results
                self._show_detection_results()
                self.update_progress(100, "検出完了")
                self.log_message("選手の検出・追跡が正常に完了しました！")
            else:
                self.log_message("動画内で選手が検出されませんでした")
                
        except Exception as e:
            self.log_message(f"検出中にエラーが発生しました: {str(e)}")
            messagebox.showerror("エラー", f"検出に失敗しました: {str(e)}")
        finally:
            self.processing = False
            self.detect_button.config(state="normal")
            
    def _show_detection_results(self):
        """検出結果を表示し、選手選択を有効化"""
        if not self.detection_results:
            return
            
        # キーをstrに変換
        self.detection_results = {str(k): v for k, v in self.detection_results.items()}
            
        # 前回のプレビューをクリア
        for widget in self.preview_frame.winfo_children():
            widget.destroy()
            
        # 検出された各選手のプレビュー画像を作成
        athlete_ids = [str(aid) for aid in self.detection_results.keys()]
        
        self.log_message(f"動画内で{len(athlete_ids)}人の選手を発見しました")
        
        # コンボボックスを選手IDで更新
        self.athlete_combo['values'] = athlete_ids
        if athlete_ids:
            self.athlete_combo.set(athlete_ids[0])
            
        # プレビューサムネイルを作成
        row = 0
        col = 0
        max_cols = 4
        
        for athlete_id in athlete_ids:
            frames_info = self.detection_results[athlete_id]
            if frames_info:
                # この選手が最初に現れるフレームを使用
                frame_info = frames_info[0]
                thumbnail = self._create_athlete_thumbnail(frame_info)
                
                if thumbnail:
                    # この選手用のフレームを作成
                    athlete_frame = ttk.Frame(self.preview_frame, relief="ridge", borderwidth=2)
                    athlete_frame.grid(row=row, column=col, padx=5, pady=5)
                    
                    # サムネイルを追加
                    photo = ImageTk.PhotoImage(thumbnail)
                    label = ttk.Label(athlete_frame, image=photo)
                    label.image = photo  # 参照を保持
                    label.grid(row=0, column=0)
                    
                    # 選手IDラベルを追加
                    id_label = ttk.Label(athlete_frame, text=f"選手ID: {athlete_id}", 
                                       font=("Arial", 10, "bold"))
                    id_label.grid(row=1, column=0, pady=(5, 0))
                    
                    # クリックハンドラーを追加
                    def select_athlete(aid=athlete_id):
                        self.athlete_combo.set(aid)
                        
                    label.bind("<Button-1>", lambda e, aid=athlete_id: select_athlete(aid))
                    athlete_frame.bind("<Button-1>", lambda e, aid=athlete_id: select_athlete(aid))
                    
                    col += 1
                    if col >= max_cols:
                        col = 0
                        row += 1
                        
        self.analyze_button.config(state="normal")
        self.pose_button.config(state="normal")
        
    def _create_athlete_thumbnail(self, frame_info):
        """選手のサムネイル画像を作成"""
        try:
            frame_path = frame_info.get('frame_path')
            bbox = frame_info.get('bbox')

            if not frame_path or bbox is None or len(bbox) != 4:
                return None

            # フレームを読み込み
            frame = cv2.imread(frame_path)
            if frame is None:
                return None

            # bbox の正規化: (x1,y1,x2,y2) または (x,y,w,h) に対応
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = bbox
            # detect if bbox is (x,y,w,h) by checking if x2,y2 look like widths/heights (小さめ・負・右端より小さい)
            if (x2 <= 0 or y2 <= 0) or (x2 < x1) or (y2 < y1) or (x2 <= w // 2 and y2 <= h // 2 and (x2 < 0 or y2 < 0 or (x2 < 100 and y2 < 100))):
                # treat as x,y,w,h
                x, y, ww, hh = bbox
                x1 = int(round(x))
                y1 = int(round(y))
                x2 = int(round(x + ww))
                y2 = int(round(y + hh))
            else:
                x1 = int(round(x1)); y1 = int(round(y1)); x2 = int(round(x2)); y2 = int(round(y2))

            # 画像境界で clamp
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x2 <= x1 or y2 <= y1:
                return None

            # 選手領域を抽出
            athlete_crop = frame[y1:y2, x1:x2]

            # RGBに変換してリサイズ
            athlete_crop = cv2.cvtColor(athlete_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(athlete_crop)

            # サムネイルサイズにリサイズ
            thumbnail_size = (120, 180)
            pil_image.thumbnail(thumbnail_size, Image.Resampling.LANCZOS)

            return pil_image

        except Exception as e:
            self.log_message(f"サムネイル作成エラー: {str(e)}")
            return None
            
    def analyze_jump(self):
        """跳躍分析を実行"""
        selected_id = self.athlete_combo.get()
        selected_id_str = str(selected_id)
        if not selected_id_str or selected_id_str not in self.detection_results:
            messagebox.showerror("エラー", "有効な選手IDを選択してください")
            return
            
        if self.processing:
            return
            
        self.processing = True
        self.analyze_button.config(state="disabled")
        
        # 別スレッドで分析処理を開始
        thread = threading.Thread(target=self._analysis_worker, args=(int(selected_id),))
        thread.daemon = True
        thread.start()
        
    def _analysis_worker(self, athlete_id):
        """跳躍分析のワーカースレッド"""
        try:
            video_path = self.input_video_path.get()
            athlete_frames = self.detection_results[str(athlete_id)]  
            results = self.jump_analyzer.analyze_jump(
                video_path,
                athlete_id,
                athlete_frames,
                progress_callback=self.update_progress,
                log_callback=self.log_message
            )
            if not results:
                self.log_message("跳躍分析結果が空です")
                messagebox.showerror("エラー", "跳躍分析に失敗しました")
                return
            self._display_analysis_results(results)
        except Exception as e:
            import traceback
            self.log_message(f"跳躍分析エラー: {str(e)}\n{traceback.format_exc()}")
            messagebox.showerror("エラー", f"跳躍分析に失敗しました\n{str(e)}")
        finally:
            self.processing = False
            self.analyze_button.config(state="normal")
            
    def _display_analysis_results(self, results):
        """分析結果を表示"""
        self.results_text.delete(1.0, tk.END)
        
        result_text = f"""
=== 走り幅跳び分析結果 ===

選手ID: {results.get('athlete_id', 'N/A')}
分析日時: {results.get('analysis_time', 'N/A')}

【跳躍距離】
記録: {results.get('jump_distance', 0):.2f} m

【フェーズ分析】
助走フェーズ: {results.get('approach_duration', 0):.2f} 秒
踏切フェーズ: {results.get('takeoff_duration', 0):.2f} 秒
飛行フェーズ: {results.get('flight_duration', 0):.2f} 秒
着地フェーズ: {results.get('landing_duration', 0):.2f} 秒

【速度分析】
助走最高速度: {results.get('max_approach_speed', 0):.2f} m/s
踏切時速度: {results.get('takeoff_speed', 0):.2f} m/s
踏切角度: {results.get('takeoff_angle', 0):.1f} 度

【技術分析】
踏切足: {results.get('takeoff_foot', 'N/A')}
最高到達点: {results.get('max_height', 0):.2f} m
着地角度: {results.get('landing_angle', 0):.1f} 度

【ポーズ分析】
踏切時姿勢スコア: {results.get('takeoff_pose_score', 0):.1f}/10
飛行時姿勢スコア: {results.get('flight_pose_score', 0):.1f}/10
着地時姿勢スコア: {results.get('landing_pose_score', 0):.1f}/10

【改善提案】
{results.get('improvement_suggestions', '分析データが不足しています')}

【詳細改善提案（フレームごと）】
{chr(10).join(results.get('detailed_suggestions', []))}
"""
        
        self.results_text.insert(tk.END, result_text)
        
    def generate_pose_video(self):
        """選択された選手のポーズ動画を生成"""
        if not self.output_video_path.get():
            messagebox.showerror("エラー", "出力動画パスを指定してください")
            return

        selected_id = self.athlete_combo.get()
        selected_id_str = str(selected_id)
        if not selected_id_str or selected_id_str not in self.detection_results:
            messagebox.showerror("エラー", "有効な選手IDを選択してください")
            return

        if self.processing:
            return

        self.processing = True
        self.pose_button.config(state="disabled")
        
        # スレッドでポーズ動画生成を開始
        thread = threading.Thread(target=self._pose_generation_worker, args=(selected_id_str,))  # athlete_id を str で渡す
        thread.daemon = True
        thread.start()
        
    def _pose_generation_worker(self, athlete_id):
        """ポーズ動画生成のワーカースレッド"""
        try:
            self.log_message(f"選手ID {athlete_id} のポーズ動画を生成しています...")
            self.update_progress(0, "ポーズ動画生成中...")

            selected_indices = [i for i, var in enumerate(self.keypoint_vars) if var.get()]

            # 踏切範囲の正規化: 0 は「未指定」として None を渡す
            start = self.takeoff_start_frame.get()
            end = self.takeoff_end_frame.get()
            start_norm = None if not isinstance(start, int) or start <= 0 else start
            end_norm = None if not isinstance(end, int) or end <= 0 else end

            # 事前チェック: 選択選手でキーポイントが実際に検出されるか簡易確認
            try:
                pose_model_path = getattr(self.config, "pose_model_path", "") or self.pose_model_path_var.get()
                if not pose_model_path or not os.path.exists(pose_model_path):
                    self.log_message("ポーズモデルが未設定/存在しません（事前チェックスキップ）")
                else:
                    # PoseEstimator を準備
                    self.pose_estimator.set_logger(self.log_message)
                    self.pose_estimator.load_model(
                        pose_model_path,
                        device=getattr(self.config, "device", "cpu"),
                        conf=0.18
                    )
                    frames_info = self.detection_results.get(athlete_id, [])  # athlete_id は str
                    if not frames_info:
                        self.log_message("事前チェック: 選手フレームが空です")
                    else:
                        import cv2
                        cap = cv2.VideoCapture(self.input_video_path.get())
                        # 最大8フレームを等間隔サンプリング
                        step = max(1, len(frames_info) // 8)
                        sample_idxs = list(range(0, len(frames_info), step))[:8]
                        detected = 0
                        total = len(sample_idxs)
                        for si in sample_idxs:
                            fi = frames_info[si]
                            img = None
                            fp = fi.get('frame_path')
                            if fp and os.path.exists(fp):
                                img = cv2.imread(fp)
                            else:
                                # フレーム番号があればCAPから取得
                                idx = fi.get('frame_index') or fi.get('frame') or fi.get('index')
                                if idx is not None:
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                                    ok, frame = cap.read()
                                    if ok:
                                        img = frame
                            if img is None:
                                continue
                            kps = self.pose_estimator.predict_keypoints(img, imgsz=960)
                            if kps is not None and kps.size > 0:
                                detected += 1
                        cap.release()
                        self.log_message(f"事前チェック: キーポイント検出 {detected}/{total} フレーム")
                        if detected == 0:
                            self.log_message("注意: この選手でキーポイントが検出されていません（設定/モデル要確認）")
            except Exception as e:
                self.log_message(f"事前チェックで例外: {e}")

            success = self.video_processor.generate_pose_video(
                self.input_video_path.get(),
                self.output_video_path.get(),
                athlete_id,  # athlete_id は str
                self.detection_results[athlete_id],  # athlete_id は str
                progress_callback=self.update_progress,
                log_callback=self.log_message,
                selected_indices=selected_indices,
                takeoff_start_frame=start_norm,
                takeoff_end_frame=end_norm,
                pose_model_path=pose_model_path,
                device=getattr(self.config, "device", "cpu"),
                # オプション: 推論のしきい値と画像サイズを下げ/上げる
                conf=0.1,  # 0.18から0.1に下げる（検出を緩く）
                imgsz=1280  # 960から1280に上げる（高解像度で推論）
            )
            
            if success:
                self.update_progress(100, "ポーズ動画生成完了")
                self.log_message("ポーズ動画が正常に生成されました！")
                messagebox.showinfo("成功", "ポーズ動画が正常に生成されました！")
            else:
                self.log_message("ポーズ動画の生成に失敗しました")
                messagebox.showerror("エラー", "ポーズ動画の生成に失敗しました")
                
        except Exception as e:
            self.log_message(f"ポーズ動画生成中にエラーが発生しました: {str(e)}")
            messagebox.showerror("エラー", f"ポーズ動画生成に失敗しました: {str(e)}")
        finally:
            self.processing = False
            self.pose_button.config(state="normal")
            
    def save_results_json(self):
        """分析結果をJSONファイルで保存"""
        if not self.jump_analysis_results:
            messagebox.showwarning("警告", "保存する分析結果がありません")
            return
            
        filename = filedialog.asksaveasfilename(
            title="分析結果を保存",
            defaultextension=".json",
            filetypes=[("JSONファイル", "*.json"), ("すべてのファイル", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(self.jump_analysis_results, f, ensure_ascii=False, indent=2)
                self.log_message(f"分析結果が保存されました: {filename}")
                messagebox.showinfo("成功", "分析結果が正常に保存されました")
            except Exception as e:
                self.log_message(f"保存エラー: {str(e)}")
                messagebox.showerror("エラー", f"保存に失敗しました: {str(e)}")
                
    def show_log_window(self):
        log_win = tk.Toplevel(self)
        log_win.title("ログ (Log)")
        log_text = tk.Text(log_win, height=20, width=100, wrap=tk.WORD)
        log_text.pack(fill=tk.BOTH, expand=True)
        log_text.insert(tk.END, self.log_text.get("1.0", tk.END))

    def set_pixel_to_meter_by_two_points(self):
        """基準距離を指定してピクセル/メートル比を設定"""
        import cv2
        from tkinter.simpledialog import askfloat

        video_path = self.input_video_path.get()
        if not video_path:
            messagebox.showerror("エラー", "入力動画ファイルを選択してください")
            return

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        points = []

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append((idx, (x, y)))  # フレーム番号と座標を記録

        window_name = "calibration_window"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, click_event)

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            temp_frame = frame.copy()
            # クリックした点を描画（フレーム番号が一致する点のみ）
            for pt_idx, (pt_frame, pt) in enumerate(points):
                if pt_frame == idx:
                    cv2.circle(temp_frame, pt, 6, (0, 0, 255), -1)
            cv2.putText(temp_frame, f"Frame: {idx+1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow(window_name, temp_frame)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('d') or key == 83:  # → or d
                idx = min(idx+1, total-1)
            elif key == ord('a') or key == 81:  # ← or a
                idx = max(idx-1, 0)
            elif key == 27:  # ESC
                points = []
                break
            # 2点クリックされたら終了
            if len(points) >= 2:
                break

        cv2.destroyAllWindows()
        cap.release()

        if len(points) < 2:
            return

        # 2点間のピクセル距離を計算（異なるフレームでもOK）
        (_, (x1, y1)), (_, (x2, y2)) = points
        pixel_dist = ((x2 - x1)**2 + (y2 - y1)**2) ** 0.5

        # 実距離を入力してもらう
        real_dist = askfloat("実距離入力", "2点間の実際の距離（メートル）を入力してください:")
        if not real_dist or real_dist <= 0:
            messagebox.showerror("エラー", "有効な距離を入力してください")
            return

        # ピクセル/メートル比を計算してセット
        ratio = pixel_dist / real_dist
        self.pixel_to_meter_ratio.set(ratio)
        self.log_message(f"基準距離指定: {pixel_dist:.1f} px / {real_dist:.2f} m → {ratio:.2f} px/m")

    def calibrate_ground_angle(self):
        """
        動画上で地面（踏切線など）の2点をクリックして地面角度をキャリブレーション
        """
        import cv2

        video_path = self.input_video_path.get()
        if not video_path:
            messagebox.showerror("エラー", "入力動画ファイルを選択してください")
            return

        self.withdraw()  # Tkinterウィンドウを隠す

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        idx = 0
        points = []

        window_name = "ground_calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)

        clicked = [False]

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
                points.append((x, y))
                clicked[0] = True

        cv2.setMouseCallback(window_name, click_event)

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break

            disp = frame.copy()
            cv2.putText(disp, f"Frame: {idx+1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            for pt in points:
                cv2.circle(disp, pt, 8, (0, 255, 255), -1)
            if len(points) == 2:
                cv2.line(disp, points[0], points[1], (0, 255, 255), 3)

            cv2.imshow(window_name, disp)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('d') or key == 83:
                idx = min(idx+1, total-1)
            elif key == ord('a') or key == 81:
                idx = max(idx-1, 0)
            elif key == ord('q') or key == 27:
                break

            if clicked[0]:
                clicked[0] = False
                continue

            if len(points) == 2:
                # 角度推定し、アプリ内と VideoProcessor の両方に反映
                ang = self.pose_estimator.estimate_ground_angle(points[0], points[1])
                self.video_processor.pose_estimator.set_ground_angle(ang)  # 同期
                # Configにも保持できるように
                setattr(self.config, "ground_angle_deg", ang)

                self.log_message(f"地面角度（度）: {ang:.2f}")
                messagebox.showinfo("キャリブレーション完了", f"地面角度: {ang:.2f}度で設定しました")
                break

        cap.release()
        cv2.destroyAllWindows()
        self.deiconify()

    def apply_settings(self):
        """設定を適用"""
        self.config.keypoint_interval = self.keypoint_interval.get()
        self.config.pose_draw_interval = int(self.pose_draw_interval.get())
        setattr(self.config, "pose_model_path", self.pose_model_path_var.get())
        # 残像設定を反映
        setattr(self.config, "show_trail", self.show_trail.get())
        # 任意: 残像長・濃さ（必要なら UI で変更できるように拡張可能）
        setattr(self.config, "trail_length", getattr(self.config, "trail_length", 15))
        setattr(self.config, "trail_alpha", getattr(self.config, "trail_alpha", 0.5))
        setattr(self.config, "trail_point_radius", getattr(self.config, "trail_point_radius", 4))
        self.log_message("設定が適用されました")

    def set_thread_count(self):
        """動画処理スレッド数を設定"""
        count = self.thread_count.get()
        if count < 1:
            messagebox.showerror("エラー", "スレッド数は1以上に設定してください")
            return
        self.config.thread_count = count
        self.log_message(f"動画処理スレッド数を{count}に設定しました")

    def load_default_settings(self):
        """デフォルト設定を読み込む"""
        self.keypoint_interval.set("1")
        self.pose_draw_interval.set("1")
        self.thread_count.set(4)
        self.show_trail.set(True)
        self.log_message("デフォルト設定を読み込みました")

    def reset_window_size(self):
        """ウィンドウサイズをリセット"""
        self.geometry("1400x900")
        self.log_message("ウィンドウサイズをリセットしました")

    def show_help(self):
        """ヘルプを表示"""
        help_message = """
走り幅跳び分析システム ヘルプ

このシステムは、走り幅跳びの選手を分析するためのツールです。
以下の機能があります。

- 動画入力: 選手の走り幅跳び動画を入力します。
- キャリブレーション設定: 踏切線や砂場開始位置を設定します。
- 処理: 選手の検出・追跡を行います。
- 選手選択: 検出された選手の中から分析対象を選択します。
- 分析結果表示: 跳躍分析の結果を表示します。
- ポーズ動画生成: 選手のポーズを示す動画を生成します。
- 設定: キャリブレーション間隔やスレッド数などを設定します。
- ログ: 処理のログを表示します。

ショートカットキー:
- スペース: 一時停止 / 再生
- c: 現在フレームをキャリブレーションポイントに設定
- d: 次のフレームにスキップ
- a: 前のフレームにスキップ
- q, Esc: 終了

不明な点があれば、ログウィンドウを確認するか、再度ヘルプを表示してください。
"""
        messagebox.showinfo("ヘルプ", help_message)
        
    def browse_pose_model(self):
        """YOLO Pose の .pt モデルを選択"""
        filename = filedialog.askopenfilename(
            title="ポーズモデルを選択",
            filetypes=[("YOLOモデル", "*.pt"), ("すべてのファイル", "*.*")]
        )
        if filename:
            self.pose_model_path_var.set(filename)
            self.config.pose_model_path = filename
            # VideoProcessor 側にも即時反映
            self.video_processor.pose_estimator.set_ground_angle(
                getattr(self.config, "ground_angle_deg", 0.0)
            )
            self.log_message(f"ポーズモデルを設定しました: {os.path.basename(filename)}")

    def preprocess_video(self, input_path, output_path, resolution=(640, 640), fps=30, crop=None, 
                     equalize_hist=False, use_clahe=False, denoise=False, brightness=0,
                     sharpen=False, gamma=1.0, dynamic_adjust=False, brightness_threshold=100):
        """
        動画を前処理して保存する。
    
        Args:
            input_path (str): 入力動画のパス。
            output_path (str): 前処理後の動画の保存先。
            resolution (tuple): (幅, 高さ) の解像度。
            fps (int): フレームレート。
            crop (tuple): (x, y, width, height) のクロップ領域（省略可）。
            equalize_hist (bool): ヒストグラム均衡化を適用するか。
            use_clahe (bool): CLAHEを適用するか。
            denoise (bool): ガウシアンフィルターでノイズ除去するか。
            brightness (int): 明るさ調整（-100 to 100）。
            sharpen (bool): シャープネス強調を適用するか。
            gamma (float): ガンマ補正値。
            dynamic_adjust (bool): フレームごとに動的に明るさを調整するか。
            brightness_threshold (int): 動的調整の輝度閾値（平均輝度がこれ未満の場合に調整）。
        """
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"動画を開けません: {input_path}")

        # 出力動画の設定
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, resolution)

        # CLAHE の準備
        if use_clahe:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        while True:
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                break  # フレームが無効ならスキップ

            # フレームが2次元以上か確認
            if frame.ndim < 2:
                continue  # 無効なフレームをスキップ

            # クロップ
            if crop:
                x, y, w, h = crop
                h_frame, w_frame = frame.shape[:2]
                x = max(0, min(x, w_frame - 1))
                y = max(0, min(y, h_frame - 1))
                w = max(1, min(w, w_frame - x))
                h = max(1, min(h, h_frame - y))
                frame = frame[y:y+h, x:x+w]
                if frame.size == 0:
                    continue  # クロップ後無効ならスキップ

            # リサイズ
            if frame.shape[:2] != resolution:
                frame = cv2.resize(frame, resolution)
                if frame.size == 0:
                    continue  # リサイズ後無効ならスキップ

            # 動的前処理: フレームごとの明るさチェックと調整
            if dynamic_adjust and frame.size > 0:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    if gray.size > 0:
                        mean_brightness = np.mean(gray)
                        if mean_brightness < brightness_threshold:
                            beta = max(0, brightness_threshold - mean_brightness)
                            frame = cv2.convertScaleAbs(frame, alpha=1, beta=int(beta))
                except cv2.error:
                    continue  # 変換失敗時はスキップ

            # 明るさ調整（静的）
            if brightness != 0 and frame.size > 0:
                frame = cv2.convertScaleAbs(frame, alpha=1, beta=brightness)

            # ノイズ除去（ガウシアンフィルター）
            if denoise and frame.size > 0:
                frame = cv2.GaussianBlur(frame, (3, 3), 0)

            # コントラスト調整
            if equalize_hist and frame.size > 0:
                try:
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                except cv2.error:
                    continue  # 変換失敗時はスキップ
            elif use_clahe and frame.size > 0:
                try:
                    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
                    frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
                except cv2.error:
                    continue  # 変換失敗時はスキップ

            # ガンマ補正（明るさ調整の代替）
            if gamma != 1.0 and frame.size > 0:
                try:
                    inv_gamma = 1.0 / gamma
                    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                    frame = cv2.LUT(frame, table)
                except cv2.error:
                    continue  # 変換失敗時はスキップ

            # シャープネス強調（エッジを保つ）
            if sharpen and frame.size > 0:
                try:
                    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    frame = cv2.filter2D(frame, -1, kernel)
                except cv2.error:
                    continue  # 変換失敗時はスキップ

            # 最終チェック: フレームが有効なら書き出し
            if frame.size > 0 and frame.ndim >= 2:
                out.write(frame)

        cap.release()
        out.release()
        print(f"前処理が完了しました: {output_path}")

    def start_preprocessing(self):
        """動画の前処理を開始"""
        input_path = self.input_video_path.get()
        if not input_path:
            messagebox.showerror("エラー", "入力動画を選択してください")
            return

        # 出力先を選択
        output_path = filedialog.asksaveasfilename(
            title="前処理後の動画を保存",
            defaultextension=".mp4",
            filetypes=[("MP4ファイル", "*.mp4"), ("すべてのファイル", "*.*")]
        )
        if not output_path:
            return

        # 前処理の設定（ここでオプションを調整）
        resolution = (640, 640)
        fps = 30
        crop = None
        equalize_hist = False
        use_clahe = False
        denoise = False
        brightness = 0
        sharpen = True      # シャープネスオン
        gamma = 1.2         # ガンマ補正（1.0より少し明るく）
        dynamic_adjust = True  # 動的前処理オン
        brightness_threshold = 100  # 閾値

        # 前処理を実行
        try:
            self.preprocess_video(input_path, output_path, resolution=resolution, fps=fps, crop=crop,
                          equalize_hist=equalize_hist, use_clahe=use_clahe, denoise=denoise, brightness=brightness,
                          sharpen=sharpen, gamma=gamma, dynamic_adjust=dynamic_adjust, brightness_threshold=brightness_threshold)
            messagebox.showinfo("成功", f"前処理が完了しました: {output_path}")
            self.log_message(f"前処理が完了しました: {output_path}")
        except Exception as e:
            messagebox.showerror("エラー", f"前処理に失敗しました: {e}")
            self.log_message(f"前処理に失敗しました: {e}")

    def _find_ffmpeg(self):
        """ffmpeg の実行ファイルパスを検出"""
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

def main():
    app = LongJumpApp()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ユーザーによって中断されました。")