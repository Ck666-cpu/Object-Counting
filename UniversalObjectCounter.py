import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time
import io
import sys
import subprocess
import threading
import math
from collections import deque


# Import ReportLab for PDF generation
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportlabImage, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
except ImportError:
    messagebox.showerror("Dependency Error", "The 'reportlab' library is not installed. "
                                           "Please install it using 'pip install reportlab' to enable PDF report generation.")
    # Create dummy classes to allow the GUI to run without the library
    class SimpleDocTemplate:
        def __init__(self, *args, **kwargs): pass
        def build(self, *args): pass
    class Paragraph:
        def __init__(self, *args, **kwargs): pass
        def setStyle(self, *args): pass
    class Spacer:
        def __init__(self, *args, **kwargs): pass
    class ReportlabImage:
        def __init__(self, *args, **kwargs): pass
    class Table:
        def __init__(self, *args, **kwargs): pass
        def setStyle(self, *args): pass
    class TableStyle:
        def __init__(self, *args, **kwargs): pass
    colors = None


class TemplateMatchingApp:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.template = None
        self.result_image = None
        self.image_path = None
        self.processing_time = 0.0
        self.report_path = None
        self.filtered_matches = []
        self.setup_gui()

    def setup_gui(self):
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        self.display_frame = ttk.Frame(self.root)
        self.display_frame.pack(side=tk.RIGHT, padx=2, pady=2, expand=True, fill=tk.BOTH)

        self.canvas = tk.Canvas(self.display_frame, width=500, height=500)
        self.canvas.pack(expand=True, fill=tk.BOTH)

        ttk.Button(self.control_frame, text="Upload Image", command=self.upload_image).pack(pady=5)
        ttk.Button(self.control_frame, text="Select Template", command=self.select_template).pack(pady=5)

        ttk.Label(self.control_frame, text="Processing Options:").pack(pady=5)

        self.grayscale_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Convert to Grayscale", variable=self.grayscale_var).pack()

        self.color_threshold_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Apply Color Thresholding", variable=self.color_threshold_var).pack()

        ttk.Label(self.control_frame, text="Color Threshold Range:").pack(pady=5)
        self.color_range_var = tk.DoubleVar(value=75.0)
        ttk.Scale(self.control_frame, from_=10.0, to=100.0, orient=tk.HORIZONTAL,
                  variable=self.color_range_var).pack()
        self.color_range_label = ttk.Label(self.control_frame, text="75.0")
        self.color_range_label.pack()
        self.color_range_var.trace('w', self.update_color_range_label)

        self.scaling_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Enable Template Scaling", variable=self.scaling_var).pack()

        self.rotation_var = tk.BooleanVar()
        ttk.Checkbutton(self.control_frame, text="Enable Template Rotation", variable=self.rotation_var).pack()

        ttk.Label(self.control_frame, text="Blur Method:").pack(pady=5)
        self.blur_var = tk.StringVar(value='None')
        ttk.Combobox(self.control_frame, textvariable=self.blur_var,
                     values=['None', 'Gaussian Blur', 'Median Blur', 'Bilateral Filter'], state='readonly').pack()

        ttk.Label(self.control_frame, text="Matching Threshold:").pack(pady=5)
        self.threshold_var = tk.DoubleVar(value=0.8)
        ttk.Scale(self.control_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL,
                  variable=self.threshold_var).pack()
        self.threshold_label = ttk.Label(self.control_frame, text="0.80")
        self.threshold_label.pack()
        self.threshold_var.trace('w', self.update_threshold_label)

        ttk.Button(self.control_frame, text="Search", command=self.perform_matching).pack(pady=20)

        self.count_label = ttk.Label(self.control_frame, text="Counted: 0")
        self.count_label.pack(pady=5)
        
        self.generate_report_button = ttk.Button(self.control_frame, text="Generate PDF Report", command=self.generate_report)
        self.generate_report_button.pack(pady=10)
        self.generate_report_button['state'] = 'disabled'

        self.open_report_button = ttk.Button(self.control_frame, text="Open Report", command=self.open_report)
        self.open_report_button.pack(pady=5)
        self.open_report_button['state'] = 'disabled'

    def update_threshold_label(self, *args):
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")

    def update_color_range_label(self, *args):
        self.color_range_label.config(text=f"{self.color_range_var.get():.1f}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image_path = file_path
            self.image = cv2.imread(file_path)
            self.display_image(self.image)
            self.generate_report_button['state'] = 'disabled'
            self.open_report_button['state'] = 'disabled'            

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        img_width, img_height = img_pil.size
        ratio = min(canvas_width / img_width, canvas_height / img_height, 1.0)
        new_size = (int(img_width * ratio), int(img_height * ratio))
        img_pil = img_pil.resize(new_size, Image.Resampling.LANCZOS)

        self.photo = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self.photo, anchor='center')

    def select_template(self):
        if self.image is None:
            messagebox.showerror("Error", "Please upload an image first!")
            return
        roi = cv2.selectROI("Select Template", self.image, False)
        cv2.destroyWindow("Select Template")
        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            self.template = self.image[y:y + h, x:x + w]
            cv2.imshow("Selected Template", self.template)
            cv2.waitKey(1)

    def perform_matching(self):
        if self.image is None or self.template is None:
            messagebox.showerror("Error", "Please upload an image and select a template first!")
            return
        
        start_time = time.time()

        result_img = self.image.copy()
        working_img = self.image.copy()
        working_template = self.template.copy()

        # Apply color thresholding (using HSV + morphological cleanup)
        if self.color_threshold_var.get() and len(self.image.shape) == 3:
            # Convert both image and template to HSV	
            hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)

            # Compute mean HSV color of the template
            mean_color = np.mean(hsv_template, axis=(0, 1))  # [H, S, V]
            color_range = self.color_range_var.get()

            # Define lower and upper bounds for HSV
            lower = np.array([
                max(0, mean_color[0] - color_range),  # Hue range
                max(50, mean_color[1] - 50),  # Saturation min (avoid gray/white)
                max(50, mean_color[2] - 50)  # Value min (avoid dark areas)
            ], dtype=np.uint8)

            upper = np.array([
                min(179, mean_color[0] + color_range),  # Hue max (HSV hue range is 0-179 in OpenCV)
                255,  # Max saturation
                255  # Max brightness
            ], dtype=np.uint8)

            # Threshold both image and template in HSV
            mask_img = cv2.inRange(hsv_img, lower, upper)
            mask_template = cv2.inRange(hsv_template, lower, upper)

            # Morphological cleanup (remove noise and fill holes)
            kernel = np.ones((3, 3), np.uint8)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)  # Remove small noise
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)  # Fill small holes
            mask_template = cv2.morphologyEx(mask_template, cv2.MORPH_OPEN, kernel)
            mask_template = cv2.morphologyEx(mask_template, cv2.MORPH_CLOSE, kernel)

            # Check if masks are empty
            if np.sum(mask_img) == 0 or np.sum(mask_template) == 0:
                messagebox.showwarning("Warning", "Color thresholding mask is empty! Try increasing the color range.")
                cv2.imshow("Debug: HSV Threshold Mask (Image)", mask_img)
                cv2.imshow("Debug: HSV Threshold Mask (Template)", mask_template)
                cv2.waitKey(1)
                return

            # Apply the masks to working images
            working_img = cv2.bitwise_and(working_img, working_img, mask=mask_img)
            working_template = cv2.bitwise_and(working_template, working_template, mask=mask_template)

            # Show the debug masks
            cv2.imshow("Debug: HSV Threshold Mask", mask_img)
            cv2.waitKey(1)

        # Apply grayscale and enhance contrast
        if self.grayscale_var.get():
            working_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
            working_template = cv2.cvtColor(working_template, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            working_img = clahe.apply(working_img)
            working_template = clahe.apply(working_template)

        # Apply blur
        blur_method = self.blur_var.get()
        if blur_method == 'Gaussian Blur':
            working_img = cv2.GaussianBlur(working_img, (5, 5), 0)
            working_template = cv2.GaussianBlur(working_template, (5, 5), 0)
        elif blur_method == 'Median Blur':
            working_img = cv2.medianBlur(working_img, 5)
            working_template = cv2.medianBlur(working_template, 5)
        elif blur_method == 'Bilateral Filter':
            working_img = cv2.bilateralFilter(working_img, 9, 75, 75)
            working_template = cv2.bilateralFilter(working_template, 9, 75, 75)

        matches = []
        th, tw = working_template.shape[:2]

        # Optimized scaling
        scales = [1.0]
        if self.scaling_var.get():
            # Generate scales from 50% to 200%
            scale_steps = np.linspace(0.5, 5.5, 50)  # 30 evenly spaced steps
            scales = np.unique(scale_steps)

        # Handle rotation
        angles = [0]
        if self.rotation_var.get():
            angles = np.linspace(0, 360, 24, endpoint=False)

        # Perform template matching
        method = cv2.TM_CCOEFF_NORMED
        threshold = self.threshold_var.get()

        for scale in scales:
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))
            # Consistent size validation
            if scaled_w < 3 or scaled_h < 3:
                continue
            scaled_template = cv2.resize(working_template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

            for angle in angles:
                center = (scaled_w // 2, scaled_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_template = cv2.warpAffine(scaled_template, M, (scaled_w, scaled_h))

                # Size validation after rotation (template might get smaller)
                if rotated_template.shape[0] < 3 or rotated_template.shape[1] < 3:
                    continue

                try:
                    result = cv2.matchTemplate(working_img, rotated_template, method)
                    locations = np.where(result >= threshold)

                    for pt in zip(*locations[::-1]):
                        matches.append((pt, scale, angle, result[pt[1], pt[0]]))
                except cv2.error:
                    continue

        # Improved Non-maximum suppression
        filtered_matches = []
        matches = sorted(matches, key=lambda x: x[3], reverse=True)

        for pt, scale, angle, score in matches:
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))

            # Calculate the bounding box for the current match
            current_bbox = [pt[0], pt[1], pt[0] + scaled_w, pt[1] + scaled_h]

            overlap = False
            for _, _, _, _, existing_bbox in filtered_matches:
                # Calculate intersection over union (IoU) for better overlap detection
                x1 = max(current_bbox[0], existing_bbox[0])
                y1 = max(current_bbox[1], existing_bbox[1])
                x2 = min(current_bbox[2], existing_bbox[2])
                y2 = min(current_bbox[3], existing_bbox[3])

                if x2 > x1 and y2 > y1:  # There is an intersection
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
                    area2 = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
                    union = area1 + area2 - intersection

                    # If IoU > 0.5, consider it an overlap
                    if intersection / union > 0.5:
                        overlap = True
                        break

            if not overlap:
                # Create proper rotated rectangle
                # Define rectangle corners relative to the match point
                rect_corners = np.array([
                    [0, 0],
                    [scaled_w, 0],
                    [scaled_w, scaled_h],
                    [0, scaled_h]
                ], dtype=np.float32)

                # Apply rotation around the template center
                if angle != 0:
                    center_template = (scaled_w / 2, scaled_h / 2)
                    M_rect = cv2.getRotationMatrix2D(center_template, angle, 1.0)
                    rect_corners = cv2.transform(rect_corners[None, :, :], M_rect)[0]

                # Translate to the match position
                rotated_rect = rect_corners + np.array([pt[0], pt[1]])
                rotated_rect = rotated_rect.astype(np.int32)

                filtered_matches.append((pt, scale, angle, score, current_bbox))

        # Draw rotated rectangles
        match_count = 0
        for i, (pt, scale, angle, score, bbox) in enumerate(filtered_matches):
            try:
                # Recreate the rotated rectangle for drawing
                scaled_w = max(1, int(tw * scale))
                scaled_h = max(1, int(th * scale))

                rect_corners = np.array([
                    [0, 0],
                    [scaled_w, 0],
                    [scaled_w, scaled_h],
                    [0, scaled_h]
                ], dtype=np.float32)

                if angle != 0:
                    center_template = (scaled_w / 2, scaled_h / 2)
                    M_rect = cv2.getRotationMatrix2D(center_template, angle, 1.0)
                    rect_corners = cv2.transform(rect_corners[None, :, :], M_rect)[0]

                rotated_rect = rect_corners + np.array([pt[0], pt[1]])
                rotated_rect = rotated_rect.astype(np.int32)

                cv2.polylines(result_img, [rotated_rect], True, (0, 255, 0), 2)
                match_count += 1
            except cv2.error as e:
                print(f"Error drawing rectangle: {e}")
                continue

        end_time = time.time()
        self.processing_time = end_time - start_time
        
        self.filtered_matches = filtered_matches
        # Update count label in GUI
        self.count_label.config(text=f"Counted: {match_count}")

        self.result_image = result_img
        self.display_image(result_img)
        self.generate_report_button['state'] = 'normal'
        self.open_report_button['state'] = 'disabled'
        self.report_path = None
        
    def generate_report(self):
        if self.result_image is None:
            messagebox.showerror("Error", "Please run a search first!")
            return

        self.report_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Template Matching Report"
        )

        if not self.report_path:
            return

        try:
            MAX_IMG_WIDTH = 450
            MAX_IMG_HEIGHT = 600

            doc = SimpleDocTemplate(self.report_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()

            # --- Summary ---
            story.append(Paragraph("Template Matching Report", styles['Title']))
            story.append(Spacer(1, 12))
            
            summary_text = (
                f"This report summarizes the results of the template matching process.<br/>"
                f"<b>Total Matches Found:</b> {len(self.filtered_matches)}<br/>"
                f"<b>Processing Time:</b> {self.processing_time:.2f} seconds"
            )
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 24))

            # --- Search Parameters Table ---
            story.append(Paragraph("Search Parameters", styles['Heading2']))
            data = [
                ['Parameter', 'Value'],
                ['Grayscale', 'Yes' if self.grayscale_var.get() else 'No'],
                ['Color Thresholding', 'Yes' if self.color_threshold_var.get() else 'No'],
                ['Color Range', f'{self.color_range_var.get():.1f}' if self.color_threshold_var.get() else 'N/A'],
                ['Scaling', 'Yes' if self.scaling_var.get() else 'No'],
                ['Rotation', 'Yes' if self.rotation_var.get() else 'No'],
                ['Blur Method', self.blur_var.get()],
                ['Matching Threshold', f'{self.threshold_var.get():.2f}'],
            ]
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ])
            param_table = Table(data)
            param_table.setStyle(table_style)
            story.append(param_table)
            story.append(Spacer(1, 24))

            # --- Full Result Image with Labels ---
            story.append(Paragraph("Image with Matches Labeled", styles['Heading2']))
            img_buffer_result = io.BytesIO()
            img_pil_result = Image.fromarray(cv2.cvtColor(self.result_image, cv2.COLOR_BGR2RGB))
            img_pil_result.save(img_buffer_result, format="PNG")
            img_buffer_result.seek(0)
            
            img_width, img_height = img_pil_result.size
            ratio = min(MAX_IMG_WIDTH / img_width, MAX_IMG_HEIGHT / img_height)
            new_width = img_width * ratio
            new_height = img_height * ratio
            report_image = ReportlabImage(img_buffer_result, width=new_width, height=new_height)
            story.append(report_image)
            story.append(Spacer(1, 24))

            # --- NEW: List of Matches and Scores ---
            story.append(Paragraph("Match Scores", styles['Heading2']))
            if not self.filtered_matches:
                story.append(Paragraph("No matches were found above the threshold.", styles['Normal']))
            else:
                score_list = []
                for i, (_, _, _, score, _) in enumerate(self.filtered_matches):
                    score_list.append([f"Match {i + 1}", f"{score:.4f}"])
                
                score_table = Table(score_list, colWidths=[100, 100])
                score_table.setStyle(TableStyle([
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('BOX', (0,0), (-1,-1), 1, colors.black),
                ]))
                story.append(score_table)
                story.append(Spacer(1, 12))


            doc.build(story)
            messagebox.showinfo("Report Generated", f"Report saved to:\n{self.report_path}")
            self.open_report_button['state'] = 'normal'
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate report: {e}")

    def open_report(self):
        """Opens the generated PDF file using the default application."""
        if not self.report_path or not os.path.exists(self.report_path):
            messagebox.showerror("Error", "No report file to open. Please generate a report first.")
            return

        try:
            if sys.platform == "win32":
                os.startfile(self.report_path)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", self.report_path])
            else:
                import subprocess
                subprocess.run(["xdg-open", self.report_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open the report file. Error: {e}")

class ObjectTracker:
    def __init__(self, max_missed_frames=10, max_distance=50, history_size=5):
        self.next_object_id = 0
        self.objects = {}
        self.max_missed_frames = max_missed_frames
        self.max_distance = max_distance
        self.history_size = history_size

    def update(self, detections):
        matched_detections = {}
        for detection in detections:
            min_dist = float('inf')
            best_match_id = -1
            det_center = detection['center']
            for obj_id, obj_data in self.objects.items():
                obj_center = obj_data['center']
                distance = np.sqrt((det_center[0] - obj_center[0])**2 + (det_center[1] - obj_center[1])**2)
                if distance < self.max_distance and distance < min_dist:
                    min_dist = distance
                    best_match_id = obj_id
            if best_match_id != -1:
                obj_data = self.objects[best_match_id]
                obj_data['center'] = det_center
                obj_data['missed_frames'] = 0
                obj_data['history'].append(det_center)
                obj_data['w'] = detection['bbox']['w']
                obj_data['h'] = detection['bbox']['h']
                matched_detections[best_match_id] = detection
            else:
                new_object_id = self.next_object_id
                self.objects[new_object_id] = {
                    'center': det_center,
                    'missed_frames': 0,
                    'history': deque([det_center], maxlen=self.history_size),
                    'counted': False,
                    'w': detection['bbox']['w'],
                    'h': detection['bbox']['h']
                }
                self.next_object_id += 1
                matched_detections[new_object_id] = detection

        for obj_id in list(self.objects.keys()):
            if obj_id not in matched_detections:
                self.objects[obj_id]['missed_frames'] += 1
        self.objects = {obj_id: obj_data for obj_id, obj_data in self.objects.items() if obj_data['missed_frames'] < self.max_missed_frames}
        
    def get_objects(self):
        for obj_id, obj_data in self.objects.items():
            if len(obj_data['history']) > 1:
                smooth_center = np.mean(list(obj_data['history']), axis=0)
                obj_data['smooth_center'] = (int(smooth_center[0]), int(smooth_center[1]))
            else:
                obj_data['smooth_center'] = obj_data['center']
        return self.objects

class VideoTemplateMatching:
    def __init__(self, root):
        self.root = root
        self.image = None
        self.template = None
        self.result_image = None
        self.image_path = None
        self.filtered_matches = []
        self.processing_time = 0.0
        self.report_path = None
        self.video_path = None
        self.counting_line_coords = None
        self.counted_objects = set()
        self.counted_label = None
        self.line_crossing_zone_width = 10
        self.object_tracker = ObjectTracker()
        self.is_video_counting = False
        self.is_processing = False
        self.video_metrics = {}
        self.setup_gui()
        self.set_button_state('initial')

    def setup_gui(self):
        # Main container frame to hold all other frames
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left Panel (Main Controls)
        self.left_panel_container = ttk.Frame(self.main_container, width=250)
        self.left_panel_container.pack(side=tk.LEFT, padx=5, fill=tk.Y)
        
        self.canvas_scroll = tk.Canvas(self.left_panel_container, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.left_panel_container, orient="vertical", command=self.canvas_scroll.yview)
        self.canvas_scroll.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas_scroll.pack(side="left", fill="both", expand=True)

        self.control_frame = ttk.Frame(self.canvas_scroll)
        self.canvas_scroll.create_window((0, 0), window=self.control_frame, anchor="nw", width=230)
        self.control_frame.bind("<Configure>", lambda e: self.canvas_scroll.configure(scrollregion=self.canvas_scroll.bbox("all")))
        
        # Central Panel (Image Display)
        self.display_frame = ttk.Frame(self.main_container)
        self.display_frame.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.BOTH)
        
        # Right Panel (Processing Options)
        self.right_panel_container = ttk.Frame(self.main_container, width=250)
        self.right_panel_container.pack(side=tk.RIGHT, padx=5, fill=tk.Y)

        self.options_canvas_scroll = tk.Canvas(self.right_panel_container, highlightthickness=0)
        self.options_scrollbar = ttk.Scrollbar(self.right_panel_container, orient="vertical", command=self.options_canvas_scroll.yview)
        self.options_canvas_scroll.configure(yscrollcommand=self.options_scrollbar.set)
        self.options_scrollbar.pack(side="right", fill="y")
        self.options_canvas_scroll.pack(side="left", fill="both", expand=True)

        self.options_frame = ttk.Frame(self.options_canvas_scroll)
        self.options_canvas_scroll.create_window((0, 0), window=self.options_frame, anchor="nw", width=230)
        self.options_frame.bind("<Configure>", lambda e: self.options_canvas_scroll.configure(scrollregion=self.options_canvas_scroll.bbox("all")))
        
        # Canvas for image display
        self.canvas = tk.Canvas(self.display_frame, bg="gray")
        self.canvas.pack(expand=True, fill=tk.BOTH)

        # --- Left Panel: Main Controls ---
        
        # File/Template Buttons
        ttk.Button(self.control_frame, text="Upload Video", command=self.upload_video).pack(pady=5, fill='x')
        ttk.Button(self.control_frame, text="Select Template", command=self.select_template).pack(pady=5, fill='x')
        
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Video Counting Section
        self.debug_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Enable Debug Windows", variable=self.debug_var).pack(anchor='w', pady=5)
        ttk.Label(self.control_frame, text="Video Counting:").pack(pady=5)
        self.define_line_button = ttk.Button(self.control_frame, text="Define Counting Line", command=self.define_counting_line)
        self.define_line_button.pack(pady=5, fill='x')
        self.video_count_button = ttk.Button(self.control_frame, text="Start Video Count", command=self.start_video_count_thread)
        self.video_count_button.pack(pady=5, fill='x')
        self.counted_label = ttk.Label(self.control_frame, text="Video Count: 0")
        self.counted_label.pack(pady=5, fill='x')
        self.progressbar = ttk.Progressbar(self.control_frame, orient='horizontal', length=200, mode='determinate')
        self.progressbar.pack(pady=5, fill='x')
        
        ttk.Separator(self.control_frame, orient='horizontal').pack(fill='x', pady=10)

        # Main Search and Report Buttons
        self.generate_report_button = ttk.Button(self.control_frame, text="Generate PDF Report", command=self.generate_report)
        self.generate_report_button.pack(pady=10, fill='x')
        self.open_report_button = ttk.Button(self.control_frame, text="Open Report", command=self.open_report)
        self.open_report_button.pack(pady=5, fill='x')

        # --- Right Panel: Processing Options ---
        ttk.Label(self.options_frame, text="Processing Options:", font=("TkDefaultFont", 12, "bold")).pack(pady=5)
        
        # Grayscale
        self.grayscale_var = tk.BooleanVar(value=True) 
        ttk.Checkbutton(self.options_frame, text="Convert to Grayscale", variable=self.grayscale_var).pack(anchor='w', pady=2)
        
        # Color Filtering
        self.color_threshold_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.options_frame, text="Apply Color Filtering", variable=self.color_threshold_var).pack(anchor='w', pady=2)
        ttk.Label(self.options_frame, text="Color Filter Range:").pack(pady=2)
        self.color_range_var = tk.DoubleVar(value=20.0)
        ttk.Scale(self.options_frame, from_=0.0, to=180.0, orient=tk.HORIZONTAL, variable=self.color_range_var).pack(fill='x')
        self.color_range_label = ttk.Label(self.options_frame, text="20.0")
        self.color_range_label.pack()
        self.color_range_var.trace('w', self.update_color_range_label)
        
        # Scaling and Rotation
        self.scaling_var = tk.BooleanVar()
        ttk.Checkbutton(self.options_frame, text="Enable Template Scaling", variable=self.scaling_var).pack(anchor='w', pady=8)
        self.rotation_var = tk.BooleanVar()
        ttk.Checkbutton(self.options_frame, text="Enable Template Rotation", variable=self.rotation_var).pack(anchor='w', pady=8)
        
        # Matching Threshold
        ttk.Label(self.options_frame, text="Matching Threshold:").pack(pady=2)
        self.threshold_var = tk.DoubleVar(value=0.8)
        ttk.Scale(self.options_frame, from_=0.1, to=1.0, orient=tk.HORIZONTAL, variable=self.threshold_var).pack(fill='x')
        self.threshold_label = ttk.Label(self.options_frame, text="0.80")
        self.threshold_label.pack()
        self.threshold_var.trace('w', self.update_threshold_label)

    def set_button_state(self, mode):
        if mode == 'initial':
            self.define_line_button['state'] = 'disabled'
            self.video_count_button['state'] = 'disabled'
            self.generate_report_button['state'] = 'disabled'
            self.open_report_button['state'] = 'disabled'
        elif mode == 'video_loaded':
            self.define_line_button['state'] = 'normal'
            self.video_count_button['state'] = 'normal'
            self.generate_report_button['state'] = 'disabled'
            self.open_report_button['state'] = 'disabled'
        elif mode == 'processing':
            self.define_line_button['state'] = 'disabled'
            self.video_count_button['state'] = 'disabled'
            self.generate_report_button['state'] = 'disabled'
            self.open_report_button['state'] = 'disabled'
        elif mode == 'video_processed':
            self.video_count_button['state'] = 'normal'
            self.define_line_button['state'] = 'normal'
            self.generate_report_button['state'] = 'normal'
            self.open_report_button['state'] = 'normal'

    def update_threshold_label(self, *args):
        self.threshold_label.config(text=f"{self.threshold_var.get():.2f}")

    def update_color_range_label(self, *args):
        self.color_range_label.config(text=f"{self.color_range_var.get():.1f}")
        
    def display_image(self, img):
        if img is None:
            self.canvas.delete("all")
            return
        h, w = img.shape[:2]
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if w > canvas_w or h > canvas_h:
            scale = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(image=img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(self.canvas.winfo_width() / 2, self.canvas.winfo_height() / 2,
                                 anchor='center', image=self.tk_image)
        self.canvas.image = self.tk_image

    def upload_video(self):
        # Clear previous state
        self.image = None
        self.template = None
        self.result_image = None
        self.image_path = None
        self.video_path = None
        
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            cap = cv2.VideoCapture(self.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.image = frame
                self.display_image(self.image)
                self.set_button_state('video_loaded')
                self.counted_label.config(text="Video Count: 0")
                messagebox.showinfo("Video Loaded", "Video loaded. Displaying the first frame on the canvas. Please select a template and define a counting line.")
            else:
                messagebox.showerror("Error", "Could not read the first frame of the video. The file may be corrupted.")
                self.display_image(None)
                self.set_button_state('initial')

    def select_template(self):
        if self.image is None:
            messagebox.showerror("Error", "Please upload an image or video first!")
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        display_img = self.image.copy()
        h, w = display_img.shape[:2]
        scale_factor = 1.0
        if w > canvas_w or h > canvas_h:
            scale_factor = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        roi = cv2.selectROI("Select Template", display_img, False)
        cv2.destroyWindow("Select Template")
        if roi != (0, 0, 0, 0):
            x, y, w, h = roi
            x_orig = int(x / scale_factor)
            y_orig = int(y / scale_factor)
            w_orig = int(w / scale_factor)
            h_orig = int(h / scale_factor)
            self.template = self.image[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]
            cv2.imshow("Selected Template", self.template)
            cv2.waitKey(0)
            cv2.destroyWindow("Selected Template")

    def define_counting_line(self):
        if self.image is None:
            messagebox.showerror("Error", "Please upload a video first!")
            return
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        display_img = self.image.copy()
        h, w = display_img.shape[:2]
        scale_factor = 1.0
        if w > canvas_w or h > canvas_h:
            scale_factor = min(canvas_w / w, canvas_h / h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            display_img = cv2.resize(display_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        messagebox.showinfo("Define Counting Line", "Click and drag to draw a line on the image.")
        def get_line_coords(event, x, y, flags, param):
            nonlocal p1, p2
            if event == cv2.EVENT_LBUTTONDOWN:
                p1 = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                p2 = (x, y)
                cv2.destroyWindow("Define Counting Line")
        p1, p2 = None, None
        cv2.namedWindow("Define Counting Line")
        cv2.setMouseCallback("Define Counting Line", get_line_coords)
        temp_img = display_img.copy()
        while True:
            cv2.imshow("Define Counting Line", temp_img)
            k = cv2.waitKey(1) & 0xFF
            if k == 27 or (p1 and p2):
                break
        if p1 and p2:
            x1_orig = int(p1[0] / scale_factor)
            y1_orig = int(p1[1] / scale_factor)
            x2_orig = int(p2[0] / scale_factor)
            y2_orig = int(p2[1] / scale_factor)
            self.counting_line_coords = ((x1_orig, y1_orig), (x2_orig, y2_orig))
            cv2.line(display_img, p1, p2, (0, 0, 255), 2)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            if length > 0:
                nx = -dy / length
                ny = dx / length
            else:
                nx, ny = 0, 0
            line_a_p1 = (int(p1[0] + nx * self.line_crossing_zone_width), int(p1[1] + ny * self.line_crossing_zone_width))
            line_a_p2 = (int(p2[0] + nx * self.line_crossing_zone_width), int(p2[1] + ny * self.line_crossing_zone_width))
            line_b_p1 = (int(p1[0] - nx * self.line_crossing_zone_width), int(p1[1] - ny * self.line_crossing_zone_width))
            line_b_p2 = (int(p2[0] - nx * self.line_crossing_zone_width), int(p2[1] - ny * self.line_crossing_zone_width))
            cv2.line(display_img, line_a_p1, line_a_p2, (0, 255, 0), 1)
            cv2.line(display_img, line_b_p1, line_b_p2, (0, 255, 0), 1)
            cv2.putText(display_img, "Counting Line", (p1[0], p1[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Counting Line Defined", display_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Counting Line Defined")
            messagebox.showinfo("Success", "Counting line defined successfully!")
    
    def start_video_count_thread(self):
        if self.is_processing:
            messagebox.showwarning("Busy", "Already processing. Please wait.")
            return
        if not self.video_path or self.template is None or self.counting_line_coords is None:
            messagebox.showerror("Error", "Please upload a video, select a template, and define a counting line first!")
            return
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        estimated_time = self.estimate_processing_time(total_frames)
        if estimated_time > 120:
            warning_message = f"Warning: Video processing is estimated to take a long time.\n\n"
            warning_message += f"Estimated duration: {estimated_time // 60} minutes and {estimated_time % 60} seconds.\n\n"
            warning_message += "This is due to the following settings:"
            if self.scaling_var.get():
                warning_message += "\n- Template Scaling"
            if self.rotation_var.get():
                warning_message += "\n- Template Rotation"
            if self.color_threshold_var.get():
                warning_message += "\n- Color Filtering"
            warning_message += "\n\nDo you want to continue with these settings?"
            response = messagebox.askyesno("Long Processing Time Warning", warning_message)
            if response:
                self.start_video_count_worker()
        else:
            self.start_video_count_worker()
            
    def estimate_processing_time(self, total_frames):
        base_factor = 0.05
        color_factor = 1.2 if self.color_threshold_var.get() else 1
        total_factor = base_factor * color_factor 
        estimated_time = total_frames * total_factor
        return int(estimated_time)
    
    def start_video_count_worker(self):
        self.set_button_state('processing')
        self.is_processing = True
        threading.Thread(target=self.count_in_video, daemon=True).start()
        
    def count_in_video(self):
        # A flag to track if processing completed successfully
        success = False
        start_time = time.time()
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video file."))
            self.root.after(0, lambda: self.set_button_state('video_loaded'))
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_metrics = {
            'total_frames': total_frames,
            'original_fps': fps,
            'resolution': (width, height),
            'start_time': start_time,
            'total_count': 0
        }
        output_filename = "processed_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        self.root.after(0, lambda: self.progressbar.config(maximum=total_frames))
        self.root.after(0, lambda: self.progressbar.config(value=0))
        self.root.after(0, lambda: self.counted_label.config(text="Video Count: 0"))
        total_count = 0
        self.object_tracker = ObjectTracker()
        p1, p2 = self.counting_line_coords
        def get_line_position(point, p1, p2):
            return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])
        self.root.after(0, lambda: messagebox.showinfo("Processing", "Starting video processing. The processed video will open automatically upon completion."))
        last_frame = None
        try:
            preprocessed_templates = self.prepare_templates()
            if not preprocessed_templates:
                self.root.after(0, lambda: messagebox.showerror("Error", "Template preprocessing failed. Check your selections."))
                self.root.after(0, lambda: self.set_button_state('video_loaded'))
                return
            while True:
                ret, frame = cap.read()
                if not ret:
                    # The loop broke because the video ended naturally
                    success = True # Set the success flag here
                    break
                last_frame = frame.copy()
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.root.after(0, lambda: self.progressbar.config(value=frame_number))
                self.root.update_idletasks()
                working_frame = self.apply_processing_to_image(frame)
                if working_frame is None:
                    # We hit a fatal error (like empty mask) in preprocessing.
                    # Stop the loop and let the finally block handle cleanup.
                    self.root.after(0, lambda: messagebox.showerror("Error", "Image preprocessing failed. Please check your color filter and try again."))
                    break # CRITICAL CHANGE
                all_matches = []
                for template_data in preprocessed_templates:
                    rotated_template = template_data['image']
                    tw, th = rotated_template.shape[:2]
                    if working_frame.shape[0] < th or working_frame.shape[1] < tw:
                         continue
                    if working_frame.ndim != rotated_template.ndim:
                        if working_frame.ndim == 3 and rotated_template.ndim == 2:
                             working_frame_match = cv2.cvtColor(working_frame, cv2.COLOR_BGR2GRAY)
                        elif working_frame.ndim == 2 and rotated_template.ndim == 3:
                             working_frame_match = cv2.cvtColor(working_frame, cv2.COLOR_GRAY2BGR)
                        else:
                             continue
                    else:
                        working_frame_match = working_frame
                    res = cv2.matchTemplate(working_frame_match, rotated_template, cv2.TM_CCOEFF_NORMED)
                    if self.debug_var.get():
                        self.root.after(0, self.display_debug_images, working_frame_match, rotated_template, res)
                    loc = np.where(res >= self.threshold_var.get())
                    for pt in zip(*loc[::-1]):
                        all_matches.append({
                            'pt': pt,
                            'w': tw,
                            'h': th,
                            'score': res[pt[1], pt[0]]
                        })
                filtered_matches = self.non_max_suppression(all_matches, 0.5)
                current_frame_detections = [{'center': (m['pt'][0] + m['w'] // 2, m['pt'][1] + m['h'] // 2), 'bbox': m} for m in filtered_matches]
                self.object_tracker.update(current_frame_detections)
                tracked_objects = self.object_tracker.get_objects()
                for obj_id, obj_data in tracked_objects.items():
                    if len(obj_data['history']) < 2:
                        continue
                    if not obj_data['counted']:
                        last_center = obj_data['history'][-2]
                        current_center = obj_data['history'][-1]
                        pos_last = get_line_position(last_center, p1, p2)
                        pos_current = get_line_position(current_center, p1, p2)
                        if pos_last * pos_current < 0:
                            total_count += 1
                            obj_data['counted'] = True
                            self.root.after(0, lambda: self.counted_label.config(text=f"Video Count: {total_count}"))
                p1_draw = p1
                p2_draw = p2
                cv2.line(frame, p1_draw, p2_draw, (0, 0, 255), 2)
                dx = p2_draw[0] - p1_draw[0]
                dy = p2_draw[1] - p1_draw[1]
                length = math.sqrt(dx**2 + dy**2)
                if length > 0:
                    nx = -dy / length
                    ny = dx / length
                else:
                    nx, ny = 0, 0
                line_a_p1 = (int(p1_draw[0] + nx * self.line_crossing_zone_width), int(p1[1] + ny * self.line_crossing_zone_width))
                line_a_p2 = (int(p2_draw[0] + nx * self.line_crossing_zone_width), int(p2[1] + ny * self.line_crossing_zone_width))
                line_b_p1 = (int(p1_draw[0] - nx * self.line_crossing_zone_width), int(p1[1] - ny * self.line_crossing_zone_width))
                line_b_p2 = (int(p2_draw[0] - nx * self.line_crossing_zone_width), int(p2[1] - ny * self.line_crossing_zone_width))
                cv2.line(frame, line_a_p1, line_a_p2, (0, 255, 0), 1)
                cv2.line(frame, line_b_p1, line_b_p2, (0, 255, 0), 1)
                for obj_id, obj_data in tracked_objects.items():
                    smooth_center = obj_data['smooth_center']
                    w = obj_data['w']
                    h = obj_data['h']
                    x1 = smooth_center[0] - w // 2
                    y1 = smooth_center[1] - h // 2
                    x2 = smooth_center[0] + w // 2
                    y2 = smooth_center[1] + h // 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_color = (0, 255, 0)
                    if obj_data['counted']:
                        label_color = (0, 0, 255)
                    label = f"ID: {obj_id}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)
                cv2.putText(frame, f"Total Count: {total_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                out.write(frame)
        finally:
            cap.release()
            out.release()
            self.is_processing = False
            # Now, use the flag to determine which final actions to take
            if success:
                end_time = time.time()
                self.video_metrics['total_count'] = total_count
                self.video_metrics['processed_time'] = end_time - start_time
                self.video_metrics['processed_fps'] = total_frames / self.video_metrics['processed_time'] if self.video_metrics['processed_time'] > 0 else 0
                self.video_metrics['final_frame'] = last_frame
                
                # Show success messages and update GUI
                self.root.after(0, lambda: self.progressbar.config(value=0))
                self.root.after(0, lambda: self.set_button_state('video_processed'))
                self.root.after(0, lambda: messagebox.showinfo("Video Counting Complete", f"Total objects counted: {total_count}"))
                self.root.after(0, self.generate_report)
                
                # Optional: Open the output file
                # ... (your code to open the output video file)
                try:
                    if sys.platform == "win32":
                        os.startfile(output_filename)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", output_filename])
                    else:
                        subprocess.run(["xdg-open", output_filename])
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showwarning("Open Video Failed", f"Could not open the processed video file. You can find it at: {output_filename}"))
            
            else:
                # Show error/aborted messages and reset GUI state
                self.root.after(0, lambda: self.progressbar.config(value=0))
                self.root.after(0, lambda: self.set_button_state('video_loaded'))
                self.root.after(0, lambda: messagebox.showinfo("Processing Aborted", "Video processing was aborted due to an error."))
           
                
    def apply_processing_to_image(self, img):
        if img is None:
            return None
        
        working_img = img.copy()
        
        # Apply color thresholding (using HSV + morphological cleanup)
        if self.color_threshold_var.get() and self.template is not None:
            # Convert both image and template to HSV
            hsv_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
            hsv_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)

            # Compute mean HSV color of the template
            mean_color = np.mean(hsv_template, axis=(0, 1))  # [H, S, V]
            color_range = self.color_range_var.get()

            # Define lower and upper bounds for HSV
            lower_bound = np.array([
                max(0, mean_color[0] - color_range),  # Hue range
                max(50, mean_color[1] - 50),  # Saturation min (avoid gray/white)
                max(50, mean_color[2] - 50)  # Value min (avoid dark areas)
            ], dtype=np.uint8)

            upper_bound = np.array([
                min(179, mean_color[0] + color_range),  # Hue max (HSV hue range is 0-179 in OpenCV)
                255,  # Max saturation
                255  # Max brightness
            ], dtype=np.uint8)

            # Threshold the image in HSV
            mask = cv2.inRange(hsv_img, lower_bound, upper_bound)
            
            # Morphological cleanup (remove noise and fill holes)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)  # Remove small noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1) # Fill small holes

            # Check if mask is empty and provide feedback
            if np.sum(mask) == 0:
                # Instead of returning a copy, we indicate a critical failure.
                # Returning None here will cause a failure in the main processing loop.
                return None # CRITICAL CHANGE
            
            working_img = cv2.bitwise_and(working_img, working_img, mask=mask)

        # Grayscale conversion
        # Note: This step is now more important after color thresholding.
        # The image is now an ANDed result, so converting to grayscale makes sense for subsequent steps.
        if self.grayscale_var.get() and len(working_img.shape) == 3:
            working_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            working_img = clahe.apply(working_img)
            
        return working_img
    
    
    def prepare_templates(self):
        templates = []
        if self.template is None or self.template.size == 0:
            return []
        base_template = self.apply_processing_to_image(self.template)
        if base_template is None or base_template.size == 0:
            return []
        if len(base_template.shape) == 3:
            tw, th = base_template.shape[1], base_template.shape[0]
        else:
            tw, th = base_template.shape[1], base_template.shape[0]
        scales = [1.0]
        if self.scaling_var.get():
            scales = np.linspace(0.5, 2.5, 10)
        angles = [0]
        if self.rotation_var.get():
            angles = np.linspace(0, 360, 24, endpoint=False)
        for scale in scales:
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))
            if scaled_w < 3 or scaled_h < 3: continue
            scaled_template = cv2.resize(base_template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
            for angle in angles:
                center = (scaled_w // 2, scaled_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_template = cv2.warpAffine(scaled_template, M, (scaled_w, scaled_h))
                if rotated_template.shape[0] < 3 or rotated_template.shape[1] < 3: continue
                templates.append({'image': rotated_template, 'size': (rotated_template.shape[1], rotated_template.shape[0])})
        return templates
    
    
    def non_max_suppression(self, boxes, overlapThresh):
        if len(boxes) == 0:
            return []
        if boxes[0]['pt'] is None:
            return []
        pick = []
        scores = np.array([b['score'] for b in boxes])
        boxes_np = np.array([[b['pt'][0], b['pt'][1], b['pt'][0] + b['w'], b['pt'][1] + b['h']] for b in boxes])
        x1 = boxes_np[:, 0]
        y1 = boxes_np[:, 1]
        x2 = boxes_np[:, 2]
        y2 = boxes_np[:, 3]
        idxs = np.argsort(scores)[::-1]
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / ((x2[i] - x1[i] + 1) * (y2[i] - y1[i] + 1))
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        return [boxes[j] for j in pick]
    
    def display_debug_images(self, working_img_match, template, res):
        try:
            if len(working_img_match.shape) == 2:
                img_for_display = cv2.cvtColor(working_img_match, cv2.COLOR_GRAY2BGR)
                template_for_display = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
            else:
                img_for_display = working_img_match.copy()
                template_for_display = template.copy()
            res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX)
            res_display = cv2.convertScaleAbs(res_norm)
            img_h, img_w = img_for_display.shape[:2]
            temp_h, temp_w = template_for_display.shape[:2]
            res_h, res_w = res_display.shape[:2]
            scale_factor = 0.5
            img_for_display = cv2.resize(img_for_display, (int(img_w * scale_factor), int(img_h * scale_factor)))
            template_for_display = cv2.resize(template_for_display, (int(temp_w * scale_factor), int(temp_h * scale_factor)))
            res_display = cv2.resize(res_display, (int(res_w * scale_factor), int(res_h * scale_factor)))
            cv2.imshow("Debug - Main Image (Pre-processed)", img_for_display)
            cv2.imshow("Debug - Template (Pre-processed)", template_for_display)
            cv2.imshow("Debug - Result Matrix", res_display)
            cv2.waitKey(1)
        except Exception as e:
            messagebox.showerror("Debug Display Error", f"Failed to display debug windows: {e}")
            self.debug_var.set(False)
            cv2.destroyAllWindows()
    
    
    def generate_report(self):
        if self.video_path:
            self.generate_video_report_handler()
            
    def generate_video_report_handler(self):
        if not self.video_metrics:
            messagebox.showerror("Error", "Please run video processing first!")
            return
        self.report_path = filedialog.asksaveasfilename(
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf")],
            title="Save Video Counting Report"
        )
        if not self.report_path:
            return
        try:
            doc = SimpleDocTemplate(self.report_path, pagesize=letter)
            story = []
            styles = getSampleStyleSheet()
            story.append(Paragraph("Video Counting Report", styles['Title']))
            story.append(Spacer(1, 12))
            summary_text = (
                f"<b>Total Objects Counted:</b> {self.video_metrics['total_count']}<br/>"
                f"<b>Video Duration:</b> {self.video_metrics['processed_time']:.2f} seconds<br/>"
                f"<b>Original FPS:</b> {self.video_metrics['original_fps']:.2f}<br/>"
                f"<b>Processed FPS:</b> {self.video_metrics['processed_fps']:.2f}<br/>"
                f"<b>Video Resolution:</b> {self.video_metrics['resolution'][0]}x{self.video_metrics['resolution'][1]}"
            )
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 24))
            story.append(Paragraph("Counting Line Coordinates", styles['Heading2']))
            p1, p2 = self.counting_line_coords
            story.append(Paragraph(f"Start Point: ({p1[0]}, {p1[1]})<br/>End Point: ({p2[0]}, {p2[1]})", styles['Normal']))
            story.append(Spacer(1, 24))
            story.append(Paragraph("Search Parameters", styles['Heading2']))
            data = [
                ['Parameter', 'Value'],
                ['Grayscale', 'Yes' if self.grayscale_var.get() else 'No'],
                ['Color Filtering', 'Yes' if self.color_threshold_var.get() else 'No'],
                ['Color Filter Range', f'{self.color_range_var.get():.1f}' if self.color_threshold_var.get() else 'N/A'],
                ['Scaling', 'Yes' if self.scaling_var.get() else 'No'],
                ['Rotation', 'Yes' if self.rotation_var.get() else 'No'],
                ['Matching Threshold', f'{self.threshold_var.get():.2f}'],
            ]
            table_style = TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BOX', (0, 0), (-1, -1), 1, colors.black),
            ])
            param_table = Table(data)
            param_table.setStyle(table_style)
            story.append(param_table)
            story.append(Spacer(1, 24))
            story.append(Paragraph("Final Frame with Detections", styles['Heading2']))
            if self.video_metrics['final_frame'] is not None:
                img_buffer = io.BytesIO()
                img_pil = Image.fromarray(cv2.cvtColor(self.video_metrics['final_frame'], cv2.COLOR_BGR2RGB))
                img_width, img_height = img_pil.size
                ratio = min(500 / img_width, 500 / img_height)
                new_width = img_width * ratio
                new_height = img_height * ratio
                img_pil = img_pil.resize((int(new_width), int(new_height)), Image.LANCZOS)
                img_pil.save(img_buffer, format="PNG")
                img_buffer.seek(0)
                story.append(ReportlabImage(img_buffer, width=new_width, height=new_height))
            doc.build(story)
            messagebox.showinfo("Report Generated", f"Video report saved to:\n{self.report_path}")
            self.open_report_button['state'] = 'normal'
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate video report: {e}")
            
    def open_report(self):
        if not self.report_path or not os.path.exists(self.report_path):
            messagebox.showerror("Error", "No report file to open. Please generate a report first.")
            return
        try:
            if sys.platform == "win32":
                os.startfile(self.report_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", self.report_path])
            else:
                subprocess.run(["xdg-open", self.report_path])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open the report file. Error: {e}")
            

# main tkinter applications run here lo
class TabbedApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Template Matching App")
        self.root.geometry("1200x600")
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Create the frames for each tab
        tab1_frame = ttk.Frame(self.notebook)
        tab2_frame = ttk.Frame(self.notebook)
        
        # Add the frames to the notebook
        self.notebook.add(tab1_frame, text="Videos Template Matching")
        self.notebook.add(tab2_frame, text="Images Template Matching")

        # Instantiate your separate app classes, passing in the correct frame
        self.app1 = VideoTemplateMatching(tab1_frame)
        self.app2 = TemplateMatchingApp(tab2_frame) # Assuming MyAppForTab2 exists


if __name__ == '__main__':
    root = tk.Tk()
    app = TabbedApp(root)
    root.mainloop()