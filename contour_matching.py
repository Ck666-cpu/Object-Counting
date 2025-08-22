import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


class TemplateMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Template Matching Tool")
        self.root.geometry("1200x800")

        self.image = None
        self.template = None
        self.result_image = None
        self.image_path = None

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
            scale_steps = np.linspace(0.5, 2.5, 30)  # 30 evenly spaced steps
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

        # Update count label in GUI
        self.count_label.config(text=f"Counted: {match_count}")

        self.result_image = result_img
        self.display_image(result_img)


if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateMatchingApp(root)
    root.mainloop()