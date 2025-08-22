import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import time
import io
import sys

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


    class Spacer:
        def __init__(self, *args, **kwargs): pass


    class ReportlabImage:
        def __init__(self, *args, **kwargs): pass


    class Table:
        def __init__(self, *args, **kwargs): pass

        def setStyle(self, *args): pass


    class TableStyle:
        def __init__(self, *args, **kwargs): pass


    class getSampleStyleSheet:
        def __init__(self, *args, **kwargs): pass

        def get(self, *args): return None


    colors = None


class TemplateMatchingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Template Matching Tool")
        self.root.geometry("1200x800")

        self.image = None
        self.template = None
        self.result_image = None
        self.image_path = None
        self.filtered_matches = []
        self.processing_time = 0.0
        self.report_path = None

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

        self.generate_report_button = ttk.Button(self.control_frame, text="Generate PDF Report",
                                                 command=self.generate_report)
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

        display_img = self.image.copy()
        h, w = display_img.shape[:2]

        MAX_DISPLAY_SIZE = 1000

        scale_factor = 1.0
        if w > MAX_DISPLAY_SIZE or h > MAX_DISPLAY_SIZE:
            scale_factor = min(MAX_DISPLAY_SIZE / w, MAX_DISPLAY_SIZE / h)
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
            cv2.waitKey(1)

    def perform_matching(self):
        if self.image is None or self.template is None:
            messagebox.showerror("Error", "Please upload an image and select a template first!")
            return

        start_time = time.time()

        result_img = self.image.copy()
        working_img = self.image.copy()
        working_template = self.template.copy()

        if self.color_threshold_var.get() and len(self.image.shape) == 3:
            hsv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
            hsv_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2HSV)
            mean_color = np.mean(hsv_template, axis=(0, 1))
            color_range = self.color_range_var.get()
            lower = np.array(
                [max(0, mean_color[0] - color_range), max(50, mean_color[1] - 50), max(50, mean_color[2] - 50)],
                dtype=np.uint8)
            upper = np.array([min(179, mean_color[0] + color_range), 255, 255], dtype=np.uint8)
            mask_img = cv2.inRange(hsv_img, lower, upper)
            mask_template = cv2.inRange(hsv_template, lower, upper)
            kernel = np.ones((3, 3), np.uint8)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_CLOSE, kernel)
            mask_template = cv2.morphologyEx(mask_template, cv2.MORPH_OPEN, kernel)
            mask_template = cv2.morphologyEx(mask_template, cv2.MORPH_CLOSE, kernel)
            if np.sum(mask_img) == 0 or np.sum(mask_template) == 0:
                messagebox.showwarning("Warning", "Color thresholding mask is empty! Try increasing the color range.")
                cv2.imshow("Debug: HSV Threshold Mask (Image)", mask_img)
                cv2.imshow("Debug: HSV Threshold Mask (Template)", mask_template)
                cv2.waitKey(1)
                return
            working_img = cv2.bitwise_and(working_img, working_img, mask=mask_img)
            working_template = cv2.bitwise_and(working_template, working_template, mask=mask_template)
            cv2.imshow("Debug: HSV Threshold Mask", mask_img)
            cv2.waitKey(1)

        if self.grayscale_var.get():
            working_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2GRAY)
            working_template = cv2.cvtColor(working_template, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            working_img = clahe.apply(working_img)
            working_template = clahe.apply(working_template)

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

        scales = [1.0]
        if self.scaling_var.get():
            scale_steps = np.linspace(0.5, 2.5, 30)
            scales = np.unique(scale_steps)

        angles = [0]
        if self.rotation_var.get():
            angles = np.linspace(0, 360, 24, endpoint=False)

        method = cv2.TM_CCOEFF_NORMED
        threshold = self.threshold_var.get()

        for scale in scales:
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))
            if scaled_w < 3 or scaled_h < 3:
                continue
            scaled_template = cv2.resize(working_template, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)

            for angle in angles:
                center = (scaled_w // 2, scaled_h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated_template = cv2.warpAffine(scaled_template, M, (scaled_w, scaled_h))

                if rotated_template.shape[0] < 3 or rotated_template.shape[1] < 3:
                    continue

                try:
                    result = cv2.matchTemplate(working_img, rotated_template, method)
                    locations = np.where(result >= threshold)

                    for pt in zip(*locations[::-1]):
                        matches.append((pt, scale, angle, result[pt[1], pt[0]]))
                except cv2.error:
                    continue

        filtered_matches = []
        matches = sorted(matches, key=lambda x: x[3], reverse=True)

        for pt, scale, angle, score in matches:
            scaled_w = max(1, int(tw * scale))
            scaled_h = max(1, int(th * scale))
            current_bbox = [pt[0], pt[1], pt[0] + scaled_w, pt[1] + scaled_h]
            overlap = False
            for _, _, _, _, existing_bbox in filtered_matches:
                x1 = max(current_bbox[0], existing_bbox[0])
                y1 = max(current_bbox[1], existing_bbox[1])
                x2 = min(current_bbox[2], existing_bbox[2])
                y2 = min(current_bbox[3], existing_bbox[3])
                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (current_bbox[2] - current_bbox[0]) * (current_bbox[3] - current_bbox[1])
                    area2 = (existing_bbox[2] - existing_bbox[0]) * (existing_bbox[3] - existing_bbox[1])
                    union = area1 + area2 - intersection
                    if union > 0 and intersection / union > 0.5:
                        overlap = True
                        break
            if not overlap:
                filtered_matches.append((pt, scale, angle, score, current_bbox))

        # --- NEW: Draw numbered labels on each bounding box ---
        for i, (pt, scale, angle, score, bbox) in enumerate(filtered_matches):
            try:
                scaled_w = max(1, int(tw * scale))
                scaled_h = max(1, int(th * scale))
                rect_corners = np.array([[0, 0], [scaled_w, 0], [scaled_w, scaled_h], [0, scaled_h]], dtype=np.float32)
                if angle != 0:
                    center_template = (scaled_w / 2, scaled_h / 2)
                    M_rect = cv2.getRotationMatrix2D(center_template, angle, 1.0)
                    rect_corners = cv2.transform(rect_corners[None, :, :], M_rect)[0]
                rotated_rect = rect_corners + np.array([pt[0], pt[1]])
                rotated_rect = rotated_rect.astype(np.int32)
                cv2.polylines(result_img, [rotated_rect], True, (0, 255, 0), 2)

                # Add a label (number) to the top-left of the bounding box
                label = f"{i + 1}"
                cv2.putText(result_img, label, (rotated_rect[0][0], rotated_rect[0][1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            except cv2.error as e:
                print(f"Error drawing rectangle or label: {e}")
                continue

        end_time = time.time()
        self.processing_time = end_time - start_time

        self.filtered_matches = filtered_matches

        self.count_label.config(text=f"Counted: {len(self.filtered_matches)}")
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
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('BOX', (0, 0), (-1, -1), 1, colors.black),
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


if __name__ == "__main__":
    root = tk.Tk()
    app = TemplateMatchingApp(root)
    root.mainloop()