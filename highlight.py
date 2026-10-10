import os
import cv2
import easyocr
import numpy as np


class CounterfactualHighlighter:
    """Generates visual audit highlight overlays on both original and edited counterfactual images."""

    def __init__(self):
        # Initialize EasyOCR reader for precise bounding box detection
        self.reader = easyocr.Reader(["en"], gpu=True)

    def generate_highlight(
        self,
        img_path: str,
        target_text: str,
        output_path: str,
        highlight_color: tuple = (0, 255, 255),  # Yellow overlay (BGR)
        border_color: tuple = (0, 0, 255),  # Red border stroke (BGR)
        alpha: float = 0.35,  # Transparency
        padding: int = 4,  # Extra spatial padding (pixels) around detected box
    ):
        """Locates target_text in img_path and saves a highlighted version with padding."""
        if not os.path.exists(img_path):
            print(f"[HIGHLIGHT ERROR] Image path not found: {img_path}")
            return False

        img = cv2.imread(img_path)
        if img is None:
            print(f"[HIGHLIGHT ERROR] Could not load image: {img_path}")
            return False

        h_img, w_img = img.shape[:2]
        results = self.reader.readtext(img)
        target_clean = " ".join(target_text.lower().strip().split())

        poly_pts = None

        for bbox, text, prob in results:
            found_text_clean = " ".join(text.lower().strip().split())

            if target_clean in found_text_clean:
                pts = np.array(bbox, dtype=np.int32)
                x_min_orig, y_min_orig = np.min(pts, axis=0)
                x_max_orig, y_max_orig = np.max(pts, axis=0)

                words = found_text_clean.split()
                if len(words) > 1 and target_clean in words:
                    # Precise sub-word bound calculation
                    full_w = x_max_orig - x_min_orig
                    target_idx = words.index(target_clean)

                    char_counts = [len(w) for w in words]
                    total_chars = sum(char_counts) + len(words) - 1

                    start_char = sum(char_counts[:target_idx]) + target_idx
                    end_char = start_char + len(target_clean)

                    sub_x1 = x_min_orig + int(full_w * (start_char / total_chars))
                    sub_x2 = x_min_orig + int(full_w * (end_char / total_chars))

                    x_min, y_min = sub_x1, y_min_orig
                    x_max, y_max = sub_x2, y_max_orig
                else:
                    x_min, y_min = x_min_orig, y_min_orig
                    x_max, y_max = x_max_orig, y_max_orig

                # --- APPLY PADDING WITH BOUNDARY CLAMPING ---
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w_img, x_max + padding)
                y_max = min(h_img, y_max + padding)

                poly_pts = np.array(
                    [
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                    ],
                    dtype=np.int32,
                )

                break

        if poly_pts is None:
            print(
                f"[HIGHLIGHT WARNING] Could not locate text '{target_text}' in '{os.path.basename(img_path)}'."
            )
            return False

        # --- DRAW SEMI-TRANSPARENT OVERLAY & BOUNDING BOX ---
        overlay = img.copy()
        cv2.fillPoly(overlay, [poly_pts], highlight_color)

        highlighted_img = cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0)
        cv2.polylines(
            highlighted_img,
            [poly_pts],
            isClosed=True,
            color=border_color,
            thickness=2,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, highlighted_img)
        print(
            f"[SUCCESS] Saved highlight audit image to: '{output_path}'"
        )
        return True