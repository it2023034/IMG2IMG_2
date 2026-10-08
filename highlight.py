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
    ):
        """Locates target_text in img_path and saves a highlighted version."""
        if not os.path.exists(img_path):
            print(f"[HIGHLIGHT ERROR] Image path not found: {img_path}")
            return False

        img = cv2.imread(img_path)
        if img is None:
            print(f"[HIGHLIGHT ERROR] Could not load image: {img_path}")
            return False

        results = self.reader.readtext(img)
        target_clean = " ".join(target_text.lower().strip().split())

        poly_pts = None

        for bbox, text, prob in results:
            found_text_clean = " ".join(text.lower().strip().split())

            if target_clean in found_text_clean:
                pts = np.array(bbox, dtype=np.int32)

                words = found_text_clean.split()
                if len(words) > 1 and target_clean in words:
                    # Precise sub-word bound calculation
                    x_min, y_min = np.min(pts, axis=0)
                    x_max, y_max = np.max(pts, axis=0)
                    full_w = x_max - x_min
                    target_idx = words.index(target_clean)

                    char_counts = [len(w) for w in words]
                    total_chars = sum(char_counts) + len(words) - 1

                    start_char = sum(char_counts[:target_idx]) + target_idx
                    end_char = start_char + len(target_clean)

                    sub_x1 = max(
                        x_min,
                        x_min + int(full_w * (start_char / total_chars)) - 2,
                    )
                    sub_x2 = min(
                        x_max,
                        x_min + int(full_w * (end_char / total_chars)) + 2,
                    )

                    poly_pts = np.array(
                        [
                            [sub_x1, y_min],
                            [sub_x2, y_min],
                            [sub_x2, y_max],
                            [sub_x1, y_max],
                        ],
                        dtype=np.int32,
                    )
                else:
                    poly_pts = pts

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