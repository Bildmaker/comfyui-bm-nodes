"""
Tool: Load Image from Dir (bm)
Author: Sebastian
Date: 2025-03-28

Description:
------------
This ComfyUI custom node loads a single image from one or more directories 
based on a specified index and sort order. It supports natural filename sorting 
(e.g., file1.jpg, file2.jpg, ..., file10.jpg), which is essential for batch-style 
workflows involving sequentially numbered files.

Usage:
------
- 'directory': A semicolon-separated list of folders to scan for image files.
- 'file_number': The index of the file to load from the sorted list (0-based).
- 'sort_by': Choose how the file list should be sorted:
    - Filename (natural alphanumeric sort)
    - Date (Newest First)
    - Date (Oldest First)
    - FileSize

Outputs:
--------
- image:       The loaded image as a tensor.
- mask:        An alpha mask if present; otherwise, an empty mask.
- filepath:    Full path to the loaded image file.
- filename:    The name of the image file.
- filename_no_ext: Filename without extension.
- file_count:  The number of matching image files found.

Compatible formats: .png, .jpg, .jpeg, .webp, .bmp, .tiff
"""



import os
import re
from PIL import Image, ImageOps
import numpy as np
import torch

class LoadImageFromDirBM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "tooltip": "Separate multiple directories with semicolon (;)"
                }),
                "file_number": ("INT", {
                    "default": 0, 
                    "min": 0, 
                    "step": 1,
                    "tooltip": "Index of the image to load"
                }),
                "sort_by": ([
                    "Filename", 
                    "Date (Newest First)", 
                    "Date (Oldest First)", 
                    "FileSize"
                ], {"default": "Filename", "tooltip": "Sorting method for files"})
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "filepath", "filename", "filename_no_ext", "file_count")
    FUNCTION = "load_image"
    CATEGORY = "image"

    def natural_sort_key(self, s):
        name = os.path.basename(s)
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', name)]

    def sort_files(self, files, sort_method):
        if sort_method == "Filename":
            return sorted(files, key=self.natural_sort_key)
        elif sort_method == "Date (Newest First)":
            return sorted(files, key=lambda x: os.path.getmtime(x), reverse=True)
        elif sort_method == "Date (Oldest First)":
            return sorted(files, key=lambda x: os.path.getmtime(x))
        elif sort_method == "FileSize":
            return sorted(files, key=lambda x: os.path.getsize(x))
        return files

    def load_image(self, directory, file_number, sort_by):
        directories = [p.strip() for p in directory.split(";") if os.path.isdir(p.strip())]
        image_files = []
        valid_extensions = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")

        for dir_path in directories:
            for filename in os.listdir(dir_path):
                if filename.lower().endswith(valid_extensions):
                    image_files.append(os.path.join(dir_path, filename))

        image_files = self.sort_files(image_files, sort_by)
        file_count = len(image_files)

        if file_count == 0:
            raise FileNotFoundError("No valid image files found in the given directory/directories.")

        index = max(0, min(file_number, file_count - 1))
        image_path = image_files[index]
        filename = os.path.basename(image_path)
        filename_no_ext = os.path.splitext(filename)[0]

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)

        has_alpha = 'A' in image.getbands()
        mask = torch.zeros((64, 64), dtype=torch.float32)

        if has_alpha:
            alpha_channel = image.getchannel('A')
            alpha_np = np.array(alpha_channel).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha_np)
            image = image.convert("RGB")
        else:
            image = image.convert("RGB")

        np_image = np.array(image).astype(np.float32) / 255.0
        tensor_image = torch.from_numpy(np_image).unsqueeze(0)

        return (
            tensor_image, 
            mask.unsqueeze(0), 
            image_path, 
            filename, 
            filename_no_ext, 
            file_count
        )
