# JPEG-Compressor

This is an early version of a JPEG compressor written in Python.  
It demonstrates the main steps of JPEG compression including:

- Color space conversion (BGR to YCbCr)
- Block-wise Discrete Cosine Transform (DCT)
- Quantization using standard tables
- Zigzag scanning and run-length encoding
- Huffman coding (dummy implementation)
- Decompression and image reconstruction

The code uses OpenCV, NumPy, and Tkinter for GUI and image handling.  
You can run and experiment with the code in a Jupyter Notebook or as a standalone Python script.

## Usage

1. Run the code in Jupyter Notebook or as a Python script.
2. Use the GUI to load an image.
3. Select compression mode: color or grayscale.
4. View and save the decompressed image.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Pillow (`PIL`)
- Tkinter

Install dependencies with:

```sh
pip install numpy opencv-python pillow
```

## Note

This implementation is for educational purposes and does not