
from tkinter import ttk, filedialog, simpledialog, messagebox
import tkinter as tk
import numpy as np
import cv2
import os
from collections import defaultdict
from PIL import Image, ImageTk

### 
# to adjust luminance 
# smaller values means less compression for that freq
Y_quantization_table = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])
# adjusts colour amount
CbCr_quantization_table = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

# order in which we read coefficients 
zigzag_index = [
    (0,0),(0,1),(1,0),(2,0),(1,1),(0,2),(0,3),(1,2),
    (2,1),(3,0),(4,0),(3,1),(2,2),(1,3),(0,4),(0,5),
    (1,4),(2,3),(3,2),(4,1),(5,0),(6,0),(5,1),(4,2),
    (3,3),(2,4),(1,5),(0,6),(0,7),(1,6),(2,5),(3,4),
    (4,3),(5,2),(6,1),(7,0),(7,1),(6,2),(5,3),(4,4),
    (3,5),(2,6),(1,7),(2,7),(3,6),(4,5),(5,4),(6,3),
    (7,2),(7,3),(6,4),(5,5),(4,6),(3,7),(4,7),(5,6),
    (6,5),(7,4),(7,5),(6,6),(5,7),(6,7),(7,6),(7,7)
]

mode_var = None

### 
# BGR to YCbCr by calculating luminance, 
# Cb+Cr and then +128 
def Convert_YCbCr(img):
    out = np.zeros_like(img, dtype=np.float32)
    print("Converting YCbCr")
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            B,G,R = img[r,c]
            Y  = 0.299*R + 0.587*G + 0.114*B
            Cb = 128 -0.168736*R -0.331264*G +0.5*B
            Cr = 128 +0.5*R -0.418688*G -0.081312*B
            out[r,c] = [Y,Cb,Cr]
    return np.clip(out,0,255).astype(np.uint8)

# YCbCr back to BGR 
# reverses conversion
def invConvert(img):
    out = np.zeros_like(img, dtype=np.uint8)
    print("invconvert")
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            Y,Cb,Cr = img[r,c].astype(np.float32)
            R = Y +1.402*(Cr-128)
            G = Y -0.34414*(Cb-128) -0.71414*(Cr-128)
            B = Y +1.772*(Cb-128)
            out[r,c] = [np.clip(B,0,255),np.clip(G,0,255),np.clip(R,0,255)]
    return out

### DCT / IDCT ###
# DCT on 8x8 blocks by setting values around 0 
# then finding frequency coefficinet
def get_dctBlock(block):
    block = block.astype(np.float32) - 128
    result = np.zeros((8, 8), dtype=np.float32)
    
    for u in range(8):
        for v in range(8):
            sum_val = 0.0
            for x in range(8):
                for y in range(8):
                    pixel = block[x, y]
                    cosx = np.cos(((2 * x + 1) * u * np.pi)/16)
                    cosy = np.cos(((2 * y + 1) * v * np.pi)/16)
                    sum_val+= pixel * cosx * cosy
            if u == 0:
                cu = 1 / np.sqrt(2)  
            else: 
                cu = 1
            if v==0:
                cv = 1 / np.sqrt(2)
            else:
                cv = 1
            result[u, v] = 0.25 * cu * cv * sum_val
    
    return result

# to get EXACT 8x8 blocks (otherwise crash)
def pad_image_to_multiple_of_8(img):
    height, width = img.shape[:2]
    new_height = (height + 7) // 8 * 8
    new_width = (width + 7) // 8 * 8
    padded_img = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    padded_img[:height, :width] = img
    return padded_img

def pad_grayscale_image_to_multiple_of_8(img):
    height, width = img.shape
    new_height = (height + 7) // 8 * 8
    new_width = (width + 7) // 8 * 8
    padded_img = np.zeros((new_height, new_width), dtype=img.dtype)
    padded_img[:height, :width] = img
    return padded_img

# convert padded img to 8x8 blocks ki list 
# (to be used for DCT)
def get_blocks(img):
    blocks = []
    rows, cols = img.shape
    for r in range(0, rows, 8):
        for c in range(0, cols, 8):
            block = img[r:r+8, c:c+8]
            blocks.append(block)
    return blocks

# MAIN AREA WHERE COMRPESSION HAPPENS (because we round values)
def quantize(dct_blocks, qtable):
    quantized_blocks = []
    for block in dct_blocks:
        quantized_block = np.round(block / qtable)
        quantized_blocks.append(quantized_block)
    return quantized_blocks

# converts 8x8 into 64 element list 
# +groups all zeros together
def zigzag_scan(block):
    zigzag = []
    for i, j in zigzag_index:
        zigzag.append(block[i][j])
        
    return zigzag

# counts consecutive 0s and then compresses 
def run_length_encode(zigzag):
    encoded = [(0, zigzag[0])]
    zeros = 0
    for coeff in zigzag[1:]:
        if coeff == 0:
            zeros += 1
        else:
            while zeros > 15:
                encoded.append((15, 0))
                zeros -= 16
            encoded.append((zeros, coeff))
            zeros = 0
    encoded.append((0, 0))
    return encoded

### HUFFMAN
def build_frequency_table(encoded_blocks):
    freq = defaultdict(int)
    for block in encoded_blocks:
        for pair in block:
            freq[pair] += 1
    return freq

#creats the codes
def generate_dummy_huffman_codes(freq_table):
    huffman_code = {}
    for i, symbol in enumerate(sorted(freq_table, key=freq_table.get, reverse=True)):
        huffman_code[symbol] = format(i, '08b') ##8bit##
    return huffman_code

def huffman_encode_block(block, huffman_codes):
    return ''.join(huffman_codes[pair] for pair in block)

###
# calls all the main funcs (pad, conversion, get_dct)
def process_image_for_dct(img):
    global mode_var
    mode = mode_var.get()
    
    if mode != "grayscale":  # Color image (Y, Cb, Cr)
        img = pad_image_to_multiple_of_8(img)
        ycbcr = Convert_YCbCr(img)
        Y_blocks = get_blocks(ycbcr[:, :, 0])
        Cb_blocks = get_blocks(ycbcr[:, :, 1])
        Cr_blocks = get_blocks(ycbcr[:, :, 2])
        Y_dct = [get_dctBlock(b) for b in Y_blocks]
        Cb_dct = [get_dctBlock(b) for b in Cb_blocks]
        Cr_dct = [get_dctBlock(b) for b in Cr_blocks]
        return Y_dct, Cb_dct, Cr_dct, img.shape[:2]
    
    else:  # Grayscale image
        print("GRAY")
        img = pad_grayscale_image_to_multiple_of_8(img)
        Y_blocks = get_blocks(img)
        Y_dct = [get_dctBlock(b) for b in Y_blocks]
        return Y_dct, None, None, img.shape[:2]

###
# matches bits togth and then outputs 
def huffman_decode_block(bitstream, inv):
    result = []
    curr_bits = ''

    for bit in bitstream:
        curr_bits += bit

        if curr_bits in inv:
            symbol = inv[curr_bits]
            result.append(symbol)
            curr_bits = ''
    return result

# converts pairs back to 64 ki list and then filling w 0s where needed
def run_length_decode(pairs):
    output = []
    for run, value in pairs:
        if run == 0 and value == 0:
            output.extend([0] * (64 - len(output))) #filling w 0s
            break
        output.extend([0] * run)
        output.append(value)
    if len(output) < 64:
        output.extend([0] * (64 - len(output)))
    return output

def inverse_zigzag(coefs):
    block = np.zeros((8, 8), dtype=np.float32)
    for idx, (i, j) in enumerate(zigzag_index):
        block[i, j] = coefs[idx]
    return block

# merges the blocks back into the image and making sure values in bw 0-255
def blocks_to_image(blocks,shape):
    height, width = shape
    image = np.zeros((height, width), dtype=np.uint8)
    block_index = 0
    for row in range(0, height, 8):
        for col in range(0, width, 8):
            image[row:row+8, col:col+8] = np.clip(blocks[block_index], 0, 255)
            block_index += 1
    return image

# Exact inverse of the dct code
def get_inversedctBlock(blocks):
    res = np.zeros((8,8), dtype=np.float32)
    for x in range(8):
        for y in range(8):
            s = 0.0
            for u in range(8):
                for v in range(8):
                    if u==0: 
                        cu = 1/np.sqrt(2) 
                    else:    
                        cu = 1
                    if v==0: 
                        cv = 1/np.sqrt(2) 
                    else:    
                        cv = 1
                    alpha= cu * cv
                    value = blocks[u, v]
                    cosx = np.cos((2*x + 1) * u * np.pi / 16)
                    cosy = np.cos((2*y + 1) * v * np.pi / 16)
                    s += alpha * value * cosx * cosy
            res[x,y] = 0.25 * s
    return res + 128.0

def decode_channals(encoded_blocks, q_table):
    inverse_codes = {bits: symbol for symbol, bits in huff_codes.items()}
    output = []
    for bits in encoded_blocks:
        coefs = run_length_decode(huffman_decode_block(bits, inverse_codes))
        
        dequant = q_table * inverse_zigzag(coefs)
        output.append(get_inversedctBlock(dequant).astype(np.float32))
    return output

    
###
def load_image():
    global mode_var,huff_codes
    path=filedialog.askopenfilename()
    if not path: return
    mode=mode_var.get()
    img=(cv2.imread(path,cv2.IMREAD_GRAYSCALE) if mode=='grayscale' else cv2.imread(path,cv2.IMREAD_COLOR))
    if img is None: return
    orig=os.path.getsize(path)
    w = simpledialog.askinteger('Width', 'Enter Width', minvalue=1, maxvalue=5000, initialvalue=img.shape[1])
    h = simpledialog.askinteger('Height', 'Enter Height', minvalue=1, maxvalue=5000, initialvalue=img.shape[0])
    img=cv2.resize(img,(w,h))
    print(f"Original: {orig/1024:.2f} KB")

    # ENCODE
    if mode!='grayscale':
        Yd,Cbd,Crd,shape=process_image_for_dct(img)
        Yq=quantize(Yd,Y_quantization_table)
        Cbq=quantize(Cbd,CbCr_quantization_table)
        Crq=quantize(Crd,CbCr_quantization_table)
        Yr=[run_length_encode(zigzag_scan(b)) for b in Yq]
        Cbr=[run_length_encode(zigzag_scan(b)) for b in Cbq]
        Crr=[run_length_encode(zigzag_scan(b)) for b in Crq]
        freq=build_frequency_table(Yr+Cbr+Crr)
        huff_codes=generate_dummy_huffman_codes(freq)
        Yh=[huffman_encode_block(b,huff_codes) for b in Yr]
        Cbh=[huffman_encode_block(b,huff_codes) for b in Cbr]
        Crh=[huffman_encode_block(b,huff_codes) for b in Crr]
        comp=sum(len(x) for x in Yh+Cbh+Crh)/8/1024
    else:
        Yd,_,_,shape=process_image_for_dct(img)
        Yq=quantize(Yd,Y_quantization_table)
        Yr=[run_length_encode(zigzag_scan(b)) for b in Yq]
        freq=build_frequency_table(Yr)
        huff_codes=generate_dummy_huffman_codes(freq)
        Yh=[huffman_encode_block(b,huff_codes) for b in Yr]
        comp=sum(len(x) for x in Yh)/8/1024

    print(f"Compressed: {comp:.2f} KB")

    # DECODE
    if mode!='grayscale':
        Yb=decode_channals(Yh,Y_quantization_table)
        Cbb=decode_channals(Cbh,CbCr_quantization_table)
        Crb=decode_channals(Crh,CbCr_quantization_table)
        Yim=blocks_to_image(Yb,shape)
        Cbim=blocks_to_image(Cbb,shape)
        Crim=blocks_to_image(Crb,shape)
        recon=invConvert(cv2.merge([Yim,Cbim,Crim]))
    else:
        Yb=decode_channals(Yh,Y_quantization_table)
        Yim=blocks_to_image(Yb,shape)
        recon=Yim

    top=tk.Toplevel(); top.title('Decompressed')
    if mode!='grayscale': disp=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(recon,cv2.COLOR_BGR2RGB)))
    else: disp=ImageTk.PhotoImage(Image.fromarray(recon))
    lbl=tk.Label(top,image=disp); lbl.image=disp; lbl.pack()
    def save_cb():
        f=filedialog.asksaveasfilename(defaultextension='.png',filetypes=[('PNG','*.png'),('All','*.*')])
        if f: cv2.imwrite(f,recon); messagebox.showinfo('Saved',f'Saved to {f}')
    tk.Button(top,text='Save As…',command=save_cb).pack(pady=5)

def main():
    global mode_var
    root=tk.Tk(); root.title('JPEG Compression'); root.geometry('300x120')
    tk.Button(root,text='Load ',command=load_image).pack(pady=10)
    tk.Label(root,text='Mode:').pack()
    mode_var=tk.StringVar(value='color')
    cb=ttk.Combobox(root,textvariable=mode_var,values=['color','grayscale']); cb.pack()
    root.mainloop()

if __name__=='__main__': 
    main()