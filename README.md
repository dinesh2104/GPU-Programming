# 🚀 GPU Programming Code

This repository contains code and resources for GPU programming, including examples, tutorials, and projects aimed at optimizing applications for **high-performance computing**.

---

## 📂 Contents

### **Assignment 1: Inverted Gray Scale & Thomas Transformation**
- **Main Code:** `CS24M017.cu`
- **Description:** Implements image processing operations — inverted grayscale conversion and Thomas transformation.

---

### **Assignment 2: 2D Convolution Using CUDA**
- Implements a **2D convolution operation** using CUDA.
- Optimized with:
  - **Memory coalescing**
  - **Shared memory usage** for faster execution.

---

### **Assignment 3: Borůvka’s MST Algorithm in CUDA**
- Implemented **Borůvka’s Minimum Spanning Tree** algorithm in CUDA.
- Based on the research paper:  
  [Borůvka's Algorithm on GPU – PDF](https://repositorium.uminho.pt/bitstream/1822/53008/1/boruvka_uminho_cameraready_v2.pdf)

---

### **Assignment 4: Game Simulation – All-Pairs Shortest Path**
- Implemented **All-Pairs Shortest Path (APSP)** algorithm in CUDA.
- **Main Code:** `APSP.cu`

---

## 🛠 Requirements
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- C++ compiler (e.g., `g++`)

---

## ▶️ How to Run
```bash
# Compile
nvcc filename.cu -o output_binary

# Run
./output_binary
