# 👤 Face Attribute Analysis – Data Science Analytics Project

This project performs **data science analysis on facial attributes** using the **CelebA dataset**.  
Instead of image processing, the analysis is **attribute-based**, focusing on demographic and facial characteristics such as **gender, smiling, glasses, beard features**, and **attribute correlations**.

---

## 🎯 Project Goal
To analyze facial attributes like:
- Gender
- Smiling behavior
- Eyeglasses
- Beard features
- Most common facial attributes  
and generate **statistical insights and visualizations** using data science techniques.

---

## 📊 Dataset Used
**CelebA Dataset – Attributes File Only**

- File name: `list_attr_celeba.csv`
- Source: Kaggle (CelebA Dataset)
- Each row represents one face
- Attributes are labeled as `-1` (absent) or `1` (present)

📌 **Note:**  
Images are optional. This project focuses purely on **data analytics**, not image processing.

---

## 📁 Project Structure
face_attribute_analysis/
│── face_attribute_analysis.py
│── list_attr_celeba.csv
│── README.md

yaml
Copy code

---

## 🛠 Technologies Used
- Python 3.11
- Pandas
- NumPy
- Matplotlib
- Seaborn
- OpenCV (optional / future use)
- VS Code

---

## ⚙️ Installation

Make sure Python is installed, then install required libraries:

```bash
pip install pandas numpy matplotlib seaborn opencv-python
                                                      ## How to Run
```bash
python face_attribute_analysis.py                          