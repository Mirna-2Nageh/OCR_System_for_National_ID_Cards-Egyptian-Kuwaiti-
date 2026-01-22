
```markdown
# Egyptian/Kuwaiti ID OCR System

📄 **Project:** OCR system for Egyptian and Kuwaiti National ID Cards  
🚀 **Framework:** Streamlit + YOLOv8 + EasyOCR  
💻 **Device:** CPU compatible (GPU optional but not required)

---

## 1️⃣ Overview

This project allows you to:

- Detect and extract key fields from Egyptian and Kuwaiti National ID cards.
- Extract the following fields:
  - Full Name (الاسم)
  - ID Number (الرقم القومي)
  - Birth Date (تاريخ الميلاد)
  - Address (العنوان)
  - Country (الجنسية)
- Automatically parse the ID number to get:
  - Age (العمر)
  - Gender (النوع)
  - Governorate (المحافظة)
- Display extraction results in a user-friendly Streamlit web interface.

The system uses:

- **YOLOv8** for object detection of ID fields.
- **EasyOCR** for Arabic and English text recognition.
- **CPU compatible** pipeline to avoid CUDA/GPU issues.

---

## 2️⃣ Project Structure

```

OCR_IDS/
│
├─ models/
│   └─ best.pt          # YOLOv8 trained weights
│
├─ app.py               # Streamlit app (main script)
│
├─ README.md            # This file
│
└─ requirements.txt     # Python dependencies (optional)

````

---

## 3️⃣ Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/OCR_IDS.git
cd OCR_IDS
````

2. Create a virtual environment (recommended):

```bash
conda create -n id_ocr python=3.10
conda activate id_ocr
```

3. Install dependencies:

```bash
pip install streamlit ultralytics easyocr opencv-python pillow numpy arabic-reshaper python-bidi
```

> ⚠️ Make sure `models/best.pt` exists.
> GPU is optional, CPU is fully supported.

---

## 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

* Open the link in your browser (usually `http://localhost:8501`).

---

## 5️⃣ Usage Steps

### Step 1: Upload ID Card Image

Click **"Browse files"** and select a jpg/jpeg/png image of an Egyptian or Kuwaiti ID card.

![Upload Screenshot](images/upload_placeholder.png)

---

### Step 2: Image Preview

The app will display the uploaded image.

![Preview Screenshot](images/preview_placeholder.png)

---

### Step 3: Extracted Data

After processing, the app will display:

* Full Name, ID Number, Birth Date, Address, Country.
* Parsed data from ID Number: Age, Gender, Governorate.

![Result Screenshot](images/result_placeholder.png)

---

## 6️⃣ Notes

* **CPU Execution:** All models run on CPU by default to ensure compatibility with older GPUs (e.g., Quadro M1200).
* **Bounding Boxes:** YOLO detects regions of interest, and EasyOCR extracts text from cropped areas.
* **Arabic Support:** Arabic reshaper and bidi algorithms ensure proper display of Arabic text.
* **ID Parsing:** The system validates the ID number and calculates age, gender, and governorate.
* **Streamlit Friendly:** The interface is simple and interactive.

---

## 7️⃣ References

* [YOLOv8 Documentation](https://docs.ultralytics.com/)
* [EasyOCR Documentation](https://www.jaided.ai/easyocr/)
* [Arabic Reshaper & Python-Bidi](https://pypi.org/project/arabic-reshaper/)

---

## 8️⃣ License

MIT License © 2026
(Modify as needed)

---

## 9️⃣ Author

Marno000onaaa

```

---

