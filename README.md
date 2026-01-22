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
- **CPU-compatible** pipeline to avoid CUDA/GPU issues.

---

## 2️⃣ Project Structure

