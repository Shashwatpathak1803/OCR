# Multilingual (Hindi + English) Visiting Card OCR to Excel System

An offline, desktop Python application that scans visiting cards, recognizes both **Hindi (Devanagari script)** and **English (Latin script)** on the same card using **PaddleOCR**, classifies contact entities into **separate English and Hindi fields**, and automatically exports to professional **Excel (.xlsx)** spreadsheets.

**100% OFFLINE**: Uses zero cloud APIs, zero AI/LLM models, and zero external translation services. Everything runs locally on your Windows machine after dependencies and OCR models are installed.

---

## 🌟 Key Features

1. **Multilingual Offline OCR (Hindi & English)**:
   - Configured with PaddleOCR's official **Devanagari recognition model (`devanagari_PP-OCRv5_mobile_rec`)** and server detection (`PP-OCRv5_server_det`).
   - Recognizes Devanagari characters, Hindi words, names, and addresses along with English letters and numbers on the same visiting card.
   - 100% offline — zero calls to Google Translate or external APIs.

2. **Language Identification & Field Segregation**:
   - Uses Python-based Unicode script detection (`[\u0900-\u097F]` for Devanagari vs. `[a-zA-Z]` for Latin).
   - Segments mixed-language lines (e.g., `'Sales Manager सेल्स मैनेजर'` or `'Sector 62, Noida सेक्टर 62, नोएडा'`) into separate language streams without character loss.
   - Extracts and maintains separate fields for English and Hindi:

| Field | English Column | Hindi Column |
|---|---|---|
| Full Name | Rahul Sharma | राहुल शर्मा |
| Designation | Sales Manager | सेल्स मैनेजर |
| Company | ABC Technologies | एबीसी टेक्नोलॉजीज |
| Address | Sector 62, Noida | सेक्टर 62, नोएडा |
| City | Noida | नोएडा |
| State | Uttar Pradesh | उत्तर प्रदेश |

   - If a card has only one language, the other language fields remain blank without hallucination or fake translation.

3. **Deterministic Contact Extraction**:
   - **Phone Numbers**: Captures Indian 10-digit mobiles, landlines with STD codes, and international numbers. Classifies by labels (`Mobile`, `Mob`, `Tel`, `Phone`, `मो.`, `फोन`).
   - **Emails & URLs**: Validates RFC 5322 emails, company websites, LinkedIn profiles, and social links.
   - **PIN / ZIP Codes**: Detects 6-digit Indian PIN codes (`\b[1-9]\d{5}\b`) and international postal codes.
   - **Data Validation**: Structured with Pydantic (`VisitingCardRecord`).

4. **Interactive Verification & In-Place Editing**:
   - Double-click any cell in the table to edit English or Hindi values independently.
   - **Side-by-Side Card Inspector**: Shows English and Hindi fields in comparative two-column view alongside the original photograph and raw OCR text.

5. **Multilingual Excel Export (OpenPyXL)**:
   - 29 standard columns with separate English and Hindi headers.
   - Full UTF-8 support for Devanagari script rendering in Excel.
   - **Preserves Leading Zeros**: Applies explicit text cell formatting (`@`) to phone numbers and PIN codes.
   - Supports creating new workbooks or appending to existing `.xlsx` files without data loss.

6. **Adaptive, Persistent Bulk Queue**:
   - Images are hashed and saved to a local disk-backed SQLite queue; uploads are not retained as a collection of decoded images in Streamlit session memory.
   - Duplicate image content is detected by SHA-256, completed cards are not queued again, and results/raw OCR text are saved immediately.
   - Processing runs sequentially to bound memory use. Pause, resume, retry, and recovery of interrupted cards are supported.
   - Progress, failures, estimated remaining time, average card time, stage timings, and total processing time are shown in the UI.
   - Data is stored outside the project folder in the operating system's local application-data directory. Set `OCR_DATA_DIR` in `.env` to choose another local folder.

7. **Hardware-Aware OCR**:
   - A compatible Paddle GPU is used when the installed Paddle build can detect one; otherwise the engine uses CPU.
   - CPU uses the cached PP-OCRv4 mobile text detector when available; the Devanagari PP-OCRv5 mobile recognizer remains shared by Hindi, English, and mixed-language cards.
   - GPU mode uses the PP-OCRv5 server detector. Document orientation/dewarping and text-line orientation models are disabled because the app already offers perspective correction and deskewing.
   - If the lightweight detector is not already cached, CPU mode uses PaddleOCR's Hindi default models instead of downloading a new model during OCR.

---

## 📁 Project Structure

```
OCR/
│
├── app.py                  # Streamlit user interface & multilingual workflow
├── ocr_engine.py           # PaddleOCR Devanagari & Latin offline engine
├── image_processing.py     # OpenCV preprocessing (warp, deskew, CLAHE, denoise)
├── rule_extractor.py       # Deterministic script detection & multilingual parser
├── excel_generator.py      # OpenPyXL 29-column multilingual Excel generator
├── test_workflow.py        # Automated test suite (English, Hindi, and Bilingual)
├── requirements.txt        # Python package dependencies
├── .env                    # Environment settings
├── .env.example            # Environment template
└── README.md               # Documentation and execution guide
```

---

## 🛠️ Windows Installation & Setup Guide

### 1. Prerequisites
- **Python 3.11+** installed on Windows.
- Make sure Python and `pip` are added to your Windows PATH.

### 2. Open PowerShell
```powershell
cd C:\Users\shash\Desktop\OCR
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

The OCR models are cached by PaddleX after first-time setup. Once those models are present, OCR inference and queue processing run locally without cloud OCR or translation services.

---

## 🚀 Running the Application

Start the Streamlit application:
```powershell
streamlit run app.py
```

The application will open in your default browser at:
```
http://localhost:8501
```

---

## 🧪 Running the Automated Test Suite

Run the multilingual test suite locally to verify English-only, Hindi-only, and mixed Hindi-English bilingual cards:
```powershell
python test_workflow.py
```

## ⚡ Performance Benchmark

The existing engine and the lightweight CPU detector were benchmarked on the same eight images in `sample_cards` (five supplied layout examples and three English/Hindi/bilingual test cards). The test machine had eight logical CPU cores, PaddlePaddle 3.3.1 CPU, and no CUDA-enabled Paddle build.

| Configuration | First model initialization | Average card (image load + preprocess + OCR + extraction) | OCR-only average | RSS after eight cards | No-text failures |
|---|---:|---:|---:|---:|---:|
| Existing PP-OCRv5 server detector, orientation models enabled | 4.18 s | 12.80 s | 12.78 s | 406 MiB | 1/8 |
| PP-OCRv4 mobile detector + same PP-OCRv5 Devanagari recognizer, orientation models disabled | 3.60 s benchmark (6.82 s in queue smoke test) | 3.29 s | 3.28 s | 351 MiB | 1/8 |

On this sample set, the CPU configuration was about **3.9× faster** and used about **13.5% less resident memory** after processing. One-time initialization varied between benchmark runs, as shown above. The table's per-card figure excludes database writes; live queue metrics report database-save time separately and include it in total processing time. Hindi, English, and bilingual test assertions (including separate names/designations/company fields and Excel append) passed; the blurry image remained unreadable in both configurations. The five layout examples do not have independent ground-truth transcriptions, so these results are not a general accuracy certification. OCR text remains available for review, and difficult cards should still be manually verified. The app does not translate or rewrite OCR output.
