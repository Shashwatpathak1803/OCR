"""
Automated Test Suite for Multilingual (Hindi + English) Visiting Card OCR System.
Tests:
1. English-only visiting card
2. Hindi-only visiting card (Devanagari script)
3. Mixed Hindi-English bilingual visiting card
4. Multilingual Excel generation with separate English & Hindi columns
5. Excel append functionality preserving Devanagari text and leading zeros
"""

import os
import io
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import openpyxl

from image_processing import preprocess_visiting_card
from ocr_engine import extract_text_from_image, get_ocr_engine
from rule_extractor import extract_visiting_card_data, VisitingCardRecord, has_devanagari, has_latin
from excel_generator import (
    create_new_excel,
    append_to_existing_excel,
    read_existing_excel_sheets,
    EXCEL_COLUMNS,
)

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_cards")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Select a TrueType font that renders Devanagari and Latin on Windows
FONT_PATH = "C:/Windows/Fonts/Nirmala.ttc"
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/arial.ttf"


def create_card_image(
    filename: str,
    lines: list,
    font_path: str = FONT_PATH,
    bg_color=(255, 255, 255),
    width: int = 750,
    height: int = 420,
) -> str:
    """Helper to render synthetic visiting card with Devanagari and Latin fonts."""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 12], fill=(30, 58, 138))  # Blue top bar
    draw.rectangle([0, 0, width - 1, height - 1], outline=(200, 200, 200), width=2)

    font = ImageFont.truetype(font_path, 22) if os.path.exists(font_path) else ImageFont.load_default()

    y_offset = 35
    for item in lines:
        text = item.get("text", "")
        color = item.get("color", (20, 20, 20))
        draw.text((35, y_offset), text, fill=color, font=font)
        y_offset += item.get("gap", 34)

    cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    filepath = os.path.join(SAMPLE_DIR, filename)
    cv2.imwrite(filepath, cv_img)
    return filepath


def run_tests():
    print("=" * 60)
    print("RUNNING MULTILINGUAL (HINDI + ENGLISH) VISITING CARD TEST SUITE")
    print("=" * 60)

    # 1. English-only Card
    card_en_lines = [
        {"text": "ALEXANDER WRIGHT", "gap": 32},
        {"text": "Vice President of Engineering", "color": (80, 80, 80), "gap": 34},
        {"text": "Quantum Leap Technologies Pvt Ltd", "color": (30, 58, 138), "gap": 36},
        {"text": "Mobile: +91 98765 43210", "gap": 30},
        {"text": "Email: alex.wright@quantumleap.io", "gap": 30},
        {"text": "Address: 42 Silicon Heights, Tech Zone 4, Noida", "gap": 30},
        {"text": "State: Uttar Pradesh, PIN: 201306, India", "gap": 30},
    ]
    card_en_path = create_card_image("test_english_only.jpg", card_en_lines)

    # 2. Hindi-only Card (Devanagari)
    card_hi_lines = [
        {"text": "राजेश शर्मा", "gap": 32},
        {"text": "प्रबंध निदेशक", "color": (80, 80, 80), "gap": 34},
        {"text": "शर्मा लॉजिस्टिक्स प्राइवेट लिमिटेड", "color": (150, 40, 20), "gap": 36},
        {"text": "मोबाइल: 09412345678", "gap": 30},
        {"text": "ईमेल: rajesh@sharmalogistics.in", "gap": 30},
        {"text": "पता: प्लॉट 12, ट्रांसपोर्ट नगर, जयपुर, राजस्थान", "gap": 30},
        {"text": "पिन कोड: 302003, भारत", "gap": 30},
    ]
    card_hi_path = create_card_image("test_hindi_only.jpg", card_hi_lines)

    # 3. Bilingual Mixed Hindi + English Card
    card_bilingual_lines = [
        {"text": "Rahul Sharma राहुल शर्मा", "gap": 32},
        {"text": "Sales Manager सेल्स मैनेजर", "color": (80, 80, 80), "gap": 34},
        {"text": "ABC Technologies एबीसी टेक्नोलॉजीज", "color": (20, 100, 50), "gap": 36},
        {"text": "Mobile: +91 98765 43210", "gap": 30},
        {"text": "Email: rahul.s@abctech.com", "gap": 30},
        {"text": "Sector 62, Noida सेक्टर 62, नोएडा", "gap": 30},
        {"text": "Uttar Pradesh उत्तर प्रदेश, PIN: 201309", "gap": 30},
    ]
    card_bi_path = create_card_image("test_bilingual_card.jpg", card_bilingual_lines)

    test_cards = [
        ("English-Only Card", card_en_path),
        ("Hindi-Only Card", card_hi_path),
        ("Bilingual (Hindi + English) Card", card_bi_path),
    ]

    ocr_engine = get_ocr_engine(lang="hi")
    extracted_records = []

    print("\n1. Testing Multilingual OCR Recognition & Separation:")
    for label, cpath in test_cards:
        print(f"\n--- Testing: {label} ({os.path.basename(cpath)}) ---")
        prep = preprocess_visiting_card(cpath)

        ocr_res = extract_text_from_image(prep["processed_cv"], ocr_engine=ocr_engine)
        print(f"OCR Success: {ocr_res['success']} | Lines: {ocr_res['line_count']} | Conf: {ocr_res['average_confidence']:.2f}")

        # Deterministic multilingual extraction
        record_model = extract_visiting_card_data(
            ocr_text=ocr_res["raw_text"],
            lines_info=ocr_res["lines"],
            image_filename=os.path.basename(cpath),
            serial_number=len(extracted_records) + 1,
        )

        assert isinstance(record_model, VisitingCardRecord), "Record is not a VisitingCardRecord!"
        rec = record_model.to_dict()

        print(f"English Name: '{rec['full_name_en']}' | Hindi Name: '{rec['full_name_hi']}'")
        print(f"English Desig: '{rec['designation_en']}' | Hindi Desig: '{rec['designation_hi']}'")
        print(f"English Comp: '{rec['company_name_en']}' | Hindi Comp: '{rec['company_name_hi']}'")
        print(f"Mobile: '{rec['mobile_number']}' | Email: '{rec['email']}' | PIN: '{rec['pin_code']}'")
        print(f"City EN: '{rec['city_en']}' | City HI: '{rec['city_hi']}' | State EN: '{rec['state_en']}' | State HI: '{rec['state_hi']}'")

        extracted_records.append(rec)

    # Validations on extracted cards
    rec_en = extracted_records[0]
    assert rec_en["full_name_en"], "English name should be populated on English card!"
    assert not rec_en["full_name_hi"], "Hindi name should be empty on English-only card!"

    rec_hi = extracted_records[1]
    assert rec_hi["full_name_hi"], "Hindi name should be populated on Hindi card!"
    assert not rec_hi["full_name_en"], "English name should be empty on Hindi-only card!"
    assert "शर्मा" in rec_hi["full_name_hi"] or "राजेश" in rec_hi["full_name_hi"], f"Expected Devanagari text in Hindi name: {rec_hi['full_name_hi']}"

    rec_bi = extracted_records[2]
    assert rec_bi["full_name_en"] and rec_bi["full_name_hi"], f"Bilingual card should populate both English and Hindi names! EN='{rec_bi['full_name_en']}', HI='{rec_bi['full_name_hi']}'"
    print("\n✅ Language identification & field segregation verified across all card types!")

    # 2. Test Excel Generation with Multilingual Columns
    print("\n2. Testing Multilingual Excel Workbook Generation:")
    excel_buf = create_new_excel(extracted_records)
    assert excel_buf is not None and excel_buf.getvalue(), "Excel buffer empty!"

    wb = openpyxl.load_workbook(excel_buf)
    ws = wb.active
    print(f"Excel Sheet: '{ws.title}' | Total Rows: {ws.max_row} | Total Columns: {ws.max_column}")
    assert ws.max_row == 4, f"Expected 4 rows (1 header + 3 cards), got {ws.max_row}"
    assert ws.max_column == len(EXCEL_COLUMNS), f"Expected {len(EXCEL_COLUMNS)} columns, got {ws.max_column}"

    # Verify column headers include separate English and Hindi columns
    headers = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    assert "Full Name (English)" in headers and "Full Name (Hindi)" in headers, "Separate Name columns missing in Excel!"
    assert "Company Name (English)" in headers and "Company Name (Hindi)" in headers, "Separate Company columns missing!"
    assert "Address (English)" in headers and "Address (Hindi)" in headers, "Separate Address columns missing!"

    # Verify Devanagari content inside Excel cell
    hi_name_val = ws.cell(3, 3).value  # Row 3 (Hindi card), Column 3 is Full Name (Hindi)
    print(f"Excel Row 3 Hindi Name Cell: '{hi_name_val}'")
    assert has_devanagari(str(hi_name_val)), f"Devanagari text not found in Excel Hindi cell: '{hi_name_val}'"

    # Verify leading-zero preservation
    hi_mobile_cell = ws.cell(3, 13)  # Mobile Number column
    print(f"Excel Row 3 Mobile Cell: Value='{hi_mobile_cell.value}' | Format='{hi_mobile_cell.number_format}'")
    assert hi_mobile_cell.number_format == "@", f"Expected '@' text format, got {hi_mobile_cell.number_format}"
    assert str(hi_mobile_cell.value).startswith("0"), f"Leading zero lost: {hi_mobile_cell.value}"
    print("✅ Multilingual Excel generation with Devanagari UTF-8 verified!")

    # 3. Test Appending to Existing Workbook
    print("\n3. Testing Appending to Multilingual Excel Workbook:")
    sheets = read_existing_excel_sheets(excel_buf)
    new_card = [{
        "full_name_en": "Deepak Verma",
        "full_name_hi": "दीपक वर्मा",
        "designation_en": "Executive Director",
        "designation_hi": "कार्यकारी निदेशक",
        "company_name_en": "Verma Infotech",
        "company_name_hi": "वर्मा इन्फोटेक",
        "mobile_number": "09811223344",
        "email": "deepak@verma.in",
        "city_en": "Lucknow",
        "city_hi": "लखनऊ",
        "state_en": "Uttar Pradesh",
        "state_hi": "उत्तर प्रदेश",
        "country": "India",
        "pin_code": "226001",
        "image_filename": "card4_deepak.jpg",
    }]

    updated_buf, count = append_to_existing_excel(excel_buf, sheets[0], new_card)
    assert count == 1, f"Expected 1 appended, got {count}"

    wb_up = openpyxl.load_workbook(updated_buf)
    ws_up = wb_up[sheets[0]]
    assert ws_up.max_row == 5, f"Expected 5 rows, got {ws_up.max_row}"

    appended_name_hi = ws_up.cell(5, 3).value
    appended_mobile = ws_up.cell(5, 13).value
    print(f"Appended S.No: {ws_up.cell(5, 1).value} | Name (HI): '{appended_name_hi}' | Mobile: '{appended_mobile}'")
    assert appended_name_hi == "दीपक वर्मा", f"Appended Hindi text altered: '{appended_name_hi}'"
    assert str(appended_mobile) == "09811223344", f"Appended mobile altered: '{appended_mobile}'"
    print("✅ Appending to Multilingual Excel preserves Devanagari and leading zeros!")

    print("\n" + "=" * 60)
    print("ALL MULTILINGUAL TESTS PASSED SUCCESSFULLY! 🚀")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
