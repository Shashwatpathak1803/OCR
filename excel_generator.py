"""
Excel Generator Module for Multilingual Visiting Card OCR.
Uses OpenPyXL to create professional, styled Excel workbooks and append records
to existing Excel workbooks with leading-zero preservation, UTF-8 Devanagari Hindi support,
and separate columns for English and Hindi fields.
"""

import io
from typing import List, Dict, Any, Tuple, Optional
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# Multilingual Excel Columns (Separate English and Hindi)
EXCEL_COLUMNS: List[Tuple[str, str]] = [
    ("Serial Number", "serial_number"),
    ("Full Name (English)", "full_name_en"),
    ("Full Name (Hindi)", "full_name_hi"),
    ("First Name (English)", "first_name_en"),
    ("First Name (Hindi)", "first_name_hi"),
    ("Last Name (English)", "last_name_en"),
    ("Last Name (Hindi)", "last_name_hi"),
    ("Designation (English)", "designation_en"),
    ("Designation (Hindi)", "designation_hi"),
    ("Company Name (English)", "company_name_en"),
    ("Company Name (Hindi)", "company_name_hi"),
    ("Department", "department"),
    ("Mobile Number", "mobile_number"),
    ("Alternate Number", "alternate_number"),
    ("Other Phone Numbers", "other_phone_numbers"),
    ("Email Address", "email"),
    ("Other Email Addresses", "other_emails"),
    ("Website", "website"),
    ("Address (English)", "address_en"),
    ("Address (Hindi)", "address_hi"),
    ("City (English)", "city_en"),
    ("City (Hindi)", "city_hi"),
    ("State (English)", "state_en"),
    ("State (Hindi)", "state_hi"),
    ("Country", "country"),
    ("PIN Code", "pin_code"),
    ("LinkedIn", "linkedin"),
    ("Other Socials", "other_socials"),
    ("Original Image Filename", "image_filename"),
]

HEADER_NAMES = [col[0] for col in EXCEL_COLUMNS]

# Styling Constants
HEADER_FILL = PatternFill(start_color="1F4E78", end_color="1F4E78", fill_type="solid")  # Deep Navy Blue
HEADER_FONT = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
ZEBRA_FILL = PatternFill(start_color="F9FAFB", end_color="F9FAFB", fill_type="solid")  # Subtle light gray
REGULAR_FONT = Font(name="Calibri", size=10, color="000000")
BORDER_THIN = Border(
    left=Side(style="thin", color="E0E0E0"),
    right=Side(style="thin", color="E0E0E0"),
    top=Side(style="thin", color="E0E0E0"),
    bottom=Side(style="thin", color="E0E0E0"),
)


def _apply_header_styles(ws, max_col: int):
    """Style header row with dark blue fill, white bold font, centered alignment, and freeze pane."""
    ws.row_dimensions[1].height = 28
    for col_idx in range(1, max_col + 1):
        cell = ws.cell(row=1, column=col_idx)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER_THIN

    # Freeze header row
    ws.freeze_panes = "A2"


def _auto_adjust_column_widths(ws, min_width: int = 14, max_width: int = 42):
    """Dynamically adjust column widths based on maximum cell content length."""
    for col in ws.columns:
        col_letter = get_column_letter(col[0].column)
        max_len = 0
        for cell in col:
            val_str = str(cell.value or "")
            if val_str:
                max_len = max(max_len, len(val_str))
        adjusted_width = min(max(max_len + 3, min_width), max_width)
        ws.column_dimensions[col_letter].width = adjusted_width


def _resolve_record_value(record: Dict[str, Any], key: str) -> str:
    """Helper to get record value with fallback to legacy keys."""
    val = record.get(key)
    if val is not None and str(val).strip():
        return str(val).strip()

    # Fallbacks for legacy records
    legacy_fallbacks = {
        "full_name_en": "full_name",
        "first_name_en": "first_name",
        "last_name_en": "last_name",
        "designation_en": "designation",
        "company_name_en": "company_name",
        "address_en": "address",
        "city_en": "city",
        "state_en": "state",
    }
    fb_key = legacy_fallbacks.get(key)
    if fb_key and record.get(fb_key):
        return str(record.get(fb_key)).strip()

    return ""


def create_new_excel(records: List[Dict[str, Any]], sheet_title: str = "Visiting Cards") -> io.BytesIO:
    """
    Generate a brand-new professional Excel workbook from multilingual card records.
    Supports Devanagari script natively in UTF-8 and preserves leading zeros.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_title

    # Write headers
    for col_idx, (header, _) in enumerate(EXCEL_COLUMNS, start=1):
        ws.cell(row=1, column=col_idx, value=header)

    _apply_header_styles(ws, len(EXCEL_COLUMNS))

    # Write data rows
    for row_idx, record in enumerate(records, start=2):
        ws.row_dimensions[row_idx].height = 20
        is_even = (row_idx % 2 == 0)

        for col_idx, (header, key) in enumerate(EXCEL_COLUMNS, start=1):
            cell = ws.cell(row=row_idx, column=col_idx)

            if key == "serial_number":
                raw_val = record.get("serial_number", row_idx - 1)
                cell.value = int(raw_val) if str(raw_val).isdigit() else raw_val
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                val_str = _resolve_record_value(record, key)

                # Preserve leading zeros for phone and PIN codes
                if key in ("mobile_number", "alternate_number", "other_phone_numbers", "pin_code"):
                    cell.number_format = "@"
                    cell.value = val_str
                    cell.alignment = Alignment(
                        horizontal="center" if key == "pin_code" else "left",
                        vertical="center"
                    )
                else:
                    cell.value = val_str
                    cell.alignment = Alignment(horizontal="left", vertical="center")

            cell.font = REGULAR_FONT
            cell.border = BORDER_THIN
            if is_even:
                cell.fill = ZEBRA_FILL

    if len(records) > 0:
        ws.auto_filter.ref = ws.dimensions

    _auto_adjust_column_widths(ws)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output


def read_existing_excel_sheets(file_data: Any) -> List[str]:
    """Inspect an existing Excel workbook and return its list of sheet names."""
    if isinstance(file_data, (bytes, bytearray)):
        wb = openpyxl.load_workbook(io.BytesIO(file_data), read_only=True)
    elif isinstance(file_data, io.BytesIO):
        wb = openpyxl.load_workbook(file_data, read_only=True)
    else:
        wb = openpyxl.load_workbook(file_data, read_only=True)
    sheets = list(wb.sheetnames)
    wb.close()
    return sheets


def preview_existing_excel_info(file_data: Any, sheet_name: Optional[str] = None) -> Dict[str, Any]:
    """Get metadata about an existing workbook and sheet."""
    if isinstance(file_data, (bytes, bytearray)):
        stream = io.BytesIO(file_data)
    elif isinstance(file_data, io.BytesIO):
        stream = io.BytesIO(file_data.getvalue())
    else:
        stream = file_data

    wb = openpyxl.load_workbook(stream, data_only=True)
    active_sheet = sheet_name if sheet_name and sheet_name in wb.sheetnames else wb.sheetnames[0]
    ws = wb[active_sheet]

    headers = [str(cell.value).strip() for cell in ws[1] if cell.value is not None]
    max_row = ws.max_row
    data_rows = max(0, max_row - 1)

    wb.close()
    return {
        "sheet_names": wb.sheetnames,
        "selected_sheet": active_sheet,
        "existing_headers": headers,
        "row_count": max_row,
        "record_count": data_rows,
    }


def append_to_existing_excel(
    file_data: Any,
    sheet_name: str,
    new_records: List[Dict[str, Any]],
) -> Tuple[io.BytesIO, int]:
    """
    Append newly extracted multilingual records into an existing Excel workbook on the selected sheet.
    Preserves all existing data, maps fields, computes new serial numbers, and maintains UTF-8 formatting.
    """
    if isinstance(file_data, (bytes, bytearray)):
        stream = io.BytesIO(file_data)
    elif isinstance(file_data, io.BytesIO):
        stream = io.BytesIO(file_data.getvalue())
    else:
        stream = file_data

    wb = openpyxl.load_workbook(stream)
    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Worksheet '{sheet_name}' not found in workbook. Available: {wb.sheetnames}")

    ws = wb[sheet_name]

    col_map: Dict[str, int] = {}
    header_row = 1
    max_col = ws.max_column or 1

    for col_idx in range(1, max_col + 1):
        cell_val = ws.cell(row=header_row, column=col_idx).value
        if cell_val is not None:
            norm_name = str(cell_val).strip().lower()
            col_map[norm_name] = col_idx

    field_to_col: Dict[str, int] = {}
    current_last_col = max_col

    for header, key in EXCEL_COLUMNS:
        norm_header = header.strip().lower()
        if norm_header in col_map:
            field_to_col[key] = col_map[norm_header]
        else:
            alt_found = False
            for alt in [key.lower(), key.replace("_", " ").lower()]:
                if alt in col_map:
                    field_to_col[key] = col_map[alt]
                    alt_found = True
                    break
            if not alt_found:
                current_last_col += 1
                new_cell = ws.cell(row=header_row, column=current_last_col, value=header)
                new_cell.fill = HEADER_FILL
                new_cell.font = HEADER_FONT
                new_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
                new_cell.border = BORDER_THIN
                field_to_col[key] = current_last_col

    # Find highest existing serial number
    max_sno = 0
    sno_col = field_to_col.get("serial_number")
    max_row = ws.max_row or 1

    if sno_col:
        for r in range(2, max_row + 1):
            val = ws.cell(row=r, column=sno_col).value
            if val is not None and str(val).isdigit():
                max_sno = max(max_sno, int(val))
    else:
        max_sno = max_row - 1

    current_insert_row = max_row + 1

    for record in new_records:
        ws.row_dimensions[current_insert_row].height = 20
        is_even = (current_insert_row % 2 == 0)
        max_sno += 1

        for header, key in EXCEL_COLUMNS:
            target_col = field_to_col.get(key)
            if not target_col:
                continue

            cell = ws.cell(row=current_insert_row, column=target_col)

            if key == "serial_number":
                cell.value = max_sno
                cell.alignment = Alignment(horizontal="center", vertical="center")
            else:
                val_str = _resolve_record_value(record, key)

                if key in ("mobile_number", "alternate_number", "other_phone_numbers", "pin_code"):
                    cell.number_format = "@"
                    cell.value = val_str
                    cell.alignment = Alignment(
                        horizontal="center" if key == "pin_code" else "left",
                        vertical="center"
                    )
                else:
                    cell.value = val_str
                    cell.alignment = Alignment(horizontal="left", vertical="center")

            cell.font = REGULAR_FONT
            cell.border = BORDER_THIN
            if is_even:
                cell.fill = ZEBRA_FILL

        current_insert_row += 1

    ws.auto_filter.ref = ws.dimensions
    _auto_adjust_column_widths(ws)

    output = io.BytesIO()
    wb.save(output)
    output.seek(0)
    return output, len(new_records)
