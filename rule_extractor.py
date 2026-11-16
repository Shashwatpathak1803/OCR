"""
Deterministic Multilingual (Hindi + English) Rule-Based Extraction Module.
Uses Unicode script detection (Devanagari vs. Latin), regular expressions,
domain dictionaries, and Pydantic validation to extract and segregate
Hindi and English contact details without any external AI or translation APIs.
100% OFFLINE & DETERMINISTIC.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


# ==========================================
# 1. Pydantic Model for Multilingual Record
# ==========================================

class VisitingCardRecord(BaseModel):
    """
    Structured, validated data model supporting separate Hindi (Devanagari)
    and English (Latin) fields as well as shared contact attributes.
    """
    serial_number: int = 1

    # Name fields
    full_name_en: str = ""
    full_name_hi: str = ""
    first_name_en: str = ""
    first_name_hi: str = ""
    last_name_en: str = ""
    last_name_hi: str = ""

    # Professional & Organization fields
    designation_en: str = ""
    designation_hi: str = ""
    company_name_en: str = ""
    company_name_hi: str = ""
    department: str = ""

    # Contact numbers (language-neutral)
    mobile_number: str = ""
    alternate_number: str = ""
    other_phone_numbers: str = ""

    # Digital presence
    email: str = ""
    other_emails: str = ""
    website: str = ""
    linkedin: str = ""
    other_socials: str = ""

    # Address components
    address_en: str = ""
    address_hi: str = ""
    city_en: str = ""
    city_hi: str = ""
    state_en: str = ""
    state_hi: str = ""
    country: str = ""
    pin_code: str = ""

    # Metadata & Quality
    image_filename: str = ""
    verification_needed: bool = False
    unverified_fields: List[str] = Field(default_factory=list)
    ocr_confidence: float = 0.0

    # Compatibility properties for existing workflows
    @property
    def full_name(self) -> str:
        if self.full_name_en and self.full_name_hi:
            return f"{self.full_name_en} / {self.full_name_hi}"
        return self.full_name_en or self.full_name_hi

    @property
    def designation(self) -> str:
        if self.designation_en and self.designation_hi:
            return f"{self.designation_en} / {self.designation_hi}"
        return self.designation_en or self.designation_hi

    @property
    def company_name(self) -> str:
        if self.company_name_en and self.company_name_hi:
            return f"{self.company_name_en} / {self.company_name_hi}"
        return self.company_name_en or self.company_name_hi

    @property
    def address(self) -> str:
        if self.address_en and self.address_hi:
            return f"{self.address_en} / {self.address_hi}"
        return self.address_en or self.address_hi

    @property
    def city(self) -> str:
        return self.city_en or self.city_hi

    @property
    def state(self) -> str:
        return self.state_en or self.state_hi

    @field_validator("email", "other_emails", mode="before")
    @classmethod
    def clean_email(cls, v: Any) -> str:
        return str(v).strip().lower() if v else ""

    @field_validator("mobile_number", "alternate_number", "other_phone_numbers", "pin_code", mode="before")
    @classmethod
    def clean_digits_str(cls, v: Any) -> str:
        return str(v).strip() if v is not None else ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict including both specific language fields and legacy fallbacks."""
        d = self.model_dump()
        # Add legacy fallback keys for compatibility
        d["full_name"] = self.full_name
        d["first_name"] = self.first_name_en or self.first_name_hi
        d["last_name"] = self.last_name_en or self.last_name_hi
        d["designation"] = self.designation
        d["company_name"] = self.company_name
        d["address"] = self.address
        d["city"] = self.city
        d["state"] = self.state
        return d


FIELD_KEYS = [
    "full_name_en", "full_name_hi",
    "first_name_en", "first_name_hi",
    "last_name_en", "last_name_hi",
    "designation_en", "designation_hi",
    "company_name_en", "company_name_hi",
    "department",
    "mobile_number", "alternate_number", "other_phone_numbers",
    "email", "other_emails", "website",
    "address_en", "address_hi",
    "city_en", "city_hi",
    "state_en", "state_hi",
    "country", "pin_code",
    "linkedin", "other_socials",
]


# ==========================================
# 2. Unicode Script Detection Helpers
# ==========================================

DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")
LATIN_REGEX = re.compile(r"[a-zA-Z]")


def has_devanagari(text: str) -> bool:
    """Return True if text contains any Devanagari (Hindi) character."""
    return bool(DEVANAGARI_REGEX.search(text))


def has_latin(text: str) -> bool:
    """Return True if text contains any Latin (English) character."""
    return bool(LATIN_REGEX.search(text))


def segment_line_by_script(line: str) -> Tuple[str, str]:
    """
    Split a potentially mixed-language line into (english_part, hindi_part).
    If purely English, returns (line, "").
    If purely Hindi, returns ("", line).
    If mixed, intelligently clusters words by script without losing characters.
    """
    has_dev = has_devanagari(line)
    has_lat = has_latin(line)

    if has_dev and not has_lat:
        return "", line.strip()
    if has_lat and not has_dev:
        return line.strip(), ""

    en_chars: List[str] = []
    hi_chars: List[str] = []
    active_script = ""
    pending = ""

    for char in line:
        if DEVANAGARI_REGEX.fullmatch(char):
            script = "hi"
        elif LATIN_REGEX.fullmatch(char):
            script = "en"
        else:
            script = ""

        if script:
            if not active_script and pending:
                (hi_chars if script == "hi" else en_chars).append(pending)
                pending = ""
            active_script = script
            (hi_chars if script == "hi" else en_chars).append(char)
        elif active_script:
            (hi_chars if active_script == "hi" else en_chars).append(char)
        else:
            pending += char

    if pending:
        en_chars.append(pending)

    return "".join(en_chars).strip(), "".join(hi_chars).strip()


# ==========================================
# 3. Multilingual Dictionaries & Patterns
# ==========================================

# English States and UTs
INDIAN_STATES_EN = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
    "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram",
    "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu",
    "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Delhi", "New Delhi", "Chandigarh", "Puducherry", "Jammu and Kashmir", "Ladakh"
]

# Hindi States and UTs
INDIAN_STATES_HI = [
    "उत्तर प्रदेश", "मध्य प्रदेश", "महाराष्ट्र", "राजस्थान", "गुजरात", "बिहार",
    "हरियाणा", "पंजाब", "कर्नाटक", "तमिलनाडु", "पश्चिम बंगाल", "दिल्ली", "नई दिल्ली",
    "उत्तराखंड", "झारखंड", "छत्तीसगढ़", "आंध्र प्रदेश", "तेलंगाना", "केरल", "गोवा",
    "हिमाचल प्रदेश", "चंडीगढ़", "असम", "ओडिशा"
]

# English Major Cities
INDIAN_CITIES_EN = [
    "Mumbai", "Delhi", "New Delhi", "Bengaluru", "Bangalore", "Hyderabad",
    "Ahmedabad", "Chennai", "Kolkata", "Surat", "Pune", "Jaipur", "Lucknow",
    "Kanpur", "Nagpur", "Indore", "Thane", "Bhopal", "Visakhapatnam", "Patna",
    "Vadodara", "Ghaziabad", "Ludhiana", "Agra", "Nashik", "Faridabad", "Meerut",
    "Rajkot", "Varanasi", "Srinagar", "Amritsar", "Navi Mumbai", "Allahabad",
    "Prayagraj", "Ranchi", "Coimbatore", "Vijayawada", "Jodhpur", "Madurai",
    "Raipur", "Kota", "Guwahati", "Chandigarh", "Mysore", "Gurugram", "Gurgaon",
    "Noida", "Greater Noida", "Bhubaneswar", "Dehradun", "Kochi", "Udaipur"
]

# Hindi Major Cities
INDIAN_CITIES_HI = [
    "नोएडा", "ग्रेटर नोएडा", "जयपुर", "दिल्ली", "नई दिल्ली", "मुंबई", "बेंगलुरु",
    "हैदराबाद", "अहमदाबाद", "चेन्नई", "कोलकाता", "सूरत", "पुणे", "लखनऊ", "कानपुर",
    "नागपुर", "इंदौर", "भोपाल", "पटना", "वडोदरा", "गाजियाबाद", "लुधियाना", "आगरा",
    "नासिक", "फरीदाबाद", "मेरठ", "वाराणसी", "प्रयागराज", "इलाहाबाद", "रांची", "गुवाहाटी",
    "चंडीगढ़", "देहरादून", "उदयपुर", "जोधपुर", "कोटा", "गुरुग्राम", "गुड़गांव"
]

# Designation Keywords (English)
DESIGNATION_KEYWORDS_EN = [
    "director", "managing director", "executive director", "associate director",
    "chief executive officer", "ceo", "chief technology officer", "cto",
    "chief operating officer", "coo", "chief financial officer", "cfo",
    "president", "vice president", "vp", "general manager", "gm",
    "founder", "co-founder", "partner", "managing partner",
    "manager", "senior manager", "assistant manager", "project manager",
    "consultant", "senior consultant", "principal consultant",
    "engineer", "software engineer", "senior engineer", "lead engineer",
    "developer", "lead developer", "full stack developer",
    "architect", "solutions architect", "enterprise architect", "ux architect",
    "specialist", "coordinator", "officer", "executive", "associate",
    "advocate", "lawyer", "attorney", "counsel",
    "professor", "prof.", "dean", "lecturer",
    "analyst", "senior analyst", "researcher", "scientist",
    "designer", "creative director", "art director", "ui/ux designer",
    "head of", "head -", "team lead", "sales manager", "marketing manager"
]

# Designation Keywords (Hindi)
DESIGNATION_KEYWORDS_HI = [
    "सेल्स मैनेजर", "प्रबंधक", "महाप्रबंधक", "निदेशक", "प्रबंध निदेशक", "प्रबंध नदेशक",
    "कार्यकारी निदेशक", "संस्थापक", "सह-संस्थापक", "अध्यक्ष", "उपाध्यक्ष",
    "सलाहकार", "वरिष्ठ सलाहकार", "अभियंता", "इंजीनियर", "वरिष्ठ अभियंता",
    "अधिकारी", "मुख्य कार्यकारी अधिकारी", "विशेषज्ञ", "वकील", "अधिवक्ता",
    "सीईओ", "सीटीओ", "सीआईओ", "प्रधान", "अग्रणी", "विश्लेषक", "कला निर्देशक",
    "डिज़ाइनर", "कार्यकारी", "प्रमुख", "सहयोगी", "प्रबंध", "मैनेजर", "डायरेक्टर"
]

# Company Suffixes (English)
COMPANY_SUFFIXES_EN = [
    "pvt ltd", "pvt. ltd.", "private limited", "ltd", "ltd.", "limited",
    "inc", "inc.", "corp", "corp.", "corporation", "llc", "llp",
    "technologies", "technology", "solutions", "services", "enterprises",
    "group", "industries", "holdings", "consulting", "software", "labs",
    "studio", "studios", "agency", "ventures", "logistics", "transports",
    "media", "infotech", "systems", "associates", "works", "pharmaceuticals"
]

# Company Suffixes (Hindi)
COMPANY_SUFFIXES_HI = [
    "प्रा. लि.", "प्रा.लि.", "प्राइवेट लिमिटेड", "लिमिटेड", "लि.",
    "इंटरप्राइजेज", "इंटरप्राइजिज", "सॉल्यूशंस", "सॉल्यूशन्स",
    "टेक्नोलॉजीज", "टेक्नोलॉजिस", "सर्विसेज", "उद्योग", "समूह",
    "ग्रुप", "कंपनी", "कॉर्पोरेशन", "इंस्टीट्यूट", "लॉजिस्टिक्स",
    "ट्रांसपोर्ट", "मीडिया", "लैब्स", "स्टूडियो", "एसोसिएट्स",
    "फर्म", "प्रतिष्ठान"
]

# Address Terms & Labels
ADDR_TERMS_EN = [
    "road", "street", "st.", "rd.", "ave", "avenue", "floor", "plot", "sector",
    "block", "opp", "near", "nagar", "marg", "building", "tower", "lane",
    "phase", "complex", "bldg", "cross", "bypass", "highway", "dist.", "pin:",
    "pin code", "pin", "india", "p.o.", "post", "district", "city", "address",
    "office", "branch", "regd office", "corp office"
]

ADDR_TERMS_HI = [
    "पता", "कार्यालय", "शाखा", "प्लॉट", "सेक्टर", "मार्ग", "रोड", "रास्ता",
    "गली", "चौक", "नगर", "भवन", "टावर", "मंजिल", "अपार्टमेंट", "परिसर",
    "ब्लॉक", "तहसील", "जिला", "पिन:", "पिन कोड:", "भारत", "डाकघर"
]

# Phone & URL Patterns
EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", re.IGNORECASE)
URL_REGEX = re.compile(
    r"(?:https?://[^\s/$.?#].[^\s]*|www\.[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|[a-zA-Z0-9-]+\.(?:com|org|net|io|in|co\.in|co|uk|ai|tech|edu|gov|biz|info|me)(?:/[^\s]*)?)",
    re.IGNORECASE,
)
PHONE_PATTERNS = [
    re.compile(r"(?:\+91[\s.-]?)?(?:0)?[6-9]\d{4}[\s.-]?\d{5}"),
    re.compile(r"(?:\+91[\s.-]?)?\(?0\d{2,4}\)?[\s.-]?\d{6,8}"),
    re.compile(r"\+\d{1,3}[\s.-]?(?:\(?\d{2,5}\)?[\s.-]?)?\d{3,5}[\s.-]?\d{3,5}"),
]
PIN_CODE_REGEX = re.compile(r"\b[1-9]\d{5}\b|\b\d{5}(?:-\d{4})?\b")


# ==========================================
# 4. Extraction Logic & Segmentation
# ==========================================

def clean_line(text: str) -> str:
    """Normalize spaces and remove leading bullet artifacts."""
    cleaned = re.sub(r"\s+", " ", str(text)).strip()
    return re.sub(r"^[|\-•*~:>]+\s*", "", cleaned).strip()


def extract_phones(lines: List[str]) -> Tuple[str, str, str]:
    """Extract mobile, alternate, and other phone numbers."""
    mobiles, landlines, others = [], [], []

    for line in lines:
        cleaned = clean_line(line)
        lower = cleaned.lower()

        is_mob = any(re.search(r"\b" + re.escape(p) + r"\b", lower) for p in ["mobile", "mob", "cell", "m:", "whatsapp", "मो.", "मोबाइल"])
        is_tel = any(re.search(r"\b" + re.escape(p) + r"\b", lower) for p in ["tel", "phone", "ph", "t:", "office", "off", "फोन", "दूरभाष"])
        is_fax = "fax" in lower or "फैक्स" in lower

        for pattern in PHONE_PATTERNS:
            for match in pattern.finditer(cleaned):
                cand = match.group(0).strip(".,;: ")
                digits = re.sub(r"\D", "", cand)

                # Skip standalone PINs
                if len(digits) == 6 and not (cand.startswith("+") or is_mob or is_tel):
                    continue

                if 8 <= len(digits) <= 15:
                    if is_fax:
                        if cand not in others:
                            others.append(cand)
                    elif is_mob:
                        if cand not in mobiles:
                            mobiles.append(cand)
                    elif is_tel:
                        if cand not in landlines:
                            landlines.append(cand)
                    else:
                        if digits.startswith("91") and len(digits) >= 12:
                            first_cell = digits[2]
                        elif digits.startswith("0") and len(digits) >= 11:
                            first_cell = digits[1]
                        else:
                            first_cell = digits[0] if digits else ""

                        if first_cell in ("6", "7", "8", "9") and len(digits) == 10:
                            if cand not in mobiles:
                                mobiles.append(cand)
                        else:
                            if cand not in landlines and cand not in mobiles:
                                landlines.append(cand)

    primary_mobile = mobiles[0] if mobiles else (landlines[0] if landlines else (others[0] if others else ""))
    alt_phone = ""
    if mobiles and len(mobiles) > 1:
        alt_phone = mobiles[1]
    elif landlines and landlines[0] != primary_mobile:
        alt_phone = landlines[0]
    elif others and others[0] != primary_mobile:
        alt_phone = others[0]

    all_remaining = [p for p in (mobiles + landlines + others) if p != primary_mobile and p != alt_phone]
    other_phones = ", ".join(dict.fromkeys(all_remaining))
    return primary_mobile, alt_phone, other_phones


def extract_visiting_card_data(
    ocr_text: str,
    lines_info: Optional[List[Dict[str, Any]]] = None,
    image_filename: str = "",
    serial_number: int = 1,
) -> VisitingCardRecord:
    """
    Multilingual Deterministic Extractor for English and Hindi.
    Extracts contact info, identifies Devanagari vs Latin characters,
    and segregates data into separate Hindi and English fields.
    """
    raw_lines = [clean_line(l) for l in ocr_text.splitlines() if clean_line(l)]
    unverified_fields: List[str] = []

    # Calculate average OCR confidence
    avg_conf = 0.0
    if lines_info:
        confs = [l.get("confidence", 0.0) for l in lines_info if "confidence" in l]
        avg_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    # 1. Emails
    all_emails = EMAIL_REGEX.findall(ocr_text)
    clean_emails = []
    for em in all_emails:
        c = em.strip(".,;:<> ").lower()
        if c and c not in clean_emails:
            clean_emails.append(c)
    primary_email = clean_emails[0] if clean_emails else ""
    other_emails = ", ".join(clean_emails[1:]) if len(clean_emails) > 1 else ""

    # 2. Websites and Socials
    website = ""
    linkedin = ""
    other_socials_list = []
    for m in URL_REGEX.findall(ocr_text):
        clean_url = m.strip(".,;:<> ()[]\t\n")
        if "@" in clean_url or any(clean_url in em for em in clean_emails):
            continue
        lower_url = clean_url.lower()
        if "linkedin.com" in lower_url:
            if not linkedin:
                linkedin = clean_url
        elif any(s in lower_url for s in ["twitter.com", "x.com", "facebook.com", "instagram.com", "github.com"]):
            if clean_url not in other_socials_list:
                other_socials_list.append(clean_url)
        else:
            if not website and "." in clean_url:
                website = clean_url
    other_socials = ", ".join(other_socials_list)

    # 3. Phone Numbers
    mobile_num, alt_num, other_nums = extract_phones(raw_lines)

    # 4. Segment remaining lines into English pool and Hindi pool
    en_lines = []
    hi_lines = []
    addr_en_parts = []
    addr_hi_parts = []

    for line in raw_lines:
        lower = line.lower()
        # Skip pure digital contact lines
        if primary_email and primary_email in lower:
            continue
        if website and website.lower() in lower and len(line) <= len(website) + 8:
            continue
        if linkedin and linkedin.lower() in lower:
            continue
        if mobile_num and mobile_num in line and len(line) <= len(mobile_num) + 10:
            continue

        # Split line if mixed language
        part_en, part_hi = segment_line_by_script(line)

        # Check English part for address
        if part_en:
            lower_en = part_en.lower()
            if any(term in lower_en for term in ADDR_TERMS_EN) or any(s.lower() in lower_en for s in INDIAN_STATES_EN):
                addr_en_parts.append(part_en)
            else:
                en_lines.append(part_en)

        # Check Hindi part for address
        if part_hi:
            if any(term in part_hi for term in ADDR_TERMS_HI) or any(s in part_hi for s in INDIAN_STATES_HI):
                addr_hi_parts.append(part_hi)
            else:
                hi_lines.append(part_hi)

    # 5. Extract English Entities (Designation, Company, Name)
    desig_en = ""
    company_en = ""
    name_en = ""
    first_en = ""
    last_en = ""

    # Designation (English)
    rem_en_1 = []
    for line in en_lines:
        lower = line.lower()
        matched = False
        if not desig_en:
            for kw in DESIGNATION_KEYWORDS_EN:
                if re.search(r"\b" + re.escape(kw) + r"\b", lower) or (len(kw) >= 5 and kw in lower):
                    desig_en = line
                    matched = True
                    break
        if not matched:
            rem_en_1.append(line)

    # Company (English)
    rem_en_2 = []
    for line in rem_en_1:
        lower = line.lower()
        matched = False
        if not company_en:
            for sfx in COMPANY_SUFFIXES_EN:
                if re.search(r"\b" + re.escape(sfx) + r"\b", lower) or sfx in lower:
                    company_en = line
                    matched = True
                    break
        if not matched:
            rem_en_2.append(line)

    # Full Name (English)
    for idx, line in enumerate(rem_en_2):
        if any(c.isdigit() for c in line) or "@" in line:
            continue
        words = line.split()
        if 1 <= len(words) <= 4:
            alpha_words = [w for w in words if w.isalpha()]
            if alpha_words and all(w.isupper() or w.istitle() for w in alpha_words):
                name_en = line
                first_en = words[0]
                if len(words) > 1:
                    last_en = " ".join(words[1:])
                rem_en_2.pop(idx)
                break

    if not name_en and rem_en_2:
        for idx, line in enumerate(rem_en_2):
            if not any(c.isdigit() for c in line) and len(line.split()) <= 4:
                name_en = line
                words = line.split()
                first_en = words[0]
                if len(words) > 1:
                    last_en = " ".join(words[1:])
                rem_en_2.pop(idx)
                break

    if not company_en and rem_en_2:
        for line in rem_en_2:
            if len(line.split()) <= 6:
                company_en = line
                break

    # 6. Extract Hindi Entities (Designation, Company, Name)
    desig_hi = ""
    company_hi = ""
    name_hi = ""
    first_hi = ""
    last_hi = ""

    # Designation (Hindi)
    rem_hi_1 = []
    for line in hi_lines:
        matched = False
        if not desig_hi:
            for kw in DESIGNATION_KEYWORDS_HI:
                if kw in line:
                    desig_hi = line
                    matched = True
                    break
        if not matched:
            rem_hi_1.append(line)

    # Company (Hindi)
    rem_hi_2 = []
    for line in rem_hi_1:
        matched = False
        if not company_hi:
            for sfx in COMPANY_SUFFIXES_HI:
                if sfx in line:
                    company_hi = line
                    matched = True
                    break
        if not matched:
            rem_hi_2.append(line)

    # Full Name (Hindi)
    for idx, line in enumerate(rem_hi_2):
        words = line.split()
        if 1 <= len(words) <= 4:
            name_hi = line
            first_hi = words[0]
            if len(words) > 1:
                last_hi = " ".join(words[1:])
            rem_hi_2.pop(idx)
            break

    if not company_hi and rem_hi_2:
        for line in rem_hi_2:
            if len(line.split()) <= 6:
                company_hi = line
                break

    # 7. Address, City, State, Country, PIN Components
    addr_en = ", ".join(dict.fromkeys(addr_en_parts)).strip()
    addr_hi = ", ".join(dict.fromkeys(addr_hi_parts)).strip()

    city_en = ""
    state_en = ""
    city_hi = ""
    state_hi = ""
    country = ""
    pin_code = ""

    # PIN Code (check across entire text)
    all_phones = [p for p in [mobile_num, alt_num, other_nums] if p]
    for pin in PIN_CODE_REGEX.findall(ocr_text):
        if not any(pin in p for p in all_phones):
            pin_code = pin
            break

    # English City & State
    search_en = f"{addr_en}\n{ocr_text}"
    for city in INDIAN_CITIES_EN:
        if re.search(r"\b" + re.escape(city) + r"\b", search_en, re.IGNORECASE):
            city_en = city
            break
    for st in INDIAN_STATES_EN:
        if re.search(r"\b" + re.escape(st) + r"\b", search_en, re.IGNORECASE):
            state_en = st
            break

    # Hindi City & State
    search_hi = f"{addr_hi}\n{ocr_text}"
    for city in INDIAN_CITIES_HI:
        if city in search_hi:
            city_hi = city
            break
    for st in INDIAN_STATES_HI:
        if st in search_hi:
            state_hi = st
            break

    # Country
    if "भारत" in ocr_text or "इंडिया" in ocr_text or "india" in ocr_text.lower():
        country = "India"
    elif "usa" in ocr_text.lower() or "united states" in ocr_text.lower():
        country = "USA"
    elif "uk" in ocr_text.lower() or "united kingdom" in ocr_text.lower():
        country = "UK"
    elif state_en or state_hi or city_en or city_hi:
        country = "India"

    # 8. Verification Checks
    has_any_name = bool(name_en or name_hi)
    has_any_desig = bool(desig_en or desig_hi)
    has_any_comp = bool(company_en or company_hi)

    if not has_any_name:
        unverified_fields.append("Full Name")
    if not mobile_num:
        unverified_fields.append("Mobile Number")
    if not primary_email:
        unverified_fields.append("Email Address")
    if not has_any_comp:
        unverified_fields.append("Company Name")
    if not has_any_desig:
        unverified_fields.append("Designation")

    needs_verification = len(unverified_fields) > 0 or avg_conf < 0.70

    return VisitingCardRecord(
        serial_number=serial_number,
        full_name_en=name_en,
        full_name_hi=name_hi,
        first_name_en=first_en,
        first_name_hi=first_hi,
        last_name_en=last_en,
        last_name_hi=last_hi,
        designation_en=desig_en,
        designation_hi=desig_hi,
        company_name_en=company_en,
        company_name_hi=company_hi,
        department="",
        mobile_number=mobile_num,
        alternate_number=alt_num,
        other_phone_numbers=other_nums,
        email=primary_email,
        other_emails=other_emails,
        website=website,
        address_en=addr_en,
        address_hi=addr_hi,
        city_en=city_en,
        city_hi=city_hi,
        state_en=state_en,
        state_hi=state_hi,
        country=country,
        pin_code=pin_code,
        linkedin=linkedin,
        other_socials=other_socials,
        image_filename=image_filename,
        verification_needed=needs_verification,
        unverified_fields=unverified_fields,
        ocr_confidence=avg_conf,
    )
