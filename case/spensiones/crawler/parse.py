import re, unicodedata
MONTHS = {"enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
          "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12}
_MN = "|".join(MONTHS)


def parse_date(text):
    """Return (iso, precision) or (None, None). Precision: day|month|year."""
    t = unicodedata.normalize("NFC", text or "").lower()
    m = re.search(rf"(\d{{1,2}})\s*(?:º|°)?\s+de\s+({_MN})\s+(?:de|del)\s+(\d{{4}})", t)
    if m:
        return f"{int(m[3]):04d}-{MONTHS[m[2]]:02d}-{int(m[1]):02d}", "day"
    m = re.search(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-]((?:19|20)\d{2})(?!\d)", t)
    if m and 1 <= int(m[2]) <= 12 and 1 <= int(m[1]) <= 31:
        return f"{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}", "day"
    m = re.search(rf"\b({_MN})\s+(?:de|del)?\s*((?:19|20)\d{{2}})\b", t)
    if m:
        return f"{int(m[2]):04d}-{MONTHS[m[1]]:02d}", "month"
    m = re.search(r"(?<![\d.,])((?:19|20)\d{2})(?![\d])", t)
    if m:
        return m[1], "year"
    return None, None


def extract_number(title):
    m = re.search(r"N\s*[°º\.]\s*([\d][\d\.]*(?:[-/]\d+)?)", title or "", re.I)
    return m[1].rstrip(".") if m else None


def classify(title, section_path):
    s = (section_path or "").lower()
    t = (title or "").lower()
    # section-driven first
    if "resoluciones" in s and "resoluci" in t:
        return "resolucion"
    if re.match(r"\s*oficio", t):
        return "oficio"
    if re.match(r"\s*(resoluci[oó]n|res\.)", t):
        return "resolucion"
    if re.match(r"\s*(decreto|d\.s\.|d\.f\.l|dfl|d\.l\.)", t):
        return "decreto"
    if re.match(r"\s*reglamento", t) or "reglamento" in t[:60]:
        return "reglamento"
    if re.match(r"\s*(ley|texto|constituci)", t) or "legislaci" in s or "leyes" in s:
        return "ley"
    if "convenio" in s or re.match(r"\s*convenio", t):
        return "convenio"
    if "licitaci" in s:
        return "licitacion"
    if "consulta" in s and "normativa" in s:
        return "normativa_consulta"
    if "ficha" in s or re.match(r"\s*ficha", t):
        return "ficha"
    if "minuta" in t or "consejo t" in s:
        return "minuta"
    if "cuenta p" in s or "balance de gesti" in s or "bgi" in s or "memoria" in t:
        return "memoria"
    if "presentaci" in s or "presentaci" in t[:40]:
        return "presentacion"
    if "bolet" in s or "bolet" in t[:30]:
        return "boletin"
    if any(k in s for k in ("documentos de trabajo", "notas t", "notas ciudadanas", "libros", "panorama", "consultor", "ocde")):
        return "estudio"
    if "informe" in s or re.match(r"\s*informe", t) or "rentabilidad" in s or "compare" in s or "comisiones" in s or "inversi" in s:
        return "informe"
    if "atenci" in s and "usuarios" in s or "educaci" in s or "perfil" in s:
        return "guia_faq"
    if "normativa" in s:
        return "norma"
    if "circular" in t[:30]:
        return "circular"
    return "otro"


def classify_area(text):
    s = (text or "").lower()
    if "cesant" in s or "afc" in s:
        return "cesantia"
    return None
