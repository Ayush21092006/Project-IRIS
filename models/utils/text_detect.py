import easyocr

reader = easyocr.Reader(['en'])

def detect_text(image):
    results = reader.readtext(image)
    if not results:
        return []
    texts = [text[1] for text in results if len(text[1].strip()) > 0]
    return texts if texts else []
