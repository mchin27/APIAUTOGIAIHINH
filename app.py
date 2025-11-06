from fastapi import FastAPI, UploadFile, File
import pytesseract
from PIL import Image, ImageOps
import openai, io, os

openai.api_key = os.getenv("OPENAI_API_KEY")  # thêm key OpenAI của bạn

app = FastAPI()

def preprocess(img):
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    img = img.resize((int(img.width * 1.5), int(img.height * 1.5)))
    return img

@app.post("/solve")
async def solve(image: UploadFile = File(...)):
    contents = await image.read()
    img = Image.open(io.BytesIO(contents))
    img = preprocess(img)

    ocr_text = pytesseract.image_to_string(img, lang="vie+eng")

    prompt = f"""
    Bạn là AI chuyên giải game 'Đuổi hình bắt chữ'. Hãy dựa trên mô tả:
    Ảnh có chữ: {ocr_text}
    Hãy suy luận xem đáp án hợp lý nhất là gì. Trả lời ngắn gọn.
    """

    res = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    answer = res["choices"][0]["message"]["content"].strip()
    return {"ocr_text": ocr_text, "suggested_answer": answer}
