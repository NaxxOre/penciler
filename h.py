import re
import asyncio
import numpy as np
from io import BytesIO
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, ApplicationBuilder, MessageHandler, filters, ContextTypes
from google import genai

# Initialize Gemini client
client = genai.Client(api_key="AIzaSyCerjxXg_b6AAW4sQEq2Tzxo_sXV40dkOI")

async def process_handwriting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Get the photo with size check
        photo = update.message.photo[-1]
        if photo.file_size > 5_000_000:
            await update.message.reply_text("❌ Image too large (max 5MB)")
            return

        # Download with timeout
        try:
            file = await photo.get_file()
            img_bytes = await asyncio.wait_for(file.download_as_bytearray(), timeout=15)
        except asyncio.TimeoutError:
            await update.message.reply_text("❌ Download timed out")
            return

        # Fast Fourier Transform processing
        try:
            img = Image.open(BytesIO(img_bytes)).convert('L')
            img_np = np.array(img)
            
            # FFT operations
            f = np.fft.fft2(img_np)
            fshift = np.fft.fftshift(f)
            
            # Create low-pass filter
            rows, cols = img_np.shape
            crow, ccol = rows//2, cols//2
            mask = np.zeros((rows, cols), np.uint8)
            mask[crow-30:crow+30, ccol-30:ccol+30] = 1
            
            # Apply filter and inverse FFT
            fshift *= mask
            img_back = np.fft.ifft2(np.fft.ifftshift(fshift)).real
            img_back = ((img_back - img_back.min()) * (255/(img_back.max()-img_back.min()))).astype(np.uint8)
            
            enhanced_img = Image.fromarray(img_back)
            buffer = BytesIO()
            enhanced_img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
            
        except Exception as e:
            print(f"Image processing failed: {e}")
            image_data = img_bytes

        # Gemini API call
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    {"mime_type": "image/png", "data": image_data},
                    "Extract text exactly as written, preserve line breaks:"
                ]
            )
            text = response.text.strip()
            
            if len(text) > 4000:
                text = text[:4000] + "\n... (truncated)"
                
            keyboard = [[InlineKeyboardButton("Try Again", callback_data='retry')]]
            await update.message.reply_text(
                f"✍️ Extracted Text:\n\n{text}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            
        except Exception as e:
            await update.message.reply_text(f"❌ Recognition failed: {str(e)}")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: {str(e)[:1000]}")

if __name__ == "__main__":
    application = ApplicationBuilder().token("7964593670:AAFEaf5IAQP60pUAKb5kPyDp73oCKz5PmM8").build()
    application.add_handler(MessageHandler(filters.PHOTO, process_handwriting))
    application.run_polling()