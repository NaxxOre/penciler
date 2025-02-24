import re
import asyncio
import numpy as np
from io import BytesIO
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, MessageHandler, filters, ContextTypes
from google import genai

# Initialize Gemini client with your API key
client = genai.Client(api_key="YOUR_API_KEY")

async def process_handwriting(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Get the highest quality photo version
        photo = update.message.photo[-1]
        
        # Size validation (3MB limit)
        if photo.file_size > 3_000_000:
            await update.message.reply_text("⚠️ Image too large (max 3MB)")
            return

        # Download with timeout
        try:
            file = await photo.get_file()
            img_bytes = await asyncio.wait_for(file.download_as_bytearray(), timeout=10)
        except asyncio.TimeoutError:
            await update.message.reply_text("⌛ Download timed out")
            return

        # Enhanced image processing with Fourier Transform
        try:
            with Image.open(BytesIO(img_bytes)) as img:
                # Optimize image size
                if img.width > 800:
                    img = img.resize((800, int(img.height * (800/img.width)))
                
                # Convert to grayscale
                img_gray = img.convert('L')
                img_np = np.array(img_gray)

                # Fast Fourier Transform processing
                fft = np.fft.fft2(img_np)
                fshift = np.fft.fftshift(fft)
                
                # Create optimized low-pass filter
                rows, cols = img_np.shape
                crow, ccol = rows//2, cols//2
                mask = np.zeros((rows, cols), np.uint8)
                mask[crow-25:crow+25, ccol-25:ccol+25] = 1

                # Apply frequency domain filtering
                fshift *= mask
                img_back = np.fft.ifft2(np.fft.ifftshift(fshift)).real
                
                # Normalize and convert
                img_back = ((img_back - img_back.min()) / 
                          (img_back.max() - img_back.min()) * 255)
                enhanced_img = Image.fromarray(img_back.astype(np.uint8))

                # Save as optimized JPEG
                buffer = BytesIO()
                enhanced_img.save(buffer, format="JPEG", quality=85)
                image_data = buffer.getvalue()

        except Exception as e:
            print(f"🖼️ Image processing error: {e}")
            image_data = img_bytes  # Fallback to original

        # Gemini API call with your credentials
        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[
                    {"mime_type": "image/jpeg", "data": image_data},
                    "Extract handwritten text EXACTLY as written. Preserve line breaks and punctuation."
                ]
            )
            
            # Process and truncate text for Telegram
            text = response.text.strip()
            if len(text) > 3500:
                text = text[:3500] + "\n... (truncated)"
            
            # Send response with retry button
            await update.message.reply_text(
                f"✍️ Extracted Text:\n\n{text}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Retry", callback_data="retry")]
                )
            )

        except Exception as e:
            await update.message.reply_text(f"🔴 Recognition error: {str(e)[:300]}")

    except Exception as e:
        await update.message.reply_text(f"⛔ Critical error: {str(e)[:300]}")

if __name__ == "__main__":
    # Initialize with your bot token
    application = Application.builder().token("YOUR_BOT_TOKEN").build()
    application.add_handler(MessageHandler(filters.PHOTO, process_handwriting))
    application.run_polling()