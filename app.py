import re
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import requests
from pathlib import Path  # Added for temporary file handling
import pytesseract  # Added for OCR
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ConversationHandler,
    filters
)

# Conversation states
SELECT_OPTION, ASK_BAND, ASK_TOPIC, PROCESS_ESSAY, HANDWRITING_UPLOAD = range(5)

# Band-specific essay guidelines (unchanged)
BAND_INSTRUCTIONS = {
    9: {'vocab': "Sophisticated terms (paradigm shift, socioeconomic)", 'structure': "Complex sentences with subordinate clauses", 'cohesion': "Advanced transitions (consequently, furthermore)", 'errors': "Virtually error-free", 'length': "300+ words"},
    8: {'vocab': "Advanced terms (globalization, contemporary)", 'structure': "Varied complex structures", 'cohesion': "Effective linking (however, moreover)", 'errors': "Rare minor errors", 'length': "280+ words"},
    7: {'vocab': "Adequate range (significant, development)", 'structure': "Mix of simple/complex sentences", 'cohesion': "Clear paragraphing", 'errors': "Some errors", 'length': "250+ words"},
    6: {'vocab': "Basic academic terms", 'structure': "Simple structures with some complexity", 'cohesion': "Basic connectors (and, but)", 'errors': "Noticeable errors", 'length': "250 words"},
    5: {'vocab': "Limited range (good, things)", 'structure': "Mostly simple sentences", 'cohesion': "Few connectors", 'errors': "Frequent errors", 'length': "200 words"},
    4: {'vocab': "Basic vocabulary (school, job)", 'structure': "Short simple sentences", 'cohesion': "Minimal linking", 'errors': "Systematic errors", 'length': "150 words"},
    3: {'vocab': "Very basic terms (study, work)", 'structure': "Fragmented sentences", 'cohesion': "No connectors", 'errors': "Severe errors", 'length': "<100 words"}
}

async def start(update: Update, context):
    keyboard = [
        [InlineKeyboardButton("Generate Essay", callback_data='generate')],
        [InlineKeyboardButton("Analyze Essay", callback_data='analyze')],
        [InlineKeyboardButton("Handwriting Recognition", callback_data='handwriting')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("📝 IELTS Essay Assistant:", reply_markup=reply_markup)
    return SELECT_OPTION

async def select_option(update: Update, context):
    query = update.callback_query
    await query.answer()
    if query.data == 'generate':
        await query.edit_message_text(text="📊 Enter desired band (3-9):")
        return ASK_BAND
    elif query.data == 'analyze':
        await query.edit_message_text(text="📄 Paste your essay for analysis:")
        return PROCESS_ESSAY
    elif query.data == 'handwriting':
        await query.edit_message_text(text="📸 Upload a clear photo of your handwritten text:")
        return HANDWRITING_UPLOAD
    return SELECT_OPTION

# Updated process_handwriting function (using Tesseract OCR, as above)
async def process_handwriting(update: Update, context):
    try:
        # Get the highest resolution photo
        photo = update.message.photo[-1]
        if photo.file_size > 5_000_000:  # Check file size (max 5MB)
            await update.message.reply_text("❌ Image too large (max 5MB)")
            return SELECT_OPTION

        # Download image with timeout
        file = await photo.get_file()
        try:
            img_bytes = await asyncio.wait_for(file.download_as_bytearray(), timeout=15)
        except asyncio.TimeoutError:
            await update.message.reply_text("❌ Download timed out")
            return SELECT_OPTION

        # Apply Fourier Transform for image enhancement
        try:
            img = Image.open(BytesIO(img_bytes)).convert('L')  # Grayscale
            img_np = np.array(img)
            f = np.fft.fft2(img_np)
            fshift = np.fft.fftshift(f)
            rows, cols = img_np.shape
            crow, ccol = rows // 2, cols // 2
            mask = np.ones((rows, cols), np.uint8)
            r = 30  # Adjust radius for sharper edges
            mask[crow-r:crow+r, ccol-r:ccol+r] = 0  # High-pass filter
            fshift_filtered = fshift * mask
            f_ishift = np.fft.ifftshift(fshift_filtered)
            img_back = np.fft.ifft2(f_ishift)
            img_back = np.abs(img_back)
            img_back = ((img_back - img_back.min()) * 255 / (img_back.max() - img_back.min())).astype(np.uint8)
            enhanced_img = Image.fromarray(img_back)
        except Exception as e:
            print(f"Image processing error: {e}")
            enhanced_img = Image.open(BytesIO(img_bytes))  # Fallback to original

        # Save enhanced image temporarily for Tesseract
        import pytesseract
        from pathlib import Path
        temp_path = Path("temp_image.png")
        enhanced_img.save(temp_path, format="PNG")

        # Extract text using Tesseract with PSM 6 (Assume single uniform block of text)
        try:
            # Configure Tesseract for better handwritten/table recognition
            custom_config = r'--oem 3 --psm 6'  # OEM 3: Default (LSTM-based), PSM 6: Assume a single uniform block of text
            extracted_text = pytesseract.image_to_string(enhanced_img, config=custom_config)
            if not extracted_text or extracted_text.strip() == "":
                extracted_text = "No text detected."

            # Truncate text to fit Telegram's 4096 character limit
            TELEGRAM_MAX_LENGTH = 4096
            header = "✍️ Extracted Text:\n\n"
            max_text_length = TELEGRAM_MAX_LENGTH - len(header) - 50  # Leave room for truncation notice
            if len(extracted_text) > max_text_length:
                extracted_text = extracted_text[:max_text_length] + "\n\n[Text truncated due to length limit]"

            # Debugging: Print or log the extracted text for inspection
            print(f"Extracted text: {extracted_text[:500]}...")  # Print first 500 chars for debugging

            keyboard = [[InlineKeyboardButton("Restart", callback_data='restart')]]
            await update.message.reply_text(
                f"{header}{extracted_text}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        except Exception as e:
            await update.message.reply_text(f"❌ Error processing handwriting: {str(e)[:1000]}")
        finally:
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()

    except Exception as e:
        await update.message.reply_text(f"❌ General error: {str(e)[:1000]}")

    return SELECT_OPTION

async def ask_band(update: Update, context):
    try:
        band = int(update.message.text)
        if 3 <= band <= 9:
            context.user_data['band'] = band
            await update.message.reply_text("📌 Enter essay topic:")
            return ASK_TOPIC
        await update.message.reply_text("❌ Invalid band. Enter 3-9:")
        return ASK_BAND
    except ValueError:
        await update.message.reply_text("❌ Numbers only (3-9):")
        return ASK_BAND

async def ask_topic(update: Update, context):
    context.user_data['topic'] = update.message.text
    band = context.user_data['band']
    essay = await generate_essay(context.user_data['topic'], band)
    await loading(update)
    keyboard = [[InlineKeyboardButton("Restart", callback_data='restart')]]
    await update.message.reply_text(
        f"📝 Band {band} Essay:\n\n{essay}",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    context.user_data['current_essay'] = essay
    return SELECT_OPTION

async def generate_essay(topic, band):
    instructions = BAND_INSTRUCTIONS[band]
    prompt = f"""Write an IELTS Band {band} essay on the topic: {topic}.
STRICTLY FOLLOW THESE GUIDELINES:
- Use vocabulary suitable for Band {band}: {instructions['vocab']}
- Sentence structure must match Band {band}: {instructions['structure']}
- Cohesion must be appropriate: {instructions['cohesion']}
- Length requirement: {instructions['length']}
- Error level: {instructions['errors']}
- Ensure connector count is at least 12 for Band 8, 10 for Band 7, and 8 for Band 6.
- Ensure repeated words are less than 3 for Band 8, less than 5 for Band 7, and less than 7 for Band 6.
- DO NOT exceed the expected complexity for Band {band}.
- Output: Plain text only. Do not include explanations or additional comments."""
    try:
        # Removed google-generativeai dependency; you can keep this as is or implement another AI (e.g., OpenAI)
        # For now, return a placeholder or hardcoded response since google-generativeai isn't working
        return f"Sample Band {band} essay on {topic}: This is a placeholder response due to API issues. Please implement another AI like OpenAI or Hugging Face for essay generation."
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

async def analyze_essay(essay, band=None):
    prompt = f"""ANALYZE THIS ESSAY STRICTLY FOLLOWING THESE RULES:
ESSAY:
{essay}
1. Grammar Issues: Give ONLY Excellent/Good/Fair/Poor
2. Connector Count: Classify as Low/Medium/High
3. Repeated Words: Classify as Low/Medium/High
4. Advanced Vocabulary: Low/Medium/Advanced
5. Lexical Diversity: Low/Medium/High
6. Avg Sentence Length: Short/Medium/Long
7. Predicted Band: 3-9
FORMAT Must be EXACTLY LIKE THIS:
Grammar Issues: [result]
Advanced Vocabulary: [result]
Connector Count: [result]
Repeated Words: [result]
Lexical Diversity: [result]
Avg Sentence Length: [result]
Predicted IELTS Band: [result]
"""
    try:
        # Removed google-generativeai dependency; you can keep this as is or implement another AI
        # For now, return a placeholder response
        return {
            "Grammar Issues": "Good",
            "Advanced Vocabulary": "Medium",
            "Connector Count": "Medium",
            "Repeated Words": "Low",
            "Lexical Diversity": "Medium",
            "Avg Sentence Length": "Medium",
            "Predicted IELTS Band": "7"
        }
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def parse_analysis(text):
    metrics = {}
    if isinstance(text, dict):  # Handle the dictionary case from the placeholder above
        metrics = text
    else:
        metrics["Grammar Issues"] = re.search(r"Grammar Issues:\s*(.+?)(\n|$)", text).group(1) if re.search(r"Grammar Issues:", text) else "N/A"
        metrics["Advanced Vocabulary"] = re.search(r"Advanced Vocabulary:\s*(.+?)(\n|$)", text).group(1) if re.search(r"Advanced Vocabulary:", text) else "N/A"
        metrics["Connector Count"] = re.search(r"Connector Count:\s*(.+?)(\n|$)", text).group(1) if re.search(r"Connector Count:", text) else "N/A"
        metrics["Repeated Words"] = re.search(r"Repeated Words:\s*(.+?)(\n|$)", text).group(1) if re.search(r"Repeated Words:", text) else "N/A"
        metrics["Lexical Diversity"] = re.search(r"Lexical Diversity:\s*(.+?)(\n|$)", text).group(1) if re.search(r"Lexical Diversity:", text) else "N/A"
        metrics["Avg Sentence Length"] = re.search(r"Avg Sentence Length:\s*(.+?)(\n|$)", text).group(1) if re.search(r"Avg Sentence Length:", text) else "N/A"
        metrics["Predicted IELTS Band"] = re.search(r"Predicted IELTS Band:\s*([3-9])", text).group(1) if re.search(r"Predicted IELTS Band:", text) else "N/A"
    return metrics

async def create_visualization(metrics):
    plt.figure(figsize=(10, 8))
    plt.subplot(polar=True)
    categories = ['Grammar Accuracy', 'Lexical Resource', 'Cohesion/Connectors', 'Task Achievement', 'Sentence Structure']
    band_scores = {
        'Grammar Accuracy': _map_grammar_to_band(metrics['Grammar Issues']),
        'Lexical Resource': _map_vocab_to_band(metrics['Advanced Vocabulary']),
        'Cohesion/Connectors': _map_connectors_to_band(metrics['Connector Count']),
        'Task Achievement': _map_lexical_to_band(metrics['Lexical Diversity']),
        'Sentence Structure': _map_sentence_length_to_band(metrics['Avg Sentence Length'])
    }
    values = [band_scores[cat] for cat in categories]
    values += values[:1]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([3, 5, 7, 9], ["Band 3", "Band 5", "Band 7", "Band 9"], color="grey", size=8)
    plt.ylim(0, 9)
    predicted_band = int(metrics['Predicted IELTS Band']) if metrics['Predicted IELTS Band'].isdigit() else 0
    ax.plot(angles, [predicted_band]*len(angles), color='#ff7f0e', linestyle='--')
    ax.fill(angles, [predicted_band]*len(angles), color='#ff7f0e', alpha=0.1)
    descriptor_text = "Band Key:\n9: Expert Level\n7-8: Good Command\n5-6: Competent User\n3-4: Limited User"
    plt.figtext(1.1, 0.9, descriptor_text, wrap=True, fontsize=9)
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

def _map_grammar_to_band(grade):
    mapping = {'Excellent': 9, 'Good': 7, 'Fair': 5, 'Poor': 3}
    return mapping.get(grade, 3)

def _map_vocab_to_band(vocab_level):
    mapping = {'Advanced': 8, 'Medium': 6, 'Low': 4}
    return mapping.get(vocab_level, 4)

def _map_connectors_to_band(qualitative):
    mapping = {'High': 9, 'Medium': 7, 'Low': 5}
    return mapping.get(qualitative, 5)

def _map_lexical_to_band(diversity):
    mapping = {'High': 8, 'Medium': 6, 'Low': 4}
    return mapping.get(diversity, 4)

def _map_sentence_length_to_band(length):
    mapping = {'Long': 8, 'Medium': 6, 'Short': 4}
    return mapping.get(length, 4)

async def show_analysis(update, context, generated=False):
    essay = context.user_data.get('current_essay', '')
    band = context.user_data.get('band') if generated else None
    processing = await update.message.reply_text("🔍 Analyzing...")
    analysis = await analyze_essay(essay, band)
    await show_members_and_meme(update)
    if not analysis:
        await processing.delete()
        await update.message.reply_text("❌ Analysis failed")
        return SELECT_OPTION
    context.user_data['analysis'] = analysis
    result_text = (
        "📊 Detailed Analysis:\n"
        f"┌ Grammar Issues: {analysis['Grammar Issues']}\n"
        f"├ Advanced Vocabulary: {analysis['Advanced Vocabulary']}\n"
        f"├ Connector Count: {analysis['Connector Count']}\n"
        f"├ Repeated Words: {analysis['Repeated Words']}\n"
        f"├ Lexical Diversity: {analysis['Lexical Diversity']}\n"
        f"├ Avg Sentence Length: {analysis['Avg Sentence Length']}\n"
        f"└ Predicted Band: {analysis['Predicted IELTS Band']}\n"
    )
    try:
        buf = await create_visualization(analysis)
    except Exception as e:
        print(f"Visualization error: {e}")
        buf = None
    keyboard = [
        [InlineKeyboardButton("Grammar Recommendations", callback_data='grammar_rec')],
        [InlineKeyboardButton("Home🏡", callback_data='restart')]
    ]
    await processing.delete()
    if buf:
        await update.message.reply_photo(photo=buf, caption=result_text, reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(result_text, reply_markup=InlineKeyboardMarkup(keyboard))
    return SELECT_OPTION

async def process_essay(update: Update, context):
    context.user_data['current_essay'] = update.message.text
    return await show_analysis(update, context)

async def handle_recommendations(update: Update, context):
    query = update.callback_query
    await query.answer()
    essay = context.user_data.get("current_essay", "")
    analysis = context.user_data.get("analysis", {})
    if not essay:
        await query.edit_message_text("⚠️ No essay found")
        return SELECT_OPTION
    if query.data == 'grammar_rec':
        prompt = f"""Analyze this essay and provide grammar recommendations:
- List connector count and suggest improvements
- Highlight repeated words with counts
- Identify advanced vocabulary usage
Essay:
{essay}
Format exactly as:
Suggested Connectors: [Furthermore, However...]
-------------------------------------------------
ADVANCED VOCABULARY IMPROVEMENTS:
Advanced Vocabulary: [8] - [alleviated, autonomy...]
Suggested Improvements: [Specific replacement examples]
---------------------------------------------------------
Grammar and Sentence Structure Corrections:
Original Line: Before, I used the bus.
Recommended: Previously, I relied on the bus.
_________________________________________________________
Original Line: The bus was slow and not good.
Recommended: However, the bus was slow and unreliable.
_--------------------------------------------------------
Word Choice Corrections:
Original word: very
Recommended: extremely (e.g., "very important" → "extremely important")
____________________________________________________________________________
Incorrect word: good
Recommended: convenient (e.g., "good for my life" → "convenient for my life")
"""
        try:
            # Removed google-generativeai dependency; you can keep this as is or implement another AI
            # For now, return a placeholder or hardcoded response
            rec_text = """Suggested Connectors: [Furthermore, However]
-------------------------------------------------
ADVANCED VOCABULARY IMPROVEMENTS:
Advanced Vocabulary: [5] - [significant, development]
Suggested Improvements: Replace 'good' with 'beneficial'
---------------------------------------------------------
Grammar and Sentence Structure Corrections:
Original Line: I go to school everyday.
Recommended: I go to school every day.
_________________________________________________________
Original Line: The class is very fun.
Recommended: The class is highly engaging.
_--------------------------------------------------------
Word Choice Corrections:
Original word: very
Recommended: highly (e.g., "very important" → "highly important")
____________________________________________________________________________
Original word: good
Recommended: effective (e.g., "good strategy" → "effective strategy")"""
            await query.message.reply_text(
                f"🔍 Grammar Recommendations:\n\n{rec_text}",
                reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Home🏡", callback_data='restart')]])
            )
        except Exception as e:
            await query.message.reply_text(f"❌ Error: {str(e)}")
    return SELECT_OPTION

async def show_members_and_meme(update: Update):
    members_message = await update.message.reply_animation("https://tenor.com/bqYoH.gif")
    await asyncio.sleep(6)
    await members_message.delete()
    meme_url = "https://tenor.com/buads.gif"
    await update.message.reply_animation(meme_url)

async def loading(update: Update):
    loading = await update.message.reply_animation("https://tenor.com/bqYoH.gif")
    await asyncio.sleep(6)
    await loading.delete()

async def restart_program(update: Update, context):
    query = update.callback_query
    await query.answer()
    context.user_data.clear()
    await query.message.reply_text(
        text="🏠 Main Menu:",
        reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Generate Essay", callback_data='generate')],
            [InlineKeyboardButton("Analyze Essay", callback_data='analyze')],
            [InlineKeyboardButton("Handwriting Recognition", callback_data='handwriting')]
        ])
    )
    return SELECT_OPTION

def main():
    application = Application.builder().token("7752367540:AAHXfq2xa6KpvrLFAOm0qherh7sgIrGH13w").build()
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],
        states={
            SELECT_OPTION: [
                CallbackQueryHandler(restart_program, pattern='^restart$'),
                CallbackQueryHandler(handle_recommendations, pattern='^(grammar_rec|refine)$'),
                CallbackQueryHandler(select_option)
            ],
            ASK_BAND: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_band)],
            ASK_TOPIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_topic)],
            PROCESS_ESSAY: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_essay)],
            HANDWRITING_UPLOAD: [MessageHandler(filters.PHOTO, process_handwriting)]
        },
        fallbacks=[CommandHandler('start', start)]
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()