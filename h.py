import re
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import requests
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ConversationHandler,
    filters
)

# Configure Gemini API with your API key
genai.configure(api_key="AIzaSyCerjxXg_b6AAW4sQEq2Tzxo_sXV40dkOI")

# Initialize Gemini model
model = genai.GenerativeModel("gemini-2.0-flash")

# Conversation states
SELECT_OPTION, ASK_BAND, ASK_TOPIC, PROCESS_ESSAY, HANDWRITING_UPLOAD, REFINE_TEXT = range(6)

# Band-specific essay guidelines
BAND_INSTRUCTIONS = {
    9: {
        'vocab': "Sophisticated terms (paradigm shift, socioeconomic)",
        'structure': "Complex sentences with subordinate clauses",
        'cohesion': "Advanced transitions (consequently, furthermore)",
        'errors': "Virtually error-free",
        'length': "300+ words"
    },
    8: {
        'vocab': "Advanced terms (globalization, contemporary)",
        'structure': "Varied complex structures",
        'cohesion': "Effective linking (however, moreover)",
        'errors': "Rare minor errors",
        'length': "280+ words"
    },
    7: {
        'vocab': "Adequate range (significant, development)",
        'structure': "Mix of simple/complex sentences",
        'cohesion': "Clear paragraphing",
        'errors': "Some errors",
        'length': "250+ words"
    },
    6: {
        'vocab': "Basic academic terms",
        'structure': "Simple structures with some complexity",
        'cohesion': "Basic connectors (and, but)",
        'errors': "Noticeable errors",
        'length': "250 words"
    },
    5: {
        'vocab': "Limited range (good, things)",
        'structure': "Mostly simple sentences",
        'cohesion': "Few connectors",
        'errors': "Frequent errors",
        'length': "200 words"
    },
    4: {
        'vocab': "Basic vocabulary (school, job)",
        'structure': "Short simple sentences",
        'cohesion': "Minimal linking",
        'errors': "Systematic errors",
        'length': "150 words"
    },
    3: {
        'vocab': "Very basic terms (study, work)",
        'structure': "Fragmented sentences",
        'cohesion': "No connectors",
        'errors': "Severe errors",
        'length': "<100 words"
    }
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
        await query.edit_message_text(text="📸 Upload a clear photo of your handwritten or printed text (any language):")
        return HANDWRITING_UPLOAD
    return SELECT_OPTION

async def process_handwriting(update: Update, context):
    try:
        # Get the photo with size check
        photo = update.message.photo[-1]
        if photo.file_size > 5_000_000:
            await update.message.reply_text("❌ Image too large (max 5MB)")
            return SELECT_OPTION

        # Download with timeout
        try:
            file = await photo.get_file()
            img_bytes = await asyncio.wait_for(file.download_as_bytearray(), timeout=30)  # Increased timeout
        except asyncio.TimeoutError:
            await update.message.reply_text("❌ Download timed out. Please try again with a smaller image.")
            return SELECT_OPTION

        # Simplified preprocessing (remove Fourier Transform for simplicity, as Gemini handles image quality)
        try:
            img = Image.open(BytesIO(img_bytes)).convert('L')  # Grayscale
            buffer = BytesIO()
            img.save(buffer, format="PNG")
            image_data = buffer.getvalue()
        except Exception as e:
            print(f"Image processing failed: {e}")
            image_data = img_bytes  # Fallback to original

        # Gemini API call with gemini-2.0-flash
        try:
            response = model.generate_content(
                contents=[
                    {"mime_type": "image/png", "data": image_data},
                    "Extract text exactly as written from this image, preserve line breaks and formatting:"
                ]
            )
            response_text = response.text.strip() if hasattr(response, 'text') else "No text detected."

            # Truncate text to fit Telegram's 4096-character limit, including header and buffer
            TELEGRAM_MAX_LENGTH = 4096
            header = "✍️ Extracted Text:\n\n"
            max_text_length = TELEGRAM_MAX_LENGTH - len(header) - 50  # Leave room for truncation notice and formatting

            if len(response_text) > max_text_length:
                response_text = response_text[:max_text_length] + "\n... (truncated due to Telegram limit)"

            context.user_data['extracted_text'] = response_text  # Store extracted text for refinement
            keyboard = [
                [InlineKeyboardButton("Restart", callback_data='restart')],
                [InlineKeyboardButton("Refine Text", callback_data='refine')]
            ]
            await update.message.reply_text(
                f"{header}{response_text}",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return HANDWRITING_UPLOAD  # Stay in HANDWRITING_UPLOAD for potential refinement

        except Exception as e:
            await update.message.reply_text(f"❌ Recognition failed: {str(e)[:1000]}")
            print(f"Gemini API error: {str(e)}")
            return SELECT_OPTION

    except Exception as e:
        await update.message.reply_text(f"❌ General error: {str(e)[:1000]}")
        print(f"General error: {str(e)}")
        return SELECT_OPTION

async def refine_text(update: Update, context):
    query = update.callback_query
    await query.answer()
    extracted_text = context.user_data.get('extracted_text', "")

    if not extracted_text:
        await query.edit_message_text("⚠️ No text found to refine.")
        return SELECT_OPTION

    try:
        prompt = f"""Refine this extracted text into a meaningful, coherent paragraph, correcting spelling, grammar, and formatting while preserving the original meaning and content as closely as possible. Do not add or remove any substantive information, only improve clarity and structure. Output only the refined paragraph, with no additional explanations or comments.

Text:
{extracted_text}"""
        
        response = model.generate_content(prompt)
        refined_text = response.text.strip() if hasattr(response, 'text') else "Refinement failed."

        # Truncate refined text to fit Telegram's limit
        TELEGRAM_MAX_LENGTH = 4096
        header = "🔄 Refined Text:\n\n"
        max_text_length = TELEGRAM_MAX_LENGTH - len(header) - 50
    
        if len(refined_text) > max_text_length:
            refined_text = refined_text[:max_text_length] + "\n... (truncated due to Telegram limit)"

        keyboard = [[InlineKeyboardButton("Restart", callback_data='restart')]]
        await query.message.reply_text(
            f"{header}{refined_text}",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return SELECT_OPTION  # Return to main menu after refinement

    except Exception as e:
        await query.edit_message_text(f"❌ Refinement failed: {str(e)[:1000]}")
        print(f"Refinement error: {str(e)}")
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
    # Send the generated essay with a restart button
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
-Generate the essay based on the FOLLOWING THESE RULES to get the exact band score you generated in the analysis result:
1. Grammar Issues: is determined by the misspelling
-Classify the grammar quality of the essay as:
Must be exactly error in these ranges
for band 9:<2 grammar issues => Excellent
for band 8:4-6 grammar issues => Good
for band 7:8-9 grammar issues=> Fair
for band 6:11-13 grammar issues=> Fair
for band 5:14-16 grammar issues => Poor
for band 4:18-20 grammar issues => Poor
for band 3:>22 grammar issues => Poor

2. Connector Count: Classify as based on the number of connectors (e.g., however, furthermore, and).
Classify the essay based on the number of connectors (e.g., "however," "furthermore," "and")(count conjunction words):
for band 3:<2 connectors => Low
for band 4:3-4 connectors => Low
for band 5:5-7 connectors => Low
for band 6:8-10 connectors => Medium
for band 7:10-12 connectors => Medium
for band 8:13-15 connectors => High
for band 9:>16 connectors => High
3. Repeated Words: Classify as Low/Medium/High based on the number of repeated nouns/verbs.
Low: 0-3 repeated words for band 9 and 8
Medium: 4-5 repeated words for band 7, 6, 5
High: 6+ repeated words for band 4, 3
Double-check the essay to ensure consistency when evaluating the same essay multiple times.
4. Advanced Vocabulary: Low/Medium/Advanced(double-check for consistency)(advanced Vocabulary determined by the complexity of the terms used in the essay)
Low: <3 advanced words for band 3, 4
Medium: 5 <= advanced words <= 10 for band 5, 6, 7
Advanced: >=11 advanced words (must use more than 12 words for band 9) for band 8, 9
Double-check the essay to ensure consistency when evaluating the same essay multiple times.

5. Lexical Diversity: Low/Medium/High (calculate Lexical Density:
    * Count all words in the essay.
    * Classify each word as either a CONTENT word (noun, verb, adjective, adverb) or a FUNCTION word (article, preposition, conjunction, pronoun, auxiliary verb). If a word can be both, consider its usage in the context.
    * Calculate Lexical Density = (Number of Content Words) / (Total Number of Words)
    * Provide the Lexical Density score as a decimal number between 0 and 1 (e.g., 0.55).
    * Also, indicate whether the Lexical Diversity is Low, Medium, or High based on these ranges:
        * Low: Lexical Density < 0.40 for band 3, 4
        * Medium: 0.40 <= Lexical Density < 0.60 for band 5, 6, 7
        * High: Lexical Density >= 0.60 for band 8, 9
6. Avg Sentence Length: Short (5-10) / Medium (11-20) / Long (20+) (calculate with Average Sentence Length = Total Number of Words / Total Number of Sentences)
Short: 5-10 words per sentence for band 3, 4
Medium: 11-20 words per sentence for band 5, 6, 7
Long: 20+ words per sentence for band 8, 9
7. Predicted Band: 3-9 (decimal can be included, e.g., 9.0)

Score calculation to update the essay:
Grammar Issues:
Excellent: 5
Good: 4
Fair: 3
Poor: 2

Connector Count:
Low: 2
Medium: 4
High: 5

Repeated Words:
Low: 5
Medium: 4
High: 3

Advanced Vocabulary:
Low: 2
Medium: 4
Advanced: 5

Lexical Diversity:
Low: 2
Medium: 4
High: 5

Average Sentence Length:
Short: 3
Medium: 4
Long: 5

Breakdown for Band Ranges:
Band 3 (3.0 - 4.5): Score: 6 to 15 points
Band 4 (4.0 - 5.0): Score: 16 to 19 points
Band 5 (5.0 - 6.0): Score: 20 to 22 points
Band 6 (6.0 - 6.5): Score: 23 to 24 points
Band 7 (7.0 - 8.0): Score: 25 to 27 points
Band 8 (8.0 - 9.0): Score: 28 to 29 points
Band 9 (9.0): Score: 30 points

- Don’t show results, show only a plain essay based on the above rules. Do not include analysis or additional comments (e.g., Grammar Issues: Excellent, Predicted Band: 9.0).
- Before generating the essay, check with the rules and score to ensure it matches the exact band score generated in the analysis result.
- Output: Plain text only. Do not include explanations or additional comments."""
    try:
        response = model.generate_content(prompt)
        return response.text.replace("**", "").strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"

async def analyze_essay(essay, band=None):
    prompt = f"""ANALYZE THIS ESSAY STRICTLY FOLLOWING THESE RULES:
ESSAY:
{essay}

1. Grammar Issues: Determined by the number of misspellings and grammatical errors.
- Classify the grammar quality of the essay as:
  - Excellent: 0-1 minor errors
  - Good: 2-3 minor errors
  - Fair: 4-5 errors or noticeable grammar issues
  - Poor: 6+ errors or significant grammar problems
  - Double-check the essay to ensure consistency when evaluating the same essay multiple times.

2. Connector Count: Classify as Low/Medium/High based on the number of connectors (e.g., however, furthermore, and).
- Classify based on the number of connectors (e.g., "however," "furthermore," "and") (count conjunction words):
  - Low: 0-5 connectors
  - Medium: 6-12 connectors
  - High: 13+ connectors
  - Double-check the essay to ensure consistency.

3. Repeated Words: Classify as Low/Medium/High based on the number of repeated nouns/verbs.
- Low: 0-2 repeated words
- Medium: 3-4 repeated words
- High: 5+ repeated words
- Double-check the essay to ensure consistency.

4. Advanced Vocabulary: Low/Medium/Advanced (double-check for consistency).
- Advanced Vocabulary is determined by the complexity of the terms used in the essay:
  - Low: Frequent use of simple words
  - Medium: A mix of everyday and slightly more advanced terms
  - Advanced: Use of sophisticated terms, formal language, and specialized vocabulary
- Double-check the essay to ensure consistency.

5. Lexical Diversity: Low/Medium/High (calculate Lexical Density):
  - Count all words in the essay.
  - Classify each word as either a CONTENT word (noun, verb, adjective, adverb) or a FUNCTION word (article, preposition, conjunction, pronoun, auxiliary verb). If a word can be both, consider its usage in the context.
  - Calculate Lexical Density = (Number of Content Words) / (Total Number of Words)
  - Provide the Lexical Density score as a decimal number between 0 and 1 (e.g., 0.55).
  - Indicate whether the Lexical Diversity is:
    - Low: Lexical Density < 0.40
    - Medium: 0.40 <= Lexical Density < 0.60
    - High: Lexical Density >= 0.60

6. Avg Sentence Length: Short (5-10) / Medium (11-20) / Long (20+) (calculate with Average Sentence Length = Total Number of Words / Total Number of Sentences):
  - Short: 5-10 words per sentence
  - Medium: 11-20 words per sentence
  - Long: 20+ words per sentence

7. Predicted Band: 3-9 (decimal can be included, e.g., 9.0)

Score calculation to determine the band:
- Grammar Issues:
  - Excellent: 5
  - Good: 4
  - Fair: 3
  - Poor: 2
- Connector Count:
  - Low: 2
  - Medium: 4
  - High: 5
- Repeated Words:
  - Low: 5
  - Medium: 4
  - High: 3
- Advanced Vocabulary:
  - Low: 2
  - Medium: 4
  - Advanced: 5
- Lexical Diversity:
  - Low: 2
  - Medium: 4
  - High: 5
- Average Sentence Length:
  - Short: 3
  - Medium: 4
  - Long: 5

Breakdown for Band Ranges:
- Band 3 (3.0 - 4.5): Score: 6 to 15 points
- Band 4 (4.0 - 5.0): Score: 16 to 19 points
- Band 5 (5.0 - 6.0): Score: 20 to 22 points
- Band 6 (6.0 - 6.5): Score: 23 to 24 points
- Band 7 (7.0 - 8.0): Score: 25 to 27 points
- Band 8 (8.0 - 9.0): Score: 28 to 29 points
- Band 9 (9.0): Score: 30 points

8. Band Prediction Criteria (if they meet these criteria):
- Band 9:
  - 'vocab': "Sophisticated terms (paradigm shift, socioeconomic)"
  - 'structure': "Complex sentences with subordinate clauses"
  - 'cohesion': "Advanced transitions (consequently, furthermore)"
  - 'errors': "Virtually error-free"
  - 'length': "300+ words"
- Band 8:
  - 'vocab': "Advanced terms (globalization, contemporary)"
  - 'structure': "Varied complex structures"
  - 'cohesion': "Effective linking (however, moreover)"
  - 'errors': "Rare minor errors"
  - 'length': "280+ words"
- Band 7:
  - 'vocab': "Adequate range (significant, development)"
  - 'structure': "Mix of simple/complex sentences"
  - 'cohesion': "Clear paragraphing"
  - 'errors': "Some errors"
  - 'length': "250+ words"
- Band 6:
  - 'vocab': "Basic academic terms"
  - 'structure': "Simple structures with some complexity"
  - 'cohesion': "Basic connectors (and, but)"
  - 'errors': "Noticeable errors"
  - 'length': "250 words"
- Band 5:
  - 'vocab': "Limited range (good, things)"
  - 'structure': "Mostly simple sentences"
  - 'cohesion': "Few connectors"
  - 'errors': "Frequent errors"
  - 'length': "200 words"
- Band 4:
  - 'vocab': "Basic vocabulary (school, job)"
  - 'structure': "Short simple sentences"
  - 'cohesion': "Minimal linking"
  - 'errors': "Systematic errors"
  - 'length': "150 words"
- Band 3:
  - 'vocab': "Very basic terms (study, work)"
  - 'structure': "Fragmented sentences"
  - 'cohesion': "No connectors"
  - 'errors': "Severe errors"
  - 'length': "<100 words"

9. Recheck the essay three (3) exact times with the above rules and guidelines. The analysis must be consistent across all checks.

FORMAT Must be EXACTLY LIKE THIS (no extra information):
Grammar Issues: [Excellent/Good/Fair/Poor]
Advanced Vocabulary: [Low/Medium/Advanced]
Connector Count: [Low/Medium/High]
Repeated Words: [Low/Medium/High]
Lexical Diversity: [Low/Medium/High]
Avg Sentence Length: [Short/Medium/Long]
Predicted IELTS Band: [3-9] (whole number or decimal, e.g., 9.0)
"""
    try:
        response = model.generate_content(prompt)
        return parse_analysis(response.text)
    except Exception as e:
        print(f"Analysis error: {e}")
        return None

def parse_analysis(text):
    metrics = {}
    # Use regex for more robust parsing
    metrics["Grammar Issues"] = re.search(r"Grammar Issues:\s*(Excellent|Good|Fair|Poor)", text).group(1) if re.search(r"Grammar Issues:", text) else "N/A"
    metrics["Advanced Vocabulary"] = re.search(r"Advanced Vocabulary:\s*(Low|Medium|Advanced)", text).group(1) if re.search(r"Advanced Vocabulary:", text) else "N/A"
    metrics["Connector Count"] = re.search(r"Connector Count:\s*(Low|Medium|High)", text).group(1) if re.search(r"Connector Count:", text) else "N/A"
    metrics["Repeated Words"] = re.search(r"Repeated Words:\s*(Low|Medium|High)", text).group(1) if re.search(r"Repeated Words:", text) else "N/A"
    metrics["Lexical Diversity"] = re.search(r"Lexical Diversity:\s*(Low|Medium|High)", text).group(1) if re.search(r"Lexical Diversity:", text) else "N/A"
    metrics["Avg Sentence Length"] = re.search(r"Avg Sentence Length:\s*(Short|Medium|Long)", text).group(1) if re.search(r"Avg Sentence Length:", text) else "N/A"
    metrics["Predicted IELTS Band"] = re.search(r"Predicted IELTS Band:\s*([3-9]\.?\d?)", text).group(1) if re.search(r"Predicted IELTS Band:", text) else "N/A"
    return metrics

async def create_visualization(metrics):
    plt.figure(figsize=(10, 8))
    plt.subplot(polar=True)
    
    # IELTS band criteria categories
    categories = [
        'Grammar Accuracy', 
        'Lexical Resource',
        'Cohesion/Connectors',
        'Task Achievement',
        'Sentence Structure'
    ]
    
    # Convert metrics to band scale (1-9)
    band_scores = {
        'Grammar Accuracy': _map_grammar_to_band(metrics['Grammar Issues']),
        'Lexical Resource': _map_vocab_to_band(metrics['Advanced Vocabulary']),
        'Cohesion/Connectors': _map_connectors_to_band(metrics['Connector Count']),
        'Task Achievement': _map_lexical_to_band(metrics['Lexical Diversity']),
        'Sentence Structure': _map_sentence_length_to_band(metrics['Avg Sentence Length'])
    }
    
    # Convert dict to list maintaining category order
    values = [band_scores[cat] for cat in categories]
    
    # Complete the loop by appending first value
    values += values[:1]
    
    # IELTS band labels and angles
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Plot main data
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid', color='#1f77b4')
    ax.fill(angles, values, color='#1f77b4', alpha=0.25)
    
    # Draw axis lines
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    
    # Set IELTS band labels (1-9)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks(
        [3, 5, 7, 9], 
        ["Band 3", "Band 5", "Band 7", "Band 9"],
        color="grey", 
        size=8
    )
    plt.ylim(0, 9)
    
    # Add predicted band
    predicted_band = float(metrics['Predicted IELTS Band']) if metrics['Predicted IELTS Band'].replace('.', '').isdigit() else 0
    ax.plot(angles, [predicted_band]*len(angles), color='#ff7f0e', linestyle='--')
    ax.fill(angles, [predicted_band]*len(angles), color='#ff7f0e', alpha=0.1)
    
    # Add band descriptors
    descriptor_text = (
        "Band Key:\n"
        "9: Expert Level\n"
        "7-8: Good Command\n"
        "5-6: Competent User\n"
        "3-4: Limited User"
    )
    plt.figtext(1.1, 0.9, descriptor_text, wrap=True, fontsize=9)
    
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return buf

# Helper mapping functions
def _map_grammar_to_band(grade):
    mapping = {'Excellent': 9, 'Good': 7, 'Fair': 5, 'Poor': 3}
    return mapping.get(grade, 3)

def _map_vocab_to_band(vocab_level):
    mapping = {'Advanced': 8, 'Medium': 6, 'Low': 4}
    return mapping.get(vocab_level, 4)

def _map_connectors_to_band(qualitative):
    mapping = {'High': 9, 'Medium': 7, 'Low': 5}
    return mapping.get(qualitative, 5)

def _map_repeated_words_to_band(qualitative):
    mapping = {'Low': 9, 'Medium': 7, 'High': 5}
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
        await update.message.reply_photo(
            photo=buf,
            caption=result_text,
            reply_markup=InlineKeyboardMarkup(keyboard))
    else:
        await update.message.reply_text(
            result_text,
            reply_markup=InlineKeyboardMarkup(keyboard))
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

- Format exactly as:

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
------------------------------------------------------------------------------
Essay:
{essay}"""
        try:
            response = model.generate_content(
                contents=prompt  # Removed 'model' parameter
            )
            rec_text = response.text.replace("**", "").strip()
            await query.message.reply_text(
                f"🔍 Grammar Recommendations:\n\n{rec_text}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Home🏡", callback_data='restart')]
                ])
            )
        except Exception as e:
            await query.message.reply_text(f"❌ Error: {str(e)}")
    elif query.data == 'refine':
        try:
            current_band = float(analysis.get('Predicted IELTS Band', 7))
        except:
            current_band = 7.0
        target_band = min(current_band + 1, 9.0)
        
        prompt = f"""Improve this essay to Band {target_band}:
- Fix grammar: {analysis.get('Grammar Issues', '')}
- Reduce repeats: {analysis.get('Repeated Words', 'Low')} instances
- Add connectors: Current {analysis.get('Connector Count', 'Low')}
- Enhance vocabulary: {analysis.get('Advanced Vocabulary', 'Low')}
-with these rules
1. Grammar Issues: Give ONLY Excellent/Good/Fair/Poor
-Classify the grammar quality of the essay as:
Excellent: 0-1 minor errors
Good: 2-3 minor errors
Fair: 4-5 errors or noticeable grammar issues
Poor: 6+ errors or significant grammar problems
Double-check the essay to ensure consistency when evaluating the same essay multiple times.

2. Connector Count: Classify as Low/Medium/High based on the number of connectors (e.g., however, furthermore, and).
Classify the essay based on the number of connectors (e.g., "however," "furthermore," "and"):
Low: 0-5 connectors
Medium: 6-12 connectors
High: 13+ connectors
Double-check the essay to ensure consistency when evaluating the same essay multiple times.

3. Repeated Words: Classify as Low/Medium/High based on the number of repeated nouns/verbs.
Low: 0-2 repeated words
Medium: 3-4 repeated words
High: 5+ repeated words
Double-check the essay to ensure consistency when evaluating the same essay multiple times.

4. Advanced Vocabulary: Low/Medium/Advanced (double-check for consistency).
- Advanced Vocabulary is determined by the complexity of the terms used in the essay:
  - Low: Frequent use of simple words
  - Medium: A mix of everyday and slightly more advanced terms
  - Advanced: Use of sophisticated terms, formal language, and specialized vocabulary
- Double-check the essay to ensure consistency when evaluating the same essay multiple times.

5. Lexical Diversity: Low/Medium/High (calculate Lexical Density):
  - Count all words in the essay.
  - Classify each word as either a CONTENT word (noun, verb, adjective, adverb) or a FUNCTION word (article, preposition, conjunction, pronoun, auxiliary verb). If a word can be both, consider its usage in the context.
  - Calculate Lexical Density = (Number of Content Words) / (Total Number of Words)
  - Provide the Lexical Density score as a decimal number between 0 and 1 (e.g., 0.55).
  - Indicate whether the Lexical Diversity is:
    - Low: Lexical Density < 0.40
    - Medium: 0.40 <= Lexical Density < 0.60
    - High: Lexical Density >= 0.60

6. Avg Sentence Length: Short (5-10) / Medium (11-20) / Long (20+) (calculate with Average Sentence Length = Total Number of Words / Total Number of Sentences):
  - Short: 5-10 words per sentence
  - Medium: 11-20 words per sentence
  - Long: 20+ words per sentence

7. Predicted Band: 3-9 (decimal can be included, e.g., 9.0)

Score calculation to update the essay:
- Grammar Issues:
  - Excellent: 5
  - Good: 4
  - Fair: 3
  - Poor: 2
- Connector Count:
  - Low: 2
  - Medium: 4
  - High: 5
- Repeated Words:
  - Low: 5
  - Medium: 4
  - High: 3
- Advanced Vocabulary:
  - Low: 2
  - Medium: 4
  - Advanced: 5
- Lexical Diversity:
  - Low: 2
  - Medium: 4
  - High: 5
- Average Sentence Length:
  - Short: 3
  - Medium: 4
  - Long: 5

Breakdown for Band Ranges:
- Band 3 (3.0 - 4.5): Score: 6 to 15 points
- Band 4 (4.0 - 5.0): Score: 16 to 19 points
- Band 5 (5.0 - 6.0): Score: 20 to 22 points
- Band 6 (6.0 - 6.5): Score: 23 to 24 points
- Band 7 (7.0 - 8.0): Score: 25 to 27 points
- Band 8 (8.0 - 9.0): Score: 28 to 29 points
- Band 9 (9.0): Score: 30 points

- Don’t show results, show only a plain essay based on the above rules. Do not include analysis or additional comments (e.g., Grammar Issues: Excellent, Predicted Band: 9.0).
- Output: Plain text only. Do not include explanations or additional comments.

Output ONLY the refined essay:
{essay}"""
        try:
            response = model.generate_content(
                contents=prompt  # Removed 'model' parameter
            )
            refined = response.text.replace("**", "").strip()
            context.user_data['current_essay'] = refined
            await query.message.reply_text(
                f"🔄 Refined Essay (Band {target_band}):\n\n{refined}",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Home🏡", callback_data='restart')]
                ])
            )
        except Exception as e:
            await query.message.reply_text(f"❌ Refine failed: {str(e)}")
    
    return SELECT_OPTION

async def show_members_and_meme(update: Update):
    # Show members message first
    members_message = await update.message.reply_animation(
        "https://tenor.com/bqYoH.gif"
    )
    await asyncio.sleep(6)
    await members_message.delete()

    # Send meme GIF
    meme_url = "https://tenor.com/buads.gif"
    await update.message.reply_animation(meme_url)

async def loading(update: Update):
    # Show loading animation
    loading = await update.message.reply_animation(
        "https://tenor.com/bqYoH.gif"
    )
    await asyncio.sleep(6)
    await loading.delete()

async def restart_program(update: Update, context):
    query = update.callback_query
    await query.answer()
    context.user_data.clear()
    # Send as new message instead of editing
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
            HANDWRITING_UPLOAD: [
                MessageHandler(filters.PHOTO, process_handwriting),
                CallbackQueryHandler(refine_text, pattern='^refine$')
            ]
        },
        fallbacks=[CommandHandler('start', start)]
    )
    application.add_handler(conv_handler)
    application.run_polling()

if __name__ == "__main__":
    main()