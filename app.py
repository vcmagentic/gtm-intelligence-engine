import os
import requests
import tiktoken
from flask import Flask, render_template, request, flash
from bs4 import BeautifulSoup

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "fallback_secret_key")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

MODEL = "gpt-4o"
MAX_TOKENS_PER_CHUNK = 10000
ENCODING = tiktoken.encoding_for_model(MODEL)

GTM_CONTEXT_TEMPLATE = """
You are an Agentic GTM Intelligence Strategist specializing in deep discovery call analysis.

Company Background:
{context}

---

You are analyzing the following sales/discovery call transcript:
{transcript}

---

Extract and structure insights strictly as follows:

1. Detected Client Company Name:
2. Participants Mapping:
   - Buyers (Name and Company)
   - BusinessNext/Seller Team (Name)
3. Deal Stage Inference:
4. Buying Signals:
5. Objections or Frictions:
6. GTM Pains:
7. Expansion Signals:
8. Technical Blocker Risks:
9. Narrative Gaps:
10. Multi-threading Need:
11. Coaching Moments for AE:
12. Recommended Next Steps:
13. Urgency Scoring (0–1):
14. Human-Readable Executive Summary:
15. GTM Ontology Signals:
   - Personas
   - Use Cases
   - Value Drivers
   - Frictions
   - Triggers

Focus on strategic, layered GTM intelligence extraction only.
No markdown, no headings, clean readable plain text.
"""

def count_tokens(text):
    return len(ENCODING.encode(text))

def chunk_transcript(text, chunk_token_limit):
    tokens = ENCODING.encode(text)
    chunks = []
    for i in range(0, len(tokens), chunk_token_limit):
        chunk = ENCODING.decode(tokens[i:i + chunk_token_limit])
        chunks.append(chunk)
    return chunks

def call_gpt(prompt_text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a GTM Intelligence Strategist. Respond only with structured insights. No commentary, no markdown formatting."},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": 0.2
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    completion = response.json()
    output_text = completion["choices"][0]["message"]["content"]
    return output_text.strip()

def scrape_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")
        combined_text = " ".join([p.get_text(separator=" ", strip=True) for p in paragraphs])
        return combined_text[:3000]  # Limit scrape size to 3000 chars
    except Exception as e:
        print(f"Website scrape error: {e}")
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        website_url = request.form.get('website_url', '').strip()
        manual_context = request.form.get('manual_context', '').strip()
        transcript_text = request.form.get('transcript_text', '').strip()

        if not transcript_text:
            flash("Please paste a transcript!", "error")
            return render_template('index.html')

        website_context = ""
        if website_url:
            website_context = scrape_website_text(website_url)

        # Choose context: Manual > Website
        if manual_context:
            used_context = manual_context
        elif website_context:
            used_context = website_context
        else:
            flash("Please provide either a Company Website URL or Manual Context!", "error")
            return render_template('index.html')

        # Build final prompt
        full_prompt = GTM_CONTEXT_TEMPLATE.format(context=used_context, transcript=transcript_text)

        # Token count check
        total_tokens = count_tokens(full_prompt)
        all_outputs = []

        if total_tokens <= MAX_TOKENS_PER_CHUNK:
            output = call_gpt(full_prompt)
            all_outputs.append(output)
        else:
            transcript_chunks = chunk_transcript(transcript_text, MAX_TOKENS_PER_CHUNK - count_tokens(GTM_CONTEXT_TEMPLATE) - 1000)
            for chunk in transcript_chunks:
                prompt = GTM_CONTEXT_TEMPLATE.format(context=used_context, transcript=chunk)
                output = call_gpt(prompt)
                all_outputs.append(output)

        combined_text_output = "\n\n".join(all_outputs)

        return render_template('index.html', result=combined_text_output)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
