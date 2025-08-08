from flask import Flask, request, jsonify
import requests
import openai
import os
import json
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from waitress import serve
import multiprocessing

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)
GNB_TICKET_URL = os.getenv("TICKET_API_URL")
if not GNB_TICKET_URL:
    raise ValueError("TICKET_API_URL is not set in the environment.")

# Priority mapping
priority_order = {
    "Urgent": 0,
    "High": 1,
    "Medium": 2,
    "Low": 3
}

# Thread pool for OpenAI
executor = ThreadPoolExecutor(max_workers=8)

# -----------------------
# VADER multiprocessing setup
# -----------------------
analyzer = None

def init_vader():
    global analyzer
    analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(ticket):
    global analyzer
    content = f"{ticket.get('title', '')} {ticket.get('content', '')}"
    vs = analyzer.polarity_scores(content)
    compound_score = vs['compound']
    scaled_score = int(round((compound_score + 1) * 5))  # [-1,1] → [0,10]
    return scaled_score

def get_vader_sentiment_score(client_id, tickets):
    try:
        with multiprocessing.Pool(processes=8, initializer=init_vader) as pool:
            scores = pool.map(analyze_sentiment, tickets)
        return int(round(sum(scores) / len(scores))) if scores else 0
    except Exception as e:
        return 0

# -----------------------
# Ticket Utilities
# -----------------------
def chunk_tickets(tickets, max_tokens=3000):
    chunks, chunk, token_count = [], [], 0
    for ticket in tickets:
        text = f"{ticket['title']} {ticket['content']}"
        tokens = len(text.split())
        if token_count + tokens > max_tokens:
            chunks.append(chunk)
            chunk, token_count = [ticket], tokens
        else:
            chunk.append(ticket)
            token_count += tokens
    if chunk:
        chunks.append(chunk)
    return chunks

def generate_summary_and_priority(ticket):
    title = ticket.get("title", "")
    content = ticket.get("content", "")
    created_time = ticket.get("created_at", "")

    prompt = f"""You are a smart support ticket analyzer.

Given this support ticket, do the following:
1. Summarize the ticket in 1 short line.
2. Determine the correct priority from (Urgent, High, Medium, Low) based on the title, content, and customer urgency.
3. Determine urgency level from 1 to 5 (1 = most urgent, 5 = least), based on the content and created datetime.

Respond exactly in this format:
Summary: <your one-line summary>
Priority: <Urgent/High/Medium/Low>
Urgency: <1-5>

Title: {title}
Created At: {created_time}
Content: {content}
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        summary, priority, urgency = "N/A", "Low", 6
        for line in output.splitlines():
            if line.lower().startswith("summary:"):
                summary = line.split(":", 1)[1].strip()
            elif line.lower().startswith("priority:"):
                p = line.split(":", 1)[1].strip().capitalize()
                if p in priority_order:
                    priority = p
            elif line.lower().startswith("urgency:"):
                try:
                    urgency = int(line.split(":", 1)[1].strip())
                except:
                    urgency = 6
        return summary, priority, urgency
    except:
        return "Could not summarize", "Low", 6

def batch_generate_summary_and_priority(tickets):
    prompt = "You are a smart support ticket analyzer.\n\n"
    prompt += "For each ticket below, summarize in 1 line, assign a priority (Urgent, High, Medium, Low), and urgency (1-5).\n"
    prompt += "Respond in this format for each ticket:\n"
    prompt += "TicketNumber: <ticket_number>\nSummary: <summary>\nPriority: <priority>\nUrgency: <urgency>\n\n"
    for t in tickets:
        prompt += f"TicketNumber: {t['ticket_number']}\nTitle: {t['title']}\nCreated At: {t['created_at']}\nContent: {t['content']}\n\n"

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        results = {}
        current_ticket = None
        for line in output.splitlines():
            if line.startswith("TicketNumber:"):
                current_ticket = line.split(":", 1)[1].strip()
                results[current_ticket] = {}
            elif line.startswith("Summary:") and current_ticket:
                results[current_ticket]["summary"] = line.split(":", 1)[1].strip()
            elif line.startswith("Priority:") and current_ticket:
                results[current_ticket]["priority"] = line.split(":", 1)[1].strip()
            elif line.startswith("Urgency:") and current_ticket:
                try:
                    results[current_ticket]["urgency"] = int(line.split(":", 1)[1].strip())
                except:
                    results[current_ticket]["urgency"] = 6
        return [
            (
                results.get(t["ticket_number"], {}).get("summary", "N/A"),
                results.get(t["ticket_number"], {}).get("priority", "Low"),
                results.get(t["ticket_number"], {}).get("urgency", 6)
            )
            for t in tickets
        ]
    except:
        return [("Could not summarize", "Low", 6)] * len(tickets)

def get_gpt_sentiment_score(client_id, tickets):
    try:
        chunks = chunk_tickets(tickets, max_tokens=3000)
        scores = []

        for chunk in chunks:
            combined = "\n".join(f"Title: {t['title']}\nContent: {t['content']}" for t in chunk)
            prompt = f"""You are a relationship evaluator. Based on this client's support ticket history, assign a sentiment score between 0 and 10.

Tickets:
{combined}

Respond with only a number between 0 and 10."""

            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            score = float(response.choices[0].message.content.strip())
            scores.append(score)

        return int(round(sum(scores) / len(scores)))
    except:
        return 0

def get_ticket_prefix(client_data):
    for _, item in client_data.items():
        if isinstance(item, dict):
            ticket_number = item.get("cnb_support_ticket_number")
            if ticket_number and "." in ticket_number:
                parts = ticket_number.split('.')
                return ".".join(parts[:2]) + "."
    return None

def batch_process_tickets_parallel(tickets, batch_size=10):
    batches = [tickets[i:i+batch_size] for i in range(0, len(tickets), batch_size)]
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        batch_results = list(executor.map(batch_generate_summary_and_priority, batches))
    for r in batch_results:
        results.extend(r)
    return results

# -----------------------
# Main Endpoint
# -----------------------
@app.route('/prioritize', methods=['POST'])
def prioritize():
    req = request.json
    cnb_ids = req.get("cnb_ids", [])
    new_tickets = req.get("new_tickets", [])

    if not cnb_ids or not isinstance(cnb_ids, list):
        return jsonify({"error": "Send a list of CNB IDs in 'cnb_ids'"}), 400

    new_tickets_map = {}
    for idx, ticket in enumerate(new_tickets):
        if not isinstance(ticket, dict):
            return jsonify({"error": f"Ticket at index {idx} is not a valid object."}), 400

        required_fields = ["client_id", "ticket_number", "content", "title", "created_at", "priority"]
        for field in required_fields:
            if field not in ticket or not str(ticket[field]).strip():
                return jsonify({"error": f"Missing or empty field '{field}' in ticket at index {idx}."}), 400

        if ticket["client_id"] not in cnb_ids:
            return jsonify({"error": f"Ticket at index {idx} has client_id '{ticket['client_id']}' not listed in cnb_ids."}), 400

        if ticket["priority"].capitalize() not in priority_order:
            return jsonify({"error": f"Invalid priority value in ticket at index {idx}. Must be one of {list(priority_order.keys())}."}), 400

        cid = ticket["client_id"]
        new_tickets_map.setdefault(cid, []).append(ticket)

    client_blocks = []

    for cnb_id in cnb_ids:
        try:
            response = requests.post(GNB_TICKET_URL, data={"cnb_id": cnb_id})
            raw = response.text.strip()

            if raw.startswith("<pre>") and raw.endswith("</pre>"):
                raw = raw[5:-6].strip()

            try:
                data = json.loads(raw)
            except:
                continue

            client_name = data.get("cnb_title", "Unknown")
            client_tickets, valid_tickets = [], []
            existing_ticket_numbers = set()
            expected_prefix = get_ticket_prefix(data)

            for _, item in data.items():
                if not isinstance(item, dict):
                    continue

                ticket_number = item.get("cnb_support_ticket_number")
                title = item.get("cnb_support_ticket_title", "")
                content = item.get("cnb_support_ticket_content", "")
                created = item.get("cnb_created_datetime", "")
                priority = item.get("cnb_support_ticket_priority", "Low")

                if not ticket_number or not content:
                    continue

                existing_ticket_numbers.add(ticket_number)

                valid_tickets.append({
                    "ticket_number": ticket_number,
                    "title": title,
                    "content": content,
                    "created_at": created,
                    "priority": priority
                })

            summaries = batch_process_tickets_parallel(valid_tickets, batch_size=10)

            for t, (summary, resolved_priority, urgency) in zip(valid_tickets, summaries):
                final_priority = t["priority"] if t["priority"] in priority_order else resolved_priority
                client_tickets.append({
                    "ticket_number": t["ticket_number"],
                    "client_id": cnb_id,
                    "client_name": client_name,
                    "title": t["title"],
                    "summary": summary,
                    "priority": final_priority,
                    "urgency": urgency,
                    "created_at": t["created_at"],
                    "content": t["content"]
                })

            new_input = new_tickets_map.get(cnb_id, [])
            seen_new = set()
            for idx, ticket in enumerate(new_input):
                t_number = ticket.get("ticket_number", "").strip()
                if not t_number:
                    return jsonify({
                        "error": f"Missing ticket number in new ticket at index {idx} for client ID {cnb_id}."
                    }), 400
                if expected_prefix and not t_number.startswith(expected_prefix):
                    return jsonify({
                        "error": f"Ticket number '{t_number}' at index {idx} does not match expected prefix '{expected_prefix}' for client ID {cnb_id}."
                    }), 400
                if t_number in seen_new:
                    return jsonify({
                        "error": f"Duplicate ticket number '{t_number}' in new tickets for client ID {cnb_id}."
                    }), 400
                if t_number in existing_ticket_numbers:
                    return jsonify({
                        "error": f"Ticket number '{t_number}' already exists in client {cnb_id}'s ticket history."
                    }), 400
                seen_new.add(t_number)

            new_prepared = [{
                "ticket_number": nt.get("ticket_number", f"NEW-{len(client_tickets) + idx + 1}"),
                "title": nt.get("title", ""),
                "content": nt.get("content", ""),
                "created_at": nt.get("created_at", ""),
                "priority": nt.get("priority", "")
            } for idx, nt in enumerate(new_input)]

            new_summaries = batch_process_tickets_parallel(new_prepared, batch_size=10)

            for nt, (summary, resolved_priority, urgency) in zip(new_prepared, new_summaries):
                input_priority = nt.get("priority", "").capitalize()
                final_priority = input_priority if input_priority in priority_order else resolved_priority
                client_tickets.append({
                    "ticket_number": nt["ticket_number"],
                    "client_id": cnb_id,
                    "client_name": client_name,
                    "title": nt["title"],
                    "summary": summary,
                    "priority": final_priority,
                    "urgency": urgency,
                    "created_at": nt["created_at"],
                    "content": nt["content"]
                })

            sentiment_score = get_gpt_sentiment_score(cnb_id, client_tickets)

            def sort_key(x):
                try:
                    created_dt = datetime.strptime(x["created_at"], "%d.%m.%y %H:%M")
                except:
                    created_dt = datetime.min
                return (
                    priority_order.get(x["priority"], 4),
                    x["urgency"],
                    created_dt
                )

            client_tickets.sort(key=sort_key)
            client_blocks.append({
                "sentiment_score": sentiment_score,
                "client_id": cnb_id,
                "tickets": client_tickets
            })

        except:
            continue

    if len(cnb_ids) > 1:
        client_blocks.sort(key=lambda x: -x["sentiment_score"])

    final_output = []
    for block in client_blocks:
        for t in block["tickets"]:
            t["sentiment_score"] = block["sentiment_score"]
            t.pop("content", None)
            t.pop("urgency", None)
            t.pop("created_at", None)
            final_output.append(t)

    return jsonify(final_output)

# -----------------------
# Serve with Waitress
# -----------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()  # Important for Windows
    serve(app, host="0.0.0.0", port=5000)
