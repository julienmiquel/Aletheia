
import html

# Mimic the logic in app.py
def mimic_app_logic(text_input):
    # Simulate GLTR results (in real app, this comes from the model)
    # The model tokenizes the input.
    # If input is "<script>alert(1)</script>", the tokens will reconstruct that.

    # Let's say we have a mock token list that represents the input
    # For simplicity, let's assume the text is split by spaces or just characters
    # But effectively, the tokenizer.decode() returns the string chunk.

    # Case: The text contains HTML special chars
    dangerous_text = text_input

    # Simulating what gltr_data might look like
    gltr_data = [
        {'token': dangerous_text, 'rank': 1, 'prob': 0.9, 'bucket': 'Green'}
    ]

    html_content = "<div style='line-height: 1.8; font-family: monospace;'>"

    colors = {
        "Green": "#d4edda",   # Light Green
    }

    for item in gltr_data:
        token = item['token']
        # Clean up GPT-2 tokens (remove Ġ)
        display_token = token.replace('Ġ', ' ')

        # FIXED LINE:
        safe_token = html.escape(display_token)

        bg_color = colors.get(item['bucket'], "#ffffff")

        # Tooltip with Rank
        tooltip = f"Rank: {item['rank']} | Prob: {item['prob']:.4f}"

        # VULNERABLE LINE (now using safe_token):
        html_content += f"<span style='background-color: {bg_color}; padding: 2px 4px; margin: 0 2px; border-radius: 4px;' title='{tooltip}'>{safe_token}</span>"

    html_content += "</div>"
    return html_content

input_payload = "<script>alert('XSS')</script>"
output = mimic_app_logic(input_payload)

print("Input:", input_payload)
print("Output:", output)

if "<script>" in output:
    print("\n[!] VULNERABILITY CONFIRMED: Script tag preserved in output.")
elif "&lt;script&gt;" in output:
    print("\n[+] SUCCESS: Script tag escaped.")
else:
    print("\n[?] Unexpected output.")
