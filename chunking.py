import re
import json

def parse_markdown(file_path):
    """
    Parses a Markdown file into structured chunks for RAG.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    chunks = []
    current_bab = "None"  # باب (Section)
    current_fasl = "None"  # فصل (Chapter)
    current_mada = "None"  # مادة (Law/Article)
    current_content = []

    for line in lines:
        line = line.strip()

        if line.startswith("# "):  # باب (Section)
            # Save the previous content as a chunk
            if current_mada:
                chunks.append({
                    "content": "\n".join(current_content),
                    "metadata": {
                        "باب": current_bab,
                        "فصل": current_fasl,
                        "مادة": current_mada,
                        "jurisdiction": "Saudi Arabia"
                    }
                })
            # Reset variables for the new باب
            current_bab = line.replace("# ", "").strip()
            current_fasl = "None"
            current_mada = "None"
            current_content = []

        elif line.startswith("## "):  # فصل (Chapter)
            # Save the previous content as a chunk
            if current_mada:
                chunks.append({
                    "content": "\n".join(current_content),
                    "metadata": {
                        "باب": current_bab,
                        "فصل": current_fasl,
                        "مادة": current_mada,
                        "jurisdiction": "Saudi Arabia"
                    }
                })
            # Update فصل
            current_fasl = line.replace("## ", "").strip()
            current_mada = "None"
            current_content = []

        elif line.startswith("### "):  # مادة (Law/Article)
            # Save the previous content as a chunk
            if current_mada:
                chunks.append({
                    "content": "\n".join(current_content),
                    "metadata": {
                        "باب": current_bab,
                        "فصل": current_fasl,
                        "مادة": current_mada,
                        "jurisdiction": "Saudi Arabia"
                    }
                })
            # Update مادة
            current_mada = line.replace("### ", "").strip()
            current_content = []

        elif line:  # Regular content
            current_content.append(line)

    # Save the last chunk
    if current_mada:
        chunks.append({
            "content": "\n".join(current_content),
            "metadata": {
                "باب": current_bab,
                "فصل": current_fasl,
                "مادة": current_mada,
                "jurisdiction": "Saudi Arabia"
            }
        })

    return chunks

# Parse the file
file_path = "output.md"  # Replace with your Markdown file path
chunks = parse_markdown(file_path)

# Save the chunks to a JSON file
output_file = "labor_laws_chunks.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=4)

print(f"Chunking completed and saved to '{output_file}'")
