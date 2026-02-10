
import os
import requests

# Dictionary of library name -> license URL
LICENSE_URLS = {
    "PySide6": "https://raw.githubusercontent.com/qt/pyside-pyside-setup/dev/LICENSE.LGPL3",  # LGPL license
    "Torch": "https://raw.githubusercontent.com/pytorch/pytorch/main/LICENSE",
    "Transformers": "https://raw.githubusercontent.com/huggingface/transformers/main/LICENSE",
    "Peft": "https://raw.githubusercontent.com/huggingface/peft/main/LICENSE",
    "CTranslate2": "https://raw.githubusercontent.com/OpenNMT/CTranslate2/master/LICENSE",
    "SentencePiece": "https://raw.githubusercontent.com/google/sentencepiece/master/LICENSE",
    "HuggingFace_Hub": "https://raw.githubusercontent.com/huggingface/huggingface_hub/main/LICENSE",
    "BitsAndBytes": "https://raw.githubusercontent.com/TimDettmers/bitsandbytes/main/LICENSE",
    "Accelerate": "https://raw.githubusercontent.com/huggingface/accelerate/main/LICENSE",
    "Requests": "https://raw.githubusercontent.com/psf/requests/main/LICENSE"
}

OUTPUT_DIR = "licenses"
SUMMARY_FILE = "LICENSE_SUMMARY.txt"

def gather_licenses():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    summary_content = "License Summary for ThinkSub2 LoRA Tools\n========================================\n\n"

    for lib, url in LICENSE_URLS.items():
        print(f"Processing {lib}...")
        try:
            # For direct text files
            if url.endswith("LICENSE") or url.endswith(".txt"):
                response = requests.get(url)
                if response.status_code == 200:
                    filename = f"{lib}_LICENSE.txt"
                    filepath = os.path.join(OUTPUT_DIR, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    summary_content += f"{lib}: {filename}\n"
                else:
                    print(f"Failed to download license for {lib} (Status: {response.status_code})")
                    summary_content += f"{lib}: Download Failed (Check {url})\n"
            else:
                # For web pages usually (like PyQt), just create a link file
                filename = f"{lib}_LICENSE_LINK.txt"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(f"License for {lib} can be found at: {url}\n")
                summary_content += f"{lib}: See {filename}\n"
                
        except Exception as e:
            print(f"Error processing {lib}: {e}")
            summary_content += f"{lib}: Error ({e})\n"

    # Write summary
    with open(os.path.join(OUTPUT_DIR, SUMMARY_FILE), "w", encoding="utf-8") as f:
        f.write(summary_content)
    
    print(f"\nLicense gathering complete. Check the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    gather_licenses()
