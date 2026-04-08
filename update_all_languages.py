import json
import os
import requests

def download_image(url, path):
    if not url.startswith('http'):
        return False
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            with open(path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
    return False

# Path to tutorials.json
TUTORIALS_PATH = "e:/gajalakshmi/project/Suhwa/static/data/tutorials.json"
BASE_IMG_PATH = "e:/gajalakshmi/project/Suhwa/static/img/signs"

# Load existing data if possible
if os.path.exists(TUTORIALS_PATH):
    with open(TUTORIALS_PATH, 'r') as f:
        master_data = json.load(f)
else:
    master_data = {}

# New data to inject/update
updates = {
    "asl": {
        "name": "American Sign Language",
        "region": "United States, Canada, parts of Africa",
        "visual_dictionary_updates": [
            {"category": "Common Words", "items": [
                {"name": "Hello", "image": "https://www.handspeak.com/word/1015/hello-asl.jpg"},
                {"name": "Thank You", "image": "https://www.lifeprint.com/asl101/pages-signs/t/thankyou.gif"},
                {"name": "Please", "image": "https://www.lifeprint.com/asl101/pages-signs/p/please.gif"},
                {"name": "Sorry", "image": "https://www.lifeprint.com/asl101/pages-signs/s/sorry.gif"}
            ]}
        ]
    },
    "bsl": {
        "name": "British Sign Language",
        "region": "United Kingdom",
        "modules": [
            {"id": "bsl_intro", "title": "BSL Basics", "lessons": [{"id": "bsl_hello", "title": "Greeting: Hello", "target": "HELLO", "instructions": ["Wave your hand near your head.", "Smile and make eye contact."], "video_url": ""}]}
        ],
        "visual_dictionary": [
            {"category": "Alphabets", "items": [{"name": chr(i), "image": f"https://placehold.co/400x400?text=BSL+{chr(i)}"} for i in range(65, 91)]},
            {"category": "Numbers", "items": [{"name": str(i), "image": f"https://placehold.co/400x400?text=BSL+{i}"} for i in range(10)]},
            {"category": "Common Words", "items": [
                {"name": "Hello", "image": "https://www.british-sign.co.uk/british-sign-language/wp-content/uploads/2013/01/hello22-892x930.png"},
                {"name": "Thank You", "image": "https://www.british-sign.co.uk/british-sign-language/wp-content/uploads/2013/11/thanks.png"},
                {"name": "Please", "image": "https://www.british-sign.co.uk/british-sign-language/wp-content/uploads/2013/11/please.png"},
                {"name": "Sorry", "image": "https://www.british-sign.co.uk/british-sign-language/wp-content/uploads/2013/11/sorry.png"}
            ]}
        ]
    },
    "isl": {
        "name": "Indian Sign Language",
        "region": "India",
        "modules": [
            {"id": "isl_intro", "title": "ISL Greetings", "lessons": [{"id": "isl_hello", "title": "Namaste / Hello", "target": "HELLO", "instructions": ["Join both palms together (Namaste) or wave.", "Standard greeting in India."], "video_url": ""}]}
        ],
        "visual_dictionary": [
            {"category": "Alphabets", "items": [{"name": chr(i), "image": f"https://placehold.co/400x400?text=ISL+{chr(i)}"} for i in range(65, 91)]},
            {"category": "Numbers", "items": [{"name": str(i), "image": f"https://placehold.co/400x400?text=ISL+{i}"} for i in range(10)]},
            {"category": "Common Words", "items": [
                {"name": "Hello", "image": "https://indiansignlanguage.org/wp-content/uploads/2017/07/hello.jpg"},
                {"name": "Thank You", "image": "https://indiansignlanguage.org/wp-content/uploads/2017/07/thank-you.jpg"},
                {"name": "Please", "image": "https://indiansignlanguage.org/wp-content/uploads/2017/07/please.jpg"},
                {"name": "Sorry", "image": "https://indiansignlanguage.org/wp-content/uploads/2017/07/sorry.jpg"}
            ]}
        ]
    },
    "jsl": {
        "name": "Japanese Sign Language",
        "region": "Japan",
        "modules": [
            {"id": "jsl_intro", "title": "JSL Greetings", "lessons": [{"id": "jsl_hello", "title": "Konnichiwa (Hello)", "target": "HELLO", "instructions": ["Point index fingers of both hands towards each other.", "Move them down slightly like a bow."], "video_url": ""}]}
        ],
        "visual_dictionary": [
            {"category": "Alphabets", "items": [{"name": chr(i), "image": f"https://placehold.co/400x400?text=JSL+{chr(i)}"} for i in range(65, 91)]},
            {"category": "Numbers", "items": [{"name": str(i), "image": f"https://placehold.co/400x400?text=JSL+{i}"} for i in range(10)]},
            {"category": "Common Words", "items": [
                {"name": "Hello", "image": "https://www.kyoto-be.ne.jp/ed-center/gakko/jsl/images/konnichiwa.gif"},
                {"name": "Thank You", "image": "https://www.kyoto-be.ne.jp/ed-center/gakko/jsl/images/arigatou.gif"},
                {"name": "Please", "image": "https://www.kyoto-be.ne.jp/ed-center/gakko/jsl/images/onegai.gif"},
                {"name": "Sorry", "image": "https://www.kyoto-be.ne.jp/ed-center/gakko/jsl/images/gomennasai.gif"}
            ]}
        ]
    },
    "ksl": {
        "name": "Korean Sign Language",
        "region": "Korea",
        "modules": [
             {"id": "ksl_intro", "title": "KSL Greetings", "lessons": [{"id": "ksl_hello", "title": "Annyeong (Hello)", "target": "HELLO", "instructions": ["Clench both fists and move them down.", "Represents a respectful bow."], "video_url": ""}]}
        ],
        "visual_dictionary": [
            {"category": "Alphabets", "items": [{"name": chr(i), "image": f"https://placehold.co/400x400?text=KSL+{chr(i)}"} for i in range(65, 91)]},
            {"category": "Numbers", "items": [{"name": str(i), "image": f"https://placehold.co/400x400?text=KSL+{i}"} for i in range(10)]},
            {"category": "Common Words", "items": [
                {"name": "Hello", "image": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20160120/162426/610_458.jpg"},
                {"name": "Thank You", "image": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20160113/112613/610_458.jpg"},
                {"name": "Please", "image": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20151229/171541/610_458.jpg"},
                {"name": "Sorry", "image": "https://sldict.korean.go.kr/multimedia/multimedia_files/convert/20160113/112613/610_458.jpg"}
            ]}
        ]
    },
    "lsf": {
        "name": "French Sign Language",
        "region": "France",
        "modules": [
             {"id": "lsf_intro", "title": "LSF Greetings", "lessons": [{"id": "lsf_hello", "title": "Bonjour (Hello)", "target": "HELLO", "instructions": ["Hand starts at mouth and moves forward.", "Standard French greeting."], "video_url": ""}]}
        ],
        "visual_dictionary": [
            {"category": "Alphabets", "items": [{"name": chr(i), "image": f"https://placehold.co/400x400?text=LSF+{chr(i)}"} for i in range(65, 91)]},
            {"category": "Numbers", "items": [{"name": str(i), "image": f"https://placehold.co/400x400?text=LSF+{i}"} for i in range(10)]},
            {"category": "Common Words", "items": [
                {"name": "Hello", "image": "https://www.elix-lsf.fr/local/cache-vignettes/L150xH150/arton1-88f6a.png"},
                {"name": "Thank You", "image": "https://www.elix-lsf.fr/local/cache-vignettes/L150xH150/arton2-88f6a.png"},
                {"name": "Please", "image": "https://www.elix-lsf.fr/local/cache-vignettes/L150xH150/arton3-88f6a.png"},
                {"name": "Sorry", "image": "https://www.elix-lsf.fr/local/cache-vignettes/L150xH150/arton4-88f6a.png"}
            ]}
        ]
    }
}

# Merge updates into master_data
for lang, lang_update in updates.items():
    if lang not in master_data:
        master_data[lang] = {
            "name": lang_update.get("name", lang.upper()),
            "region": lang_update.get("region", ""),
            "modules": lang_update.get("modules", []),
            "visual_dictionary": lang_update.get("visual_dictionary", [])
        }
    else:
        # Update name and region if provided
        if "name" in lang_update: master_data[lang]["name"] = lang_update["name"]
        if "region" in lang_update: master_data[lang]["region"] = lang_update["region"]
        
        # Merge modules (simple extend if missing)
        if "modules" in lang_update:
            existing_mod_ids = [m["id"] for m in master_data[lang].get("modules", [])]
            for mod in lang_update["modules"]:
                if mod["id"] not in existing_mod_ids:
                    if "modules" not in master_data[lang]: master_data[lang]["modules"] = []
                    master_data[lang]["modules"].append(mod)
        
        # Merge visual dictionary categories
        if "visual_dictionary" in lang_update:
            for cat_update in lang_update["visual_dictionary"]:
                cat_name = cat_update["category"]
                found_cat = None
                for existing_cat in master_data[lang].get("visual_dictionary", []):
                    if existing_cat["category"] == cat_name:
                        found_cat = existing_cat
                        break
                
                if found_cat:
                    # Merge items and FORCING update of image URL from updates
                    for item in cat_update["items"]:
                        found_item = False
                        for existing_item in found_cat["items"]:
                            if existing_item["name"] == item["name"]:
                                existing_item["image"] = item["image"] # Overwrite with real URL
                                found_item = True
                                break
                        if not found_item:
                            found_cat["items"].append(item)
                else:
                    if "visual_dictionary" not in master_data[lang]: master_data[lang]["visual_dictionary"] = []
                    master_data[lang]["visual_dictionary"].append(cat_update)

    # Special handling for "visual_dictionary_updates" which only adds items to existing categories
    if "visual_dictionary_updates" in lang_update:
        for cat_update in lang_update["visual_dictionary_updates"]:
            cat_name = cat_update["category"]
            found_cat = None
            for existing_cat in master_data[lang].get("visual_dictionary", []):
                if existing_cat["category"] == cat_name:
                    found_cat = existing_cat
                    break
            if found_cat:
                for item in cat_update["items"]:
                    found_item = False
                    for existing_item in found_cat["items"]:
                        if existing_item["name"] == item["name"]:
                            existing_item["image"] = item["image"] # Overwrite
                            found_item = True
                            break
                    if not found_item:
                        found_cat["items"].append(item)
            else:
                if "visual_dictionary" not in master_data[lang]: master_data[lang]["visual_dictionary"] = []
                master_data[lang]["visual_dictionary"].append(cat_update)

# Now download all remote images and update paths to local
for lang, lang_data in master_data.items():
    for cat in lang_data.get("visual_dictionary", []):
        for item in cat.get("items", []):
            url = item["image"]
            if url.startswith('http'):
                ext = url.split(".")[-1] if "." in url else "png"
                if "?" in ext: ext = ext.split("?")[0]
                if len(ext) > 4 or "/" in ext: ext = "png"
                
                local_filename = f"{item['name'].lower().replace(' ', '_')}.{ext}"
                local_path = os.path.join(BASE_IMG_PATH, lang, local_filename)
                
                print(f"Downloading {item['name']} for {lang} from {url}...")
                if download_image(url, local_path):
                    print(f"  Successfully downloaded to {local_path}")
                    item["image"] = f"/static/img/signs/{lang}/{local_filename}"
                else:
                    print(f"  Failed to download {item['name']} for {lang}")
                    # Fallback
                    item["image"] = f"https://placehold.co/400x400?text={lang.upper()}+{item['name']}"

# Save updated data
with open(TUTORIALS_PATH, 'w') as f:
    json.dump(master_data, f, indent=2)

print("Tutorials data merged and images updated.")
