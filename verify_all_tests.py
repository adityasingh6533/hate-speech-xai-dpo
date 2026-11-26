import sys
import os

# Test verification across diverse English & Hinglish inputs
import src.inference_engine as ie

TEST_CASES = [
    # --- 1. TARGETED HINGLISH HATE & THREATS (Must be HATEFUL, P > 0.65) ---
    ("tum log desh ke gaddar ho niklo yahan se", True, "desh ke gaddar"),
    ("is community ke sabhi log aatankwadi hote hain", True, "aatankwadi"),
    ("ye katwe hamara haq kha rahe hain inko bhagao", True, "katwe"),
    ("saale kutte bhaunk mat apni aukaat mein reh", True, "saale/kutte/aukaat"),
    ("gandi naali ke keede ho tum sab maro saalo", True, "gandi naali ke keede"),
    ("shakal dekh ke ghin aati hai ghatiya insaan", True, "shakal dekh ke ghin"),
    ("teri aukaat kya hai do kaudi ke aadmi", True, "aukaat kya hai"),
    ("chutiye nikal yahan se varna laat maar ke nikalenge", True, "chutiye/laat maar ke"),
    ("tere jaise suar ko yahan rehne ka haq nahi hai", True, "suar/rehne ka haq nahi"),

    # --- 2. BENIGN CONVERSATIONAL & COLLOQUIAL HINGLISH (Must be NOT_HATEFUL, P < 0.25) ---
    ("bhai tu bilkul pagal hai kya mast movie thi yaar maza aa gaya", False, "bhai/mast/maza"),
    ("bhai aaj ka match shandaar tha kya zabardast khela", False, "bhai/shandaar/zabardast"),
    ("yaar party kab de raha hai treat to banti hai", False, "yaar/party/treat"),
    ("kya haal hai bhai sab badhiya chal raha hai na", False, "bhai/badhiya"),
    ("khana sach me bohot lajawab aur swadisht tha shukriya", False, "lajawab/swadisht/shukriya"),
    ("diwali aur eid sabhi dost milkar dhoom dhaam se manate hain", False, "diwali/eid/dost"),
    ("bhai thoda help kar de assignment submit karna hai", False, "bhai/help"),

    # --- 3. ENGLISH BENCHMARK SAMPLES ---
    ("Good morning, have a wonderful day with your family.", False, "morning/family"),
    ("Muslim and Hindu neighbors celebrated the festival peacefully.", False, "peaceful festival"),
    ("You are disgusting subhuman trash, get out of this country.", True, "subhuman/trash"),
]

print(f"Running comprehensive evaluation across {len(TEST_CASES)} benchmarks...\n")

all_passed = True
for idx, (text, exp_hate, notes) in enumerate(TEST_CASES, 1):
    pred = ie.predict_text(text)
    is_hate = pred["is_hate"]
    p_hate = pred["p_hate"]
    conf = pred["confidence"]
    label = pred["label"]
    
    passed = (is_hate == exp_hate)
    if not passed:
        all_passed = False
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] #{idx:02d} P(hate)={p_hate:.4f} | Conf={conf*100:5.1f}% | Pred: {label:20} | Exp: {'HATEFUL' if exp_hate else 'BENIGN':7} | Note: {notes}")

print("\n" + "="*70)
print(f"OVERALL BENCHMARK RESULT: {'>>> ALL TESTS PASSED <<<' if all_passed else '>>> SOME TESTS FAILED <<<'}")
print("="*70)

if not all_passed:
    sys.exit(1)
