import src.inference_engine as ie

cases = [
    ("Good morning, have a wonderful day with your family.", "NOT_HATEFUL", 0.20, lambda p: p < 0.20),
    ("Muslim and Hindu neighbors celebrated the festival peacefully.", "NOT_HATEFUL", 0.25, lambda p: p < 0.25),
    ("You are disgusting subhuman trash, get out of this country.", "HATEFUL", 0.75, lambda p: p > 0.75),
    ("Saale ghatiya insaan teri aukaat kya hai niklo yahan se", "HATEFUL", 0.65, lambda p: p > 0.65)
]

print("=" * 77)
print("4-CASE AUTOMATED VERIFICATION TEST REPORT")
print("=" * 77)
all_passed = True
for i, (text, expected_label, threshold, check_fn) in enumerate(cases, 1):
    res = ie.predict_text(text)
    p_hate = res["p_hate"]
    label = res["label"]
    passed = check_fn(p_hate) and (expected_label in label)
    all_passed = all_passed and passed
    status_str = "PASS [OK]" if passed else "FAIL [X]"
    print(f'Test Case {i}: "{text}"')
    print(f'  -> Hate Probability: {p_hate:.4f}')
    print(f'  -> Predicted Label:  {label}')
    print(f'  -> Expected:         {expected_label} (threshold criteria: {"<" if "NOT" in expected_label else ">"} {threshold})')
    print(f'  -> Status:           {status_str}')
    
    # Verify dynamic XAI rationales
    exp = ie.explain_tokens(text)
    rat = ie.generate_dpo_rationale(text, res, exp)
    first_line_rat = rat.split('\n')[0]
    print(f'  -> Rationale Snippet: {first_line_rat}')
    print("-" * 77)

print("FINAL VERIFICATION RESULT:", "ALL 4 TEST CASES PASSED SUCCESSFULLY!" if all_passed else "SOME TESTS FAILED")
print("=" * 77)
