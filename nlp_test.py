import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp import NLPInference

m = NLPInference("nlp/saved_models")
print("NLP Manual Test — type a query and press Enter. Type 'quit' to exit.")
print("-" * 55)

while True:
    try:
        q = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        break

    if q.lower() in ("quit", "exit", "q"):
        break
    if not q:
        continue

    r = m.predict(q)

    occasion = r["occasion"] or "unknown"
    style    = r["style"]    or "unknown"

    if r["weather"]:
        weather = "{} ({}C)".format(r["weather_class"], r["weather"]["temperature"])
    else:
        weather = "unknown"

    print("  Occasion :", occasion)
    print("  Weather  :", weather)
    print("  Style    :", style)
    print()
