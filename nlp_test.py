import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp import NLPInference

m = NLPInference()
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
    conf = r.get("confidence", {})

    occasion = f"{r['occasion']} ({conf.get('occasion', 0.0):.3f})" if r["occasion"] else "unknown"
    style    = f"{r['style']} ({conf.get('style', 0.0):.3f})" if r["style"] else "unknown"

    if r["weather"]:
        weather = f"{r['weather_class']} ({r['weather']['temperature']}C) ({conf.get('weather', 0.0):.3f})"
    else:
        weather = "unknown"

    intent = f"{r.get('intent')} ({conf.get('intent', 0.0):.3f})"

    print("  Intent   :", intent)
    print("  Occasion :", occasion)
    print("  Weather  :", weather)
    print("  Style    :", style)
    print()
