import os
import sys

# Ghost Catcher Script
# --------------------
# This file was created to resolve a phantom FileNotFoundError where the 
# system looks for "test_vision_app.py" during a Streamlit launch.
# If this file is executed or imported, it will log the event.

def log_event(message):
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghost_log.txt")
    with open(log_path, "a") as f:
        import datetime
        f.write(f"[{datetime.datetime.now()}] {message}\n")

if __name__ == "__main__":
    log_event(f"Executed as __main__ with args: {sys.argv}")
    print(f"Ghost Catcher alert: test_vision_app.py was executed with args: {sys.argv}")
    # Redirect to the main app
    main_app = os.path.join("Frontend", "Home.py")
    if os.path.exists(main_app):
        print(f"Redirecting to {main_app}...")
        # Avoid recursion – only run if this isn't already a streamlit run call for Home.py
        if "streamlit" not in sys.argv:
            os.system(f"{sys.executable} -m streamlit run {main_app}")
    else:
        print("Main app not found.")
else:
    log_event(f"Imported by {__name__}")
