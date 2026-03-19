
import sys
import os

# CRITICAL: Force the backend before ANY other imports touch matplotlib
import matplotlib
os.environ['MPLBACKEND'] = 'TkAgg'
matplotlib.use('TkAgg', force=True)

import pickle
import tkinter as tk

# Add project root to path
sys.path.insert(0, r"/Users/apple/Documents/GitHub/DL-Projects/micrograd-visualized")

try:
    from micrograd.debugger import DebuggerUI
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

class TelemetryProxy:
    def __init__(self, data):
        self.loss_history = data['loss_history']
        self.accuracy_history = data['accuracy_history']
        self.batch_history = data['batch_history']
        self.parameter_history = data['parameter_history']
        self.current_step = data['current_step']

def main():
    try:
        if not os.path.exists(r"/Users/apple/Documents/GitHub/DL-Projects/micrograd-visualized/debugger_payload.pkl"):
            print("Payload not found.")
            return
        with open(r"/Users/apple/Documents/GitHub/DL-Projects/micrograd-visualized/debugger_payload.pkl", "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading payload: {e}")
        return

    db = TelemetryProxy(data)
    root = tk.Tk()
    root.title("Neural Replay Debugger")
    
    # macOS focus steal logic
    if sys.platform == "darwin":
        os.system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "Python" to true' 2>/dev/null || true''')
    
    app = DebuggerUI(root, db)
    root.mainloop()

if __name__ == "__main__":
    main()
