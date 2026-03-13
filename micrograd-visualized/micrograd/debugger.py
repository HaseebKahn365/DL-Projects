import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

COLORS = {
    'bg': '#07101d', 'panel': '#0d1929', 'raised': '#111f30',
    'surface': '#162435', 'border': '#1c2e42', 'border2': '#243d57',
    't1': '#cee0f5', 't2': '#5a7fa0', 'tdim': '#2d4560',
    'blue': '#2f8fff', 'cyan': '#00ccf5', 'green': '#00dfa0',
    'amber': '#f5a020', 'red': '#ff3d5a',
    # New Higher Contrast Graph Colors
    'v_bg': '#0f172a', 'v_border': '#38bdf8', 'v_text': '#e2e8f0', 'grad': '#f87171',
    'op_bg': '#1e293b', 'op_border': '#94a3b8', 'op_text': '#fbbf24', 'edge': '#94a3b8'
}

class TrainingDebugger:
    def __init__(self, model):
        self.model = model
        self.loss_history = []
        self.accuracy_history = []
        self.batch_history = []
        self.parameter_history = []
        self.graph_history = []
        self.current_step = 0

    def record(self, step, loss=None, acc=None, batch_indices=None, dataset_size=100):
        if loss is not None:
            self.loss_history.append(float(loss.data if hasattr(loss, 'data') else loss))
            self.graph_history.append(loss)
        if acc is not None:
            self.accuracy_history.append(float(acc))
        
        # BUG 1 FIX: Record Gradients AND input activations for ALL Layers
        # Uses neuron._last_input (set in nn.py during forward pass) for O(1) access
        # to actual input values and gradients — no expensive graph traversal needed.
        layers_snapshot = []
        for layer_idx, layer in enumerate(self.model.layers):
            neurons_snapshot = []
            for neuron in layer.neurons:
                # Capture weights and grads after backward pass
                w_data = [w.data for w in neuron.w]
                w_grads = [w.grad for w in neuron.w]

                # BUG 2 & 3 FIX: Read actual inputs from neuron._last_input
                # These are the Value objects passed into the neuron in the last forward call.
                # For layer 0: these are raw input features (X).
                # For layer k>0: these are outputs of the previous hidden layer (H).
                input_label_prefix = "X" if layer_idx == 0 else "H"
                try:
                    last_input = getattr(neuron, '_last_input', None)
                    if last_input and len(last_input) == len(neuron.w):
                        # _last_input contains Value objects from the LAST forward sample.
                        # After backward(), those Value objects carry accumulated gradients
                        # from the full batch (or whatever was last backprop'd through).
                        input_vals = [xi.data if hasattr(xi, 'data') else float(xi) for xi in last_input]
                        input_grads = [xi.grad if hasattr(xi, 'grad') else 0.0 for xi in last_input]
                    else:
                        input_vals = [0.0] * len(neuron.w)
                        input_grads = [0.0] * len(neuron.w)
                except Exception:
                    input_vals = [0.0] * len(neuron.w)
                    input_grads = [0.0] * len(neuron.w)

                neurons_snapshot.append({
                    "weights": w_data,
                    "weight_grads": w_grads,
                    "bias": neuron.b.data,
                    "bias_grad": neuron.b.grad,
                    "input_vals": input_vals,
                    "input_grads": input_grads,
                    "input_label_prefix": input_label_prefix,
                })
            layers_snapshot.append(neurons_snapshot)
        self.parameter_history.append(layers_snapshot)
        
        # BUG 4 FIX: Use 10 bins (10% each) instead of 5.
        # If batch_indices is None, it means full-batch gradient descent:
        # every sample was used equally — fill all 10 bins with dataset_size/10.
        NUM_SEGMENTS = 10
        segment_size = max(1, dataset_size // NUM_SEGMENTS)
        if batch_indices is None:
            # Full-batch: distribute total samples uniformly across all 10 bins
            bins = [segment_size] * NUM_SEGMENTS
        else:
            bins = [0] * NUM_SEGMENTS
            for idx in batch_indices:
                segment = min(int(idx) // segment_size, NUM_SEGMENTS - 1)
                bins[segment] += 1
        self.batch_history.append(bins)

    def show(self):
        """
        Launches the Debugger UI as a detached process.
        This is designed to be non-blocking and robust for Jupyter Notebooks.
        """
        if not self.loss_history or not self.parameter_history:
            print("[WARNING] No training data recorded yet! Call model.show() AFTER training.")
            return

        import subprocess
        import sys
        import os
        import pickle

        # 1. Prepare Telemetry Payload
        data = {
            'loss_history': self.loss_history,
            'accuracy_history': self.accuracy_history,
            'batch_history': self.batch_history,
            'parameter_history': self.parameter_history,
            'current_step': self.current_step
        }
        
        # Save payload to current directory (overwrites previous)
        telemetry_path = os.path.abspath("debugger_payload.pkl")
        with open(telemetry_path, 'wb') as f:
            pickle.dump(data, f)
            
        # 2. Identify the Package Root
        # Calculated relative to this file (micrograd/debugger.py)
        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # 3. Create a Launcher Script in current directory
        launcher_path = os.path.abspath("debugger_launcher.py")
        log_path = os.path.abspath("debugger.log")
        
        launcher_content = f"""
import sys
import os

# CRITICAL: Force the backend before ANY other imports touch matplotlib
import matplotlib
os.environ['MPLBACKEND'] = 'TkAgg'
matplotlib.use('TkAgg', force=True)

import pickle
import tkinter as tk

# Add project root to path
sys.path.insert(0, r"{pkg_root}")

try:
    from micrograd.debugger import DebuggerUI
except ImportError as e:
    print(f"ImportError: {{e}}")
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
        if not os.path.exists(r"{telemetry_path}"):
            print("Payload not found.")
            return
        with open(r"{telemetry_path}", "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading payload: {{e}}")
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
"""
        with open(launcher_path, 'w') as f:
            f.write(launcher_content)
            
        # 4. Determine Executable
        # Hardcoded to your verified homebrew path that has Tkinter linked
        executable = '/opt/homebrew/opt/python@3.11/bin/python3.11'
        if not os.path.exists(executable):
            executable = sys.executable 
            
        # 5. Launch detached
        try:
            # Prepare environment
            env = os.environ.copy()
            if "MPLBACKEND" in env:
                del env["MPLBACKEND"]
            env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")

            # Use Popen with start_new_session to decouple from the Jupyter kernel process group
            with open(log_path, 'w') as log_file:
                subprocess.Popen(
                    [executable, launcher_path],
                    env=env,
                    start_new_session=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    cwd=os.getcwd(),
                    close_fds=True
                )
            
            print("+" + "-"*58 + "+")
            print("|" + " " * 20 + "DEBUGGER UI LAUNCHED" + " " * 18 + "|")
            print("+" + "-"*58 + "+")
            print(f"| Log: {log_path.split('/')[-1]:<51} |")
            print(f"| Command: {executable} {launcher_path.split('/')[-1]:<18} |")
            print("+" + "-"*58 + "+")
            print("NOTE: If the window doesn't appear, run the 'Command' above in your terminal.")
            
        except Exception as e:
            print(f"[ERROR] Failed to start background process: {e}")
            print(f"Manual fix: {executable} {launcher_path}")






class PanZoomCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg='#04090f', highlightthickness=0, **kwargs)
        self.bind("<ButtonPress-1>", self.pan_start)
        self.bind("<B1-Motion>", self.pan_move)
        self.bind("<ButtonPress-2>", self.pan_start)
        self.bind("<B2-Motion>", self.pan_move)
        self.bind("<ButtonPress-3>", self.pan_start)
        self.bind("<B3-Motion>", self.pan_move)
        self.bind("<MouseWheel>", self.zoom) # Windows/macOS
        self.bind("<Button-4>", self.zoom) # Linux
        self.bind("<Button-5>", self.zoom)
        self.scale_factor = 1.0

    def pan_start(self, event):
        self.scan_mark(event.x, event.y)

    def pan_move(self, event):
        self.scan_dragto(event.x, event.y, gain=1)

    def zoom(self, event):
        if event.num == 4 or getattr(event, 'delta', 1) > 0:
            scale = 1.1
        else:
            scale = 0.9
        
        self.scale_factor *= scale
        x = self.canvasx(event.x)
        y = self.canvasy(event.y)
        self.scale(tk.ALL, x, y, scale, scale)
        self.configure(scrollregion=self.bbox("all"))


class ScrollableFrame(tk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        self.canvas = tk.Canvas(self, bg='#04090f', highlightthickness=0)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg='#04090f')

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", tags="frame")
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        # Mousewheel binding
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
    def _on_mousewheel(self, event):
        if event.num == 4 or event.delta > 0:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:
            self.canvas.yview_scroll(1, "units")

    def on_canvas_configure(self, event):
        self.canvas.itemconfig("frame", width=event.width)


class DebuggerUI:
    def __init__(self, root, debugger):
        self.root = root
        self.db = debugger
        self.step = 0
        self.max_steps = max(0, len(self.db.loss_history) - 1)
        self.playing = False
        self.selected_neuron = (1, 0)

        self.root.title("Neural Replay Debugger")
        self.root.geometry("1400x850")
        self.root.configure(bg=COLORS['bg'])
        self.root.minsize(1000, 700)

        self.setup_layout()
        self.setup_plot()
        self.update_ui()

    def create_panel(self, parent, title, badge=None):
        f = tk.Frame(parent, bg=COLORS['panel'], highlightbackground=COLORS['border'], highlightthickness=1)
        h = tk.Frame(f, bg=COLORS['surface'], height=24)
        h.pack(fill='x')
        
        tl = tk.Label(h, text=title.upper(), fg=COLORS['t2'], bg=COLORS['surface'], font=('JetBrains Mono', 9, 'bold'))
        tl.pack(side='left', padx=10, pady=4)
        
        if badge:
            b = tk.Label(h, text=badge, fg=COLORS['cyan'], bg=COLORS['raised'], font=('JetBrains Mono', 8, 'bold'),
                         highlightbackground=COLORS['border2'], highlightthickness=1)
            b.pack(side='right', padx=10, pady=4)
            b.badge_label = b
            f.badge = b
            
        content = tk.Frame(f, bg=COLORS['panel'])
        content.pack(fill='both', expand=True)
        return f, content

    def setup_layout(self):
        # Base Wrappers - Ensure the main root expands correctly
        self.root.grid_rowconfigure(1, weight=10) # mid wrap gets most space
        self.root.grid_columnconfigure(0, weight=1)

        self.top_wrap = tk.Frame(self.root, bg=COLORS['bg'])
        self.top_wrap.grid(row=0, column=0, sticky='nsew', padx=7, pady=7)
        
        self.mid_wrap = tk.Frame(self.root, bg=COLORS['bg'])
        self.mid_wrap.grid(row=1, column=0, sticky='nsew', padx=7, pady=2)
        
        self.bot_wrap = tk.Frame(self.root, bg=COLORS['panel'], highlightbackground=COLORS['border'], highlightthickness=1)
        self.bot_wrap.grid(row=2, column=0, sticky='nsew', padx=7, pady=7)

        # Config row/cols for Mid Wrap (The panels)
        self.mid_wrap.grid_rowconfigure(0, weight=1)
        # Column weights: Map(1), Arch(2), Params(3), Graph(4)
        for i, weight in enumerate([1, 2, 3, 4]):
            self.mid_wrap.grid_columnconfigure(i, weight=weight)

        # TOP ROW
        self.top_wrap.grid_columnconfigure(0, weight=1, minsize=200)
        self.top_wrap.grid_columnconfigure(1, weight=5)

        p_metrics, c_metrics = self.create_panel(self.top_wrap, "Metrics")
        p_metrics.grid(row=0, column=0, sticky='nsew', padx=(0,5))
        
        # Build Metrics
        c_metrics.configure(padx=10, pady=10)
        self.m_iter = self._make_metric(c_metrics, "Iteration", "0", COLORS['cyan'])
        self.m_loss = self._make_metric(c_metrics, "Loss", "0.000", COLORS['red'])
        self.m_acc = self._make_metric(c_metrics, "Accuracy", "0.0%", COLORS['green'])

        p_chart, c_chart = self.create_panel(self.top_wrap, "Performance", "\u25fc LOSS \u25fc ACC")
        p_chart.grid(row=0, column=1, sticky='nsew')
        self.plot_container = c_chart

        # MID ROW (4 columns)
        # 1. Heatmap
        p_map, c_map = self.create_panel(self.mid_wrap, "Samples")
        p_map.grid(row=0, column=0, sticky='nsew', padx=(0,2))
        c_map.grid_rowconfigure(0, weight=1)
        c_map.grid_columnconfigure(0, weight=1)
        self.map_c = tk.Canvas(c_map, bg=COLORS['panel'], highlightthickness=0)
        self.map_c.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        
        # 2. Architecture
        p_arch, c_arch = self.create_panel(self.mid_wrap, "Network")
        p_arch.grid(row=0, column=1, sticky='nsew', padx=2)
        c_arch.grid_rowconfigure(0, weight=1)
        c_arch.grid_columnconfigure(0, weight=1)
        self.arch_c = PanZoomCanvas(c_arch)
        self.arch_c.grid(row=0, column=0, sticky='nsew')

        # 3. Parameters
        p_param, c_param = self.create_panel(self.mid_wrap, "Matrices", "STEP 0")
        p_param.grid(row=0, column=2, sticky='nsew', padx=2)
        c_param.grid_rowconfigure(0, weight=1)
        c_param.grid_columnconfigure(0, weight=1)
        self.param_badge = p_param.badge
        self.param_scroll = ScrollableFrame(c_param)
        self.param_scroll.grid(row=0, column=0, sticky='nsew')
        self.param_blocks = {} 

        # 4. Computation Graph
        p_graph, c_graph = self.create_panel(self.mid_wrap, "Graph Flow", "N")
        p_graph.grid(row=0, column=3, sticky='nsew', padx=(2,0))
        c_graph.grid_rowconfigure(0, weight=1)
        c_graph.grid_columnconfigure(0, weight=1)
        self.graph_badge = p_graph.badge
        self.graph_c = PanZoomCanvas(c_graph)
        self.graph_c.grid(row=0, column=0, sticky='nsew')

        # BOTTOM CONTROLS
        self.ctrl_inner = tk.Frame(self.bot_wrap, bg=COLORS['panel'])
        self.ctrl_inner.pack(fill='both', expand=True, pady=10)

        self.btn_play = tk.Button(self.ctrl_inner, text="\u25b6 PLAY", command=self.toggle_play, bg=COLORS['blue'], fg='white', font=('JetBrains Mono', 11, 'bold'), borderwidth=0, cursor='hand2')
        self.btn_play.pack(side='left', padx=20)
        
        self.btn_step = tk.Button(self.ctrl_inner, text="STEP \u2192", command=self.step_fwd, bg=COLORS['raised'], fg=COLORS['t1'], font=('JetBrains Mono', 11), borderwidth=1, cursor='hand2')
        self.btn_step.pack(side='left', padx=10)
        
        tk.Label(self.ctrl_inner, text="TIMELINE", fg=COLORS['blue'], bg=COLORS['panel'], font=('JetBrains Mono', 9, 'bold')).pack(side='left', padx=10)
        self.slider = tk.Scale(self.ctrl_inner, from_=0, to=self.max_steps, orient='horizontal',
                               bg=COLORS['panel'], troughcolor=COLORS['raised'], showvalue=False,
                               highlightthickness=0, command=self.on_slider)
        self.slider.pack(side='left', fill='x', expand=True, padx=20)

    def _make_metric(self, parent, label, val, color):
        f = tk.Frame(parent, bg=COLORS['raised'], highlightbackground=COLORS['border'], highlightthickness=1)
        f.pack(fill='x', pady=4)
        tk.Label(f, text=label.upper(), fg=COLORS['tdim'], bg=COLORS['raised'], font=('Inter', 8, 'bold')).pack(anchor='nw', padx=5, pady=2)
        val_lbl = tk.Label(f, text=val, fg=color, bg=COLORS['raised'], font=('JetBrains Mono', 16, 'bold'))
        val_lbl.pack(anchor='w', padx=5, pady=(0,5))
        return val_lbl

    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 1.5), dpi=100)
        self.fig.patch.set_facecolor(COLORS['panel'])
        self.ax.set_facecolor(COLORS['panel'])
        
        self.ax.plot(self.db.loss_history, color=COLORS['blue'], label='Loss', lw=1.5)
        self.ax.plot(self.db.accuracy_history, color=COLORS['green'], label='Acc', lw=1, linestyle='--')
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.spines['left'].set_color(COLORS['border'])
        self.ax.spines['bottom'].set_color(COLORS['border'])
        self.ax.grid(color=COLORS['border'], linestyle='-', alpha=0.5)
        self.ax.tick_params(axis='both', which='major', labelsize=8, colors=COLORS['t2'])
        
        self.cursor_line = self.ax.axvline(x=self.step, color=COLORS['amber'], lw=2)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas_plot.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)

    def update_ui(self):
        # 8. Mandatory Debug Validation
        if not hasattr(self.db, 'parameter_history') or not self.db.parameter_history:
            raise ValueError("Telemetry database is missing 'parameter_history'!")
        
        # 7. Gradient Presence Validation
        s = self.step
        found_grads = False
        for layer in self.db.parameter_history[s]:
            for neuron in layer:
                if any(g != 0 for g in neuron['weight_grads']) or neuron['bias_grad'] != 0:
                    found_grads = True
                    break
            if found_grads: break
            
        if not found_grads and s > 0: # Check only after step 0
            print("[CRITICAL ERROR] No gradients recorded! record() MUST be called AFTER loss.backward().")
        
        # Metrics
        self.m_iter.config(text=str(s))
        self.m_loss.config(text=f"{self.db.loss_history[s]:.4f}")
        self.m_acc.config(text=f"{self.db.accuracy_history[s]:.1f}%")
        self.param_badge.config(text=f"STEP {s}")
        
        # Chart Cursor
        self.cursor_line.set_xdata([s, s])
        self.canvas_plot.draw_idle()
        self.slider.set(s)

        self.draw_heatmap()
        self.draw_arch()
        self.draw_params() # Re-engineered matrix update
        self.draw_graph()

    def draw_heatmap(self):
        self.map_c.delete('all')
        if not self.db.batch_history: return
        
        cw = self.map_c.winfo_width()
        ch = self.map_c.winfo_height()
        if cw < 10 or ch < 10: 
            self.root.after(100, self.draw_heatmap)
            return

        # BUG 4 FIX: 10 bins, cumulative frequency up to current step, % labels
        NUM_BINS = 10
        # Cumulative frequency: sum all batch_history[0..step] bins together
        cumulative = [0] * NUM_BINS
        for s in range(self.step + 1):
            step_bins = self.db.batch_history[s]
            for i in range(min(NUM_BINS, len(step_bins))):
                cumulative[i] += step_bins[i]
        
        total_samples = max(sum(cumulative), 1)
        # Percentage of total samples that fell into each bin (over all steps so far)
        pct_per_bin = [int(round(100.0 * cumulative[i] / total_samples)) for i in range(NUM_BINS)]
        
        mx = max(max(cumulative), 1)
        top_pad = 10
        h_cell = (ch - top_pad) / NUM_BINS
        
        for i in range(NUM_BINS):
            # Opacity-based color (dark navy -> bright cyan)
            opacity = cumulative[i] / mx
            r_c = int(13 + (56-13) * (0.15 + opacity*0.85))
            g_c = int(25 + (189-25) * (0.15 + opacity*0.85))
            b_c = int(41 + (248-41) * (0.15 + opacity*0.85))
            color = f'#{r_c:02x}{g_c:02x}{b_c:02x}'
            
            y0 = top_pad + i * h_cell
            y1 = top_pad + (i+1) * h_cell - 2
            self.map_c.create_rectangle(5, y0, cw-5, y1, fill=color, outline='#1c2e42')
            
            # Bin range label (small, top of cell)
            bin_label = f"{i*10}–{(i+1)*10}%"
            cy = (y0 + y1) / 2
            self.map_c.create_text(cw/2, cy - h_cell*0.2, text=bin_label, fill='#5a7fa0', font=('Inter', 6))
            # Percentage label in center of each cell
            self.map_c.create_text(cw/2, cy + h_cell*0.2, text=f"{pct_per_bin[i]}%", fill='white', font=('JetBrains Mono', 8, 'bold'))

    def get_val_color(self, v, max_abs=1.0):
        # Normalized Intensity logic: value > 0 -> blue, value < 0 -> red
        v = max(-max_abs, min(max_abs, v)) # clamp
        norm = abs(v) / max_abs
        intensity = int(80 + norm * 175)
        if v > 0: return f'#0000{intensity:02x}' # Blue scale
        if v < 0: return f'#{intensity:02x}0000' # Red scale
        return '#1e293b' # Neutral dark gray for zero

    def draw_arch(self):
        c = self.arch_c
        c.delete('all')
        if not self.db.parameter_history: return
        first = self.db.parameter_history[0]
        in_s = len(first[0][0]['weights'])
        L = [in_s] + [len(l) for l in first]
        
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 10:
            self.root.after(100, self.draw_arch)
            return

        lx = [(i+1)*w/(len(L)+1) for i in range(len(L))]
        colors = [COLORS['t2'], COLORS['blue'], COLORS['amber']]
        
        def gC(i): return colors[0] if i==0 else (colors[-1] if i==len(L)-1 else colors[1])

        # Edges
        for l in range(len(L)-1):
            n1 = L[l]; n2 = L[l+1]
            for i in range(n1):
                y1 = (i+1)*h/(n1+1)
                for j in range(n2):
                    y2 = (j+1)*h/(n2+1)
                    # highlight check
                    sel = self.selected_neuron and (
                        (self.selected_neuron[0]==l+1 and self.selected_neuron[1]==j) or
                        (self.selected_neuron[0]==l and self.selected_neuron[1]==i)
                    )
                    c.create_line(lx[l], y1, lx[l+1], y2, fill=COLORS['blue'] if sel else COLORS['border'], width=1.5 if sel else 0.5)

        # Nodes
        for l in range(len(L)):
            num = L[l]
            for i in range(num):
                y = (i+1)*h/(num+1)
                is_sel = self.selected_neuron and self.selected_neuron[0] == l and self.selected_neuron[1] == i
                r = 10 if is_sel else 6
                c.create_oval(lx[l]-r, y-r, lx[l]+r, y+r, fill=gC(l), outline=COLORS['amber'] if is_sel else gC(l), width=2 if is_sel else 0)
                
                # Tag and bind
                tag = f"n_{l}_{i}"
                c.addtag_enclosed(tag, lx[l]-r-2, y-r-2, lx[l]+r+2, y+r+2)
                if l > 0:
                    c.tag_bind(tag, "<Button-1>", lambda e, ln=l, nn=i: self.on_node_click(ln, nn))

    def on_node_click(self, l, i):
        self.selected_neuron = (l, i)
        self.graph_badge.config(text=f"L{l} \u00b7 N{i}")
        self.draw_arch()
        self.draw_graph()

    def draw_params(self):
        # 3. Matrix Reconstruction Logic
        f = self.param_scroll.scrollable_frame
        step_params = self.db.parameter_history[self.step]
        
        # Determine global max weight for normalization
        max_abs_w = 1e-6
        for layer in step_params:
            for n in layer:
                max_abs_w = max(max_abs_w, max([abs(w) for w in n['weights']] + [abs(n['bias'])]))

        if not f.winfo_children():
            # Initial setup (Do not recreate each step)
            for l_idx, layer_data in enumerate(step_params):
                cols = len(layer_data[0]['weights']) 
                rows = len(layer_data)
                
                lf = tk.Frame(f, bg='#04090f', pady=15)
                lf.pack(fill='x', padx=10)
                
                tk.Label(lf, text=f"LAYER {l_idx+1}", fg=COLORS['amber'], bg='#04090f', font=('JetBrains Mono', 9, 'bold')).pack(anchor='w')
                
                # Weights Matrix Grid
                tk.Label(lf, text="WEIGHT MATRIX", fg=COLORS['t2'], bg='#04090f', font=('Inter', 7, 'bold')).pack(anchor='w', pady=(5,0))
                w_frame = tk.Frame(lf, bg=COLORS['panel'])
                w_frame.pack(anchor='w', pady=2)
                
                row_widgets = []
                for r in range(rows):
                    col_widgets = []
                    for c in range(cols):
                        lbl = tk.Label(w_frame, text="0.000", width=7, height=2, highlightthickness=1,
                                       highlightbackground=COLORS['border'], font=('JetBrains Mono', 7), fg='white')
                        lbl.grid(row=r, column=c, padx=1, pady=1)
                        col_widgets.append(lbl)
                    row_widgets.append(col_widgets)
                
                # Bias Vector Grid
                tk.Label(lf, text="BIAS VECTOR", fg=COLORS['t2'], bg='#04090f', font=('Inter', 7, 'bold')).pack(anchor='w', pady=(5,0))
                b_frame = tk.Frame(lf, bg=COLORS['panel'])
                b_frame.pack(anchor='w', pady=2)
                
                bias_widgets = []
                for r in range(rows):
                    lbl = tk.Label(b_frame, text="0.000", width=7, height=2, highlightthickness=1,
                                   highlightbackground=COLORS['border'], font=('JetBrains Mono', 7), fg='white')
                    lbl.grid(row=r, column=0, padx=1, pady=1)
                    bias_widgets.append(lbl)
                
                self.param_blocks[l_idx] = (row_widgets, bias_widgets)

        # Update values efficiently
        for l_idx, (w_grids, b_grids) in self.param_blocks.items():
            layer_data = step_params[l_idx]
            for r_idx, neuron in enumerate(layer_data):
                b_val = neuron['bias']
                b_grids[r_idx].config(text=f"{b_val:.3f}", bg=self.get_val_color(b_val, max_abs_w))
                for c_idx, w_val in enumerate(neuron['weights']):
                    if c_idx < len(w_grids[r_idx]):
                        w_grids[r_idx][c_idx].config(text=f"{w_val:.3f}", bg=self.get_val_color(w_val, max_abs_w))

    def draw_graph(self):
        c = self.graph_c
        c.delete('all')
        if not self.selected_neuron or self.selected_neuron[0] == 0: return
        w = c.winfo_width()
        h = c.winfo_height()
        if w < 50 or h < 50:
            self.root.after(100, self.draw_graph)
            return

        li, ni = self.selected_neuron[0]-1, self.selected_neuron[1]
        p = self.db.parameter_history[self.step][li][ni]
        weights = p['weights']
        bias = p['bias']
        grads = p['weight_grads']
        b_grad = p['bias_grad']
        num_inputs = len(weights)

        # 7. Computation Graph Scaling Rule
        node_size = max(20, min(w, h) / 20)
        v_gap = h / (num_inputs + 1)
        mid_y = h / 2
        
        # Stages (Horizontal stages, vertical branches)
        # 0: Inputs/Weight, 1: Mul Ops, 2: Addition node, 3: Activation, 4: Final Output
        col_x = [w*0.12, w*0.35, w*0.58, w*0.78, w*0.92]

        def draw_val_node(x, y, label, val, grad, border_color='#38bdf8'):
            # Value Nodes: Bg #0f172a, Border requested color, Value #e2e8f0, Grad #f87171
            rw, rh = node_size * 1.5, node_size * 0.8
            c.create_rectangle(x-rw, y-rh, x+rw, y+rh, fill='#0f172a', outline=border_color, width=2)
            c.create_text(x, y-rh/2 - 2, text=label, fill='#e2e8f0', font=('Inter', 7, 'bold'))
            c.create_text(x, y, text=f"{val:.3f}", fill='#e2e8f0', font=('JetBrains Mono', 8))
            c.create_text(x, y+rh/2 + 2, text=f"g:{grad:.4f}", fill='#f87171', font=('JetBrains Mono', 7))

        def draw_op_node(x, y, op):
            # Op Nodes: Fill #1e293b, Border #94a3b8, Operator #fbbf24
            r = node_size * 0.6
            c.create_oval(x-r, y-r, x+r, y+r, fill='#1e293b', outline='#94a3b8', width=2)
            c.create_text(x, y, text=op, fill='#fbbf24', font=('JetBrains Mono', 14, 'bold'))

        # BUG 2 & 3 FIX: Use real input values/grads and correct label prefix (X vs H)
        input_vals = p.get('input_vals', [0.0] * num_inputs)
        input_grads = p.get('input_grads', [0.0] * num_inputs)
        input_prefix = p.get('input_label_prefix', 'X')

        # Layout: two sub-columns for W node (left) and Input node (right) within col_x[0]
        w_col_x = col_x[0] - node_size * 1.2
        x_col_x = col_x[0] + node_size * 1.2

        # 1. Inputs and Mul Nodes
        for i in range(num_inputs):
            yy = (i+1) * v_gap
            
            # BUG 1 FIX: W Node — show real weight value AND gradient
            draw_val_node(w_col_x, yy, f"W{i}", weights[i], grads[i], border_color='#38bdf8')

            # BUG 2 & 3 FIX: Input Node — real input value/grad, correct prefix
            draw_val_node(x_col_x, yy, f"{input_prefix}{i}", input_vals[i], input_grads[i], border_color='#a78bfa')
            
            # Mul Operation
            draw_op_node(col_x[1], yy, "\u00d7")
            
            # Edges
            c.create_line(w_col_x+node_size*1.5, yy, col_x[1]-node_size*0.6, yy, fill='#94a3b8', width=2, arrow=tk.LAST)
            c.create_line(x_col_x+node_size*1.5, yy, col_x[1]-node_size*0.6, yy, fill='#a78bfa', width=1, dash=(3,2))
            c.create_line(col_x[1]+node_size*0.6, yy, col_x[2]-node_size*0.6, mid_y, fill='#94a3b8', width=2, arrow=tk.LAST)

        # 2. Addition node stage
        draw_op_node(col_x[2], mid_y, "+")
        
        # 3. Bias branching into Add
        bias_y = mid_y - (h*0.25)
        draw_val_node(col_x[2], bias_y, "BIAS", bias, b_grad, border_color='#fbbf24')
        c.create_line(col_x[2], bias_y+node_size*0.8, col_x[2], mid_y-node_size*0.6, fill='#94a3b8', width=2, arrow=tk.LAST)

        # 4. Activation stage
        is_relu = "ReLU" in str(self.db.model.layers[li].neurons[ni])
        draw_op_node(col_x[3], mid_y, "ReLU" if is_relu else "Lin")
        c.create_line(col_x[2]+node_size*0.6, mid_y, col_x[3]-node_size*0.6, mid_y, fill='#94a3b8', width=2, arrow=tk.LAST)

        # 5. Output stage — compute output value from weights and inputs
        try:
            out_val = sum(weights[j] * input_vals[j] for j in range(num_inputs)) + bias
        except Exception:
            out_val = 0.0
        draw_val_node(col_x[4], mid_y, "OUT", out_val, 0.0, border_color='#00dfa0')
        c.create_line(col_x[3]+node_size*0.6, mid_y, col_x[4]-node_size*1.4, mid_y, fill='#94a3b8', width=2, arrow=tk.LAST)

    def on_slider(self, val):
        self.step = int(val)
        self.update_ui()

    def step_fwd(self):
        self.step = (self.step + 1) % (self.max_steps + 1)
        self.update_ui()

    def toggle_play(self):
        self.playing = not self.playing
        self.btn_play.config(text="\u23f8 PAUSE" if self.playing else "\u25b6 PLAY", 
                             bg=COLORS['surface'] if self.playing else COLORS['blue'])
        if self.playing:
            self.play_loop()

    def play_loop(self):
        if self.playing:
            self.step_fwd()
            self.root.after(145, self.play_loop)

