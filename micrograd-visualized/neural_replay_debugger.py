import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time
import threading
import random

# For generating real data
from micrograd.engine import Value
from micrograd.nn import MLP

# --- Constants & Theming ---
COLORS = {
    'bg': '#07101d',
    'panel': '#0d1929',
    'raised': '#162435',
    'border': '#1c2e42',
    'text': '#cee0f5',
    'text_dim': '#5a7fa0',
    'blue': '#2f8fff',
    'cyan': '#00ccf5',
    'green': '#00dfa0',
    'amber': '#f5a020',
    'red': '#ff3d5a',
    'neutral': '#1c2e42'
}

FONT_MAIN = ('Inter', 10)
FONT_MONO = ('JetBrains Mono', 9)
FONT_BOLD = ('Inter', 10, 'bold')

class Logger:
    def __init__(self):
        self.loss_history = []
        self.accuracy_history = []
        self.heatmap_history = []
        self.parameter_history = [] # [step][layer][neuron] = {weights: [], bias: v}

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_window = None

    def show_tip(self):
        if self.tip_window or not self.text:
            return
        x, y, _cx, cy = self.widget.bbox("current")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify=tk.LEFT,
                      background="#000000", foreground=COLORS['cyan'],
                      relief=tk.SOLID, borderwidth=1,
                      font=FONT_MONO)
        label.pack(ipadx=1)

    def hide_tip(self):
        tw = self.tip_window
        self.tip_window = None
        if tw:
            tw.destroy()

class MatrixCanvas(tk.Canvas):
    """Component for rendering weight matrices and bias vectors."""
    def __init__(self, master, rows, cols, title, **kwargs):
        super().__init__(master, bg=COLORS['panel'], highlightthickness=0, **kwargs)
        self.rows = rows
        self.cols = cols
        self.title = title
        self.cell_size = 28
        self.padding = 35
        self.tooltips = {}
        
    def render(self, data, is_bias=False):
        self.delete('all')
        self.create_text(5, 5, text=self.title, fill=COLORS['cyan'], anchor='nw', font=FONT_BOLD)
        
        for r in range(self.rows):
            for c in range(self.cols):
                val = data[r][c] if not is_bias else data[r]
                color = self.get_color(val)
                x0 = self.padding + c * (self.cell_size + 2)
                y0 = self.padding + r * (self.cell_size + 2)
                x1 = x0 + self.cell_size
                y1 = y0 + self.cell_size
                
                rect_id = self.create_rectangle(x0, y0, x1, y1, fill=color, outline=COLORS['border'], tags="cell")
                self.create_text((x0+x1)/2, (y0+y1)/2, text=f"{val:.2f}", 
                                fill='white' if abs(val) > 0.4 else COLORS['text_dim'], font=FONT_MONO)
                
                self.tag_bind(rect_id, "<Enter>", lambda e, v=val: self.show_val_tip(e, v))
                self.tag_bind(rect_id, "<Leave>", lambda e: self.hide_val_tip())

    def show_val_tip(self, event, val):
        x = self.winfo_rootx() + event.x + 10
        y = self.winfo_rooty() + event.y + 10
        self.tip = tk.Toplevel(self)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self.tip, text=f"Value: {val:.6f}", bg="black", fg=COLORS['cyan'], font=FONT_MONO, borderwidth=1, relief="solid").pack()

    def hide_val_tip(self):
        if hasattr(self, 'tip'):
            self.tip.destroy()

    def get_color(self, val):
        # Red spectrum for negative, Blue for positive
        mag = min(abs(val), 1.0)
        if val < 0:
            r = int(28 + 200 * mag)
            return f'#{r:02x}2040'
        else:
            b = int(28 + 200 * mag)
            return f'#2040{b:02x}'

class NeuralReplayDebugger:
    def __init__(self, root, logger):
        self.root = root
        self.root.title("Neural Replay Debugger")
        self.root.geometry("1400x900")
        self.root.configure(bg=COLORS['bg'])
        
        self.logger = logger
        self.current_step = 0
        self.playing = False
        self.selected_neuron = (1, 0) # (layer_idx, neuron_idx)
        
        self.setup_layout()
        self.setup_styles()
        self.update_ui()
        
    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('default')
        style.configure('TFrame', background=COLORS['bg'])
        style.configure('Panel.TFrame', background=COLORS['panel'], borderwidth=1, relief='flat')
        
    def setup_layout(self):
        # Define proportions
        self.root.grid_rowconfigure(0, weight=25)
        self.root.grid_rowconfigure(1, weight=60)
        self.root.grid_rowconfigure(2, weight=15)
        self.root.grid_columnconfigure(0, weight=1)

        # --- Top Section: Training Overview ---
        top_frame = tk.Frame(self.root, bg=COLORS['bg'], height=200)
        top_frame.grid(row=0, column=0, sticky='nsew', padx=5, pady=5)
        top_frame.grid_columnconfigure(0, weight=1)
        top_frame.grid_columnconfigure(1, weight=4)
        
        # Metrics Sub-panel
        self.metrics_sub = tk.Frame(top_frame, bg=COLORS['panel'], highlightbackground=COLORS['border'], highlightthickness=1)
        self.metrics_sub.grid(row=0, column=0, sticky='nsew', padx=2, pady=2)
        
        self.lbl_iter = tk.Label(self.metrics_sub, text="ITER: 0", fg=COLORS['cyan'], bg=COLORS['panel'], font=FONT_BOLD)
        self.lbl_iter.pack(pady=5, anchor='w', padx=10)
        self.lbl_loss = tk.Label(self.metrics_sub, text="LOSS: 0.0000", fg=COLORS['red'], bg=COLORS['panel'], font=FONT_BOLD)
        self.lbl_loss.pack(pady=5, anchor='w', padx=10)
        self.lbl_acc = tk.Label(self.metrics_sub, text="ACC: 0.0%", fg=COLORS['green'], bg=COLORS['panel'], font=FONT_BOLD)
        self.lbl_acc.pack(pady=5, anchor='w', padx=10)
        
        # Plot Sub-panel
        self.plot_sub = tk.Frame(top_frame, bg=COLORS['bg'])
        self.plot_sub.grid(row=0, column=1, sticky='nsew', padx=2, pady=2)
        self.setup_plot()

        # --- Middle Section ---
        mid_frame = tk.Frame(self.root, bg=COLORS['bg'])
        mid_frame.grid(row=1, column=0, sticky='nsew', padx=5)
        for i in range(4): mid_frame.grid_columnconfigure(i, weight=1)
        mid_frame.grid_rowconfigure(0, weight=1)
        
        # 1. Heatmap Panel
        self.hmap_canvas = tk.Canvas(mid_frame, bg=COLORS['panel'], highlightthickness=1, highlightbackground=COLORS['border'])
        self.hmap_canvas.grid(row=0, column=0, sticky='nsew', padx=2)
        
        # 2. Architecture Panel
        self.arch_canvas = tk.Canvas(mid_frame, bg='#04090f', highlightthickness=1, highlightbackground=COLORS['border'])
        self.arch_canvas.grid(row=0, column=1, sticky='nsew', padx=2)
        self.arch_canvas.bind("<Button-1>", self.on_arch_click)
        
        # 3. Parameter Inspector Panel
        self.params_frame = tk.Frame(mid_frame, bg='#04090f', highlightthickness=1, highlightbackground=COLORS['border'])
        self.params_frame.grid(row=0, column=2, sticky='nsew', padx=2)
        self.setup_param_inspector()

        # 4. Computation Graph Panel
        self.graph_canvas = tk.Canvas(mid_frame, bg='#04090f', highlightthickness=1, highlightbackground=COLORS['border'])
        self.graph_canvas.grid(row=0, column=3, sticky='nsew', padx=2)

        # --- Bottom Section: Controls ---
        bot_frame = tk.Frame(self.root, bg=COLORS['panel'], height=60, highlightbackground=COLORS['border'], highlightthickness=1)
        bot_frame.grid(row=2, column=0, sticky='nsew', padx=5, pady=5)
        
        self.play_btn = tk.Button(bot_frame, text="▶ PLAY", command=self.toggle_play, font=FONT_BOLD, 
                                 bg=COLORS['blue'], fg='white', activebackground=COLORS['cyan'], borderwidth=0)
        self.play_btn.pack(side='left', padx=20, pady=10)
        
        self.step_btn = tk.Button(bot_frame, text="STEP →", command=self.step_forward, font=FONT_BOLD,
                                 bg=COLORS['raised'], fg=COLORS['text'], borderwidth=1, highlightbackground=COLORS['border'])
        self.step_btn.pack(side='left', padx=5, pady=10)
        
        self.timeline = tk.Scale(bot_frame, from_=0, to=len(self.logger.loss_history)-1, orient='horizontal',
                                bg=COLORS['panel'], fg=COLORS['text_dim'], highlightthickness=0,
                                troughcolor=COLORS['raised'], activebackground=COLORS['cyan'],
                                command=self.on_slider_move)
        self.timeline.pack(side='left', fill='x', expand=True, padx=20)

    def setup_plot(self):
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(8, 2), dpi=100)
        self.fig.patch.set_facecolor(COLORS['bg'])
        self.ax.set_facecolor(COLORS['bg'])
        
        self.loss_line, = self.ax.plot(self.logger.loss_history, color=COLORS['blue'], label='Loss', lw=1.5)
        self.acc_line, = self.ax.plot(self.logger.accuracy_history, color=COLORS['green'], label='Acc', lw=1.5, linestyle='--')
        self.cursor_line = self.ax.axvline(x=0, color=COLORS['amber'], lw=1)
        
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)
        self.ax.grid(color='#1c2e42', linestyle='--', alpha=0.5)
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_sub)
        self.plot_canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_param_inspector(self):
        # List of layers to inspect
        self.layer_views = []
        # architecture: [2, 16, 16, 1] -> 3 layers: 2->16, 16->16, 16->1
        layer_configs = [
            ("Layer 1 (Input \u2192 Hidden)", 16, 2),
            ("Layer 2 (Hidden \u2192 Hidden)", 16, 16),
            ("Layer 3 (Hidden \u2192 Output)", 1, 16)
        ]
        
        for title, rows, cols in layer_configs:
            container = tk.Frame(self.params_frame, bg='#04090f')
            container.pack(fill='x', pady=10)
            
            w_view = MatrixCanvas(container, rows, cols, title, height=max(rows*30 + 50, 80))
            w_view.pack(fill='x')
            self.layer_views.append({'w': w_view})

    def update_ui(self):
        step = self.current_step
        
        # Metrics
        self.lbl_iter.config(text=f"ITER: {step}")
        self.lbl_loss.config(text=f"LOSS: {self.logger.loss_history[step]:.4f}")
        self.lbl_acc.config(text=f"ACC: {self.logger.accuracy_history[step]*100:.1f}%")
        
        # Plot cursor
        self.cursor_line.set_xdata([step, step])
        self.plot_canvas.draw_idle()
        
        self.update_heatmap(step)
        if self.arch_canvas.winfo_width() > 10:
            self.update_architecture()
        self.update_params(step)
        self.update_graph(step)
        
        # Update timeline if it's not being dragged
        self.timeline.set(step)

    def update_heatmap(self, step):
        self.hmap_canvas.delete('all')
        self.hmap_canvas.create_text(10, 10, text="SAMPLE MAP", fill=COLORS['text_dim'], font=FONT_BOLD, anchor='nw')
        
        data = self.logger.heatmap_history[step]
        W, H = self.hmap_canvas.winfo_width(), self.hmap_canvas.winfo_height()
        if W < 10: return
        
        cell_w = (W - 20) // 2
        cell_h = (H - 40) // 5
        
        for i, val in enumerate(data):
            r, c = i % 5, i // 5
            # Color mapping: dark blue -> bright cyan
            cyan_mag = int(val * 215) + 40
            color = f'#00{cyan_mag//2:02x}{cyan_mag:02x}'
            
            x0, y0 = 10 + c * cell_w, 40 + r * cell_h
            x1, y1 = x0 + cell_w - 4, y0 + cell_h - 4
            self.hmap_canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline=COLORS['border'], width=1)

    def update_architecture(self):
        canvas = self.arch_canvas
        canvas.delete('all')
        W, H = canvas.winfo_width(), canvas.winfo_height()
        
        # [2, 16, 16, 1]
        layers = [2, 16, 16, 1]
        layer_colors = [COLORS['text_dim'], COLORS['blue'], COLORS['blue'], COLORS['amber']]
        cx = [W * (i+1)/(len(layers)+1) for i in range(len(layers))]
        
        # Draw connections
        for l in range(len(layers)-1):
            n1 = layers[l]
            n2 = layers[l+1]
            for i in range(n1):
                y1 = H * (i+1)/(n1+1)
                for j in range(n2):
                    y2 = H * (j+1)/(n2+1)
                    is_sel = (self.selected_neuron[0] == l+1 and self.selected_neuron[1] == j)
                    color = COLORS['blue'] if is_sel else '#101a26'
                    width = 1.5 if is_sel else 1
                    canvas.create_line(cx[l], y1, cx[l+1], y2, fill=color, width=width)

        # Draw neurons
        for l, n in enumerate(layers):
            for i in range(n):
                y = H * (i+1)/(n+1)
                is_sel = (self.selected_neuron[0] == l and self.selected_neuron[1] == i)
                color = layer_colors[l]
                r = 7 if is_sel else 5
                outline = COLORS['amber'] if is_sel else COLORS['border']
                width = 2 if is_sel else 1
                
                tags = f"neuron_{l}_{i}"
                canvas.create_oval(cx[l]-r, y-r, cx[l]+r, y+r, fill=color, outline=outline, width=width, tags=tags)

    def on_arch_click(self, event):
        item = self.arch_canvas.find_closest(event.x, event.y)
        tags = self.arch_canvas.gettags(item)
        for tag in tags:
            if tag.startswith("neuron_"):
                parts = tag.split("_")
                l_idx = int(parts[1])
                n_idx = int(parts[2])
                if l_idx > 0: # Only hidden/output layers
                    self.selected_neuron = (l_idx, n_idx)
                    self.update_ui()
                    break

    def update_params(self, step):
        params = self.logger.parameter_history[step]
        for li, layer in enumerate(params):
            if li < len(self.layer_views):
                layer_w_data = [n['weights'] for n in layer]
                self.layer_views[li]['w'].render(layer_w_data)

    def update_graph(self, step):
        canvas = self.graph_canvas
        canvas.delete('all')
        if not self.selected_neuron:
            canvas.create_text(canvas.winfo_width()/2, canvas.winfo_height()/2, 
                              text="Select a neuron to inspect its computation graph.", fill=COLORS['text_dim'])
            return
        
        W, H = canvas.winfo_width(), canvas.winfo_height()
        if W < 10: return
        canvas.create_text(10, 10, text=f"EXECUTION GRAPH: L{self.selected_neuron[0]} N{self.selected_neuron[1]}", 
                          fill=COLORS['cyan'], font=FONT_BOLD, anchor='nw')
        
        li, ni = self.selected_neuron[0]-1, self.selected_neuron[1]
        param = self.logger.parameter_history[step][li][ni]
        
        # Nodes positions
        mid_y = H / 2
        
        # Draw inputs (x) and weights (w)
        num_in = min(len(param['weights']), 4)
        for i in range(num_in):
            y = mid_y - (num_in-1)*25 + i*50
            # Input circles
            canvas.create_oval(30, y-15, 60, y+15, fill=COLORS['raised'], outline=COLORS['border'])
            canvas.create_text(45, y, text=f"x{i}", fill=COLORS['text_dim'], font=FONT_MONO)
            
            # Weight nodes
            w_val = param['weights'][i]
            canvas.create_rectangle(90, y-15, 140, y+15, fill=self.layer_views[0]['w'].get_color(w_val), outline=COLORS['border'])
            canvas.create_text(115, y, text=f"w{i}\n{w_val:.2f}", fill='white', font=FONT_MONO, justify='center')
            
            # Multiplication nodes
            canvas.create_oval(170, y-12, 194, y+12, fill=COLORS['raised'], outline=COLORS['blue'])
            canvas.create_text(182, y, text="*", fill=COLORS['blue'], font=FONT_BOLD)
            
            # Lines
            canvas.create_line(60, y, 90, y, fill=COLORS['border'], arrow=tk.LAST)
            canvas.create_line(140, y, 170, y, fill=COLORS['border'], arrow=tk.LAST)
            canvas.create_line(194, y, 230, mid_y, fill=COLORS['border'], arrow=tk.LAST, smooth=True)

        # Addition node
        canvas.create_oval(230, mid_y-20, 270, mid_y+20, fill=COLORS['raised'], outline=COLORS['green'])
        canvas.create_text(250, mid_y, text="+", fill=COLORS['green'], font=FONT_BOLD)
        
        # Bias node
        b_val = param['bias']
        canvas.create_rectangle(230, mid_y+60, 270, mid_y+90, fill=self.layer_views[0]['w'].get_color(b_val), outline=COLORS['border'])
        canvas.create_text(250, mid_y+75, text=f"b\n{b_val:.2f}", fill='white', font=FONT_MONO, justify='center')
        canvas.create_line(250, mid_y+60, 250, mid_y+20, fill=COLORS['border'], arrow=tk.LAST)

        # Activation node (tanh/relu)
        canvas.create_oval(310, mid_y-20, 360, mid_y+20, fill=COLORS['raised'], outline=COLORS['amber'])
        canvas.create_text(335, mid_y, text="ReLU", fill=COLORS['amber'], font=FONT_MONO)
        canvas.create_line(270, mid_y, 310, mid_y, fill=COLORS['border'], arrow=tk.LAST)
        
        # Output node
        canvas.create_rectangle(400, mid_y-15, 460, mid_y+15, fill=COLORS['raised'], outline=COLORS['cyan'])
        canvas.create_text(430, mid_y, text="OUT", fill=COLORS['cyan'], font=FONT_BOLD)
        canvas.create_line(360, mid_y, 400, mid_y, fill=COLORS['border'], arrow=tk.LAST)

    def on_slider_move(self, val):
        self.current_step = int(val)
        self.update_ui()

    def toggle_play(self):
        self.playing = not self.playing
        self.play_btn.config(text="⏸ PAUSE" if self.playing else "▶ PLAY")
        if self.playing:
            self.playback_loop()

    def step_forward(self):
        self.current_step = (self.current_step + 1) % len(self.logger.loss_history)
        self.update_ui()

    def playback_loop(self):
        if self.playing:
            self.step_forward()
            self.root.after(120, self.playback_loop)

def train_and_log():
    # Simulate some micrograd training to fill the logger
    logger = Logger()
    
    # Simple XOR-like moons problem
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=100, noise=0.1)
    y = y*2 - 1 
    
    # Match demo: MLP(2, [16, 16, 1])
    model = MLP(2, [16, 16, 1])
    
    steps = 100
    for k in range(steps):
        # Forward
        inputs = [list(map(Value, xrow)) for xrow in X]
        scores = list(map(model, inputs))
        losses = [(1 + -yi*scorei).relu() for yi, scorei in zip(y, scores)]
        data_loss = sum(losses) * (1.0 / len(losses))
        reg_loss = 1e-4 * sum((p*p) for p in model.parameters())
        total_loss = data_loss + reg_loss
        
        acc = sum((yi > 0) == (scorei.data > 0) for yi, scorei in zip(y, scores)) / len(y)
        
        # Backward
        model.zero_grad()
        total_loss.backward()
        
        # Update
        lr = 1.0 - 0.9*k/steps
        for p in model.parameters():
            p.data -= lr * p.grad
            
        # Log
        logger.loss_history.append(total_loss.data)
        logger.accuracy_history.append(acc)
        
        # Heatmap: sample frequency simulation
        hmap = [random.random() for _ in range(10)]
        logger.heatmap_history.append(hmap)
        
        # Params
        step_params = []
        for layer in model.layers:
            layer_params = []
            for neuron in layer.neurons:
                layer_params.append({
                    'weights': [w.data for w in neuron.w],
                    'bias': neuron.b.data
                })
            step_params.append(layer_params)
        logger.parameter_history.append(step_params)
        
    return logger

if __name__ == "__main__":
    print("Training micrograd model...")
    logger = train_and_log()
    print("Launching debugger...")
    
    root = tk.Tk()
    app = NeuralReplayDebugger(root, logger)
    root.mainloop()
