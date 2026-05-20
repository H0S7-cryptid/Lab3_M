import tkinter as tk
from ui import SimulationApp

def main():
    root = tk.Tk()
    app = SimulationApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
