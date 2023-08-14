import multiprocessing as mp
import random
import signal
import time
import tkinter as tk


def video_processing(q):
    while True:
        start_time = time.time()

        # Simulate video frame processing.
        time.sleep(random.uniform(0.01, 0.04))

        end_time = time.time()
        fps = 1.0 / (end_time - start_time)
        q.put(fps)


class StreamInfoUI(mp.Process):
    def __init__(self):
        super().__init__()
        self.fps_label = None
        self.root = None
        self.fps_queue = mp.Queue(maxsize=1)
        self.exit_event = mp.Event()

    def get_fps_queue(self):
        return self.fps_queue

    def exit(self):
        print("ui exit is called")
        self.exit_event.set()

    def update_fps(self):
        while not self.fps_queue.empty():
            fps_value = self.fps_queue.get()
            self.fps_label.config(text=f"FPS: {fps_value:.2f}")
            if self.exit_event.is_set():
                print("ui exit is set")
                return
        if not self.exit_event.is_set():
            self.root.after(100, self.update_fps)

    def quit(self, event):
        print("exiting on command")
        self.root.quit()

    def run(self):
        self.root = tk.Tk()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.root.title("FPS Display")
        self.root.geometry("300x100")
        self.fps_label = tk.Label(self.root, text="FPS: ---")
        self.fps_label.pack(pady=20)

        self.update_fps()
        self.root.mainloop()
        self.quit(None)


if __name__ == "__main__":
    gui_proc = StreamInfoUI()
    q = gui_proc.get_fps_queue()

    video_proc = mp.Process(target=video_processing, args=(q,))
    try:
        video_proc.start()
        gui_proc.start()
    except KeyboardInterrupt:
        print("exit on keyboard ")
        gui_proc.exit()
        video_proc.terminate()
    video_proc.join()
    gui_proc.join()
