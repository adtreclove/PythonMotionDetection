import cv2
import pandas as pd
from datetime import datetime
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.utils import get_color_from_hex
import os


class MotionApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.static_back = None
        self.motion_list = [None, None]
        self.time = []
        self.df = pd.DataFrame(columns=["Start", "End"])
        self.motion_count = 0  # new counter for total detections

        # --- Layout setup ---
        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)

        # Video feed
        self.img = Image()
        layout.add_widget(self.img)

        # Status + counter layout
        status_layout = BoxLayout(orientation='horizontal', size_hint=(1, 0.1), spacing=20)

        self.status_label = Label(
            text="No motion detected",
            color=get_color_from_hex("#00FF00"),  # green
            font_size='20sp',
        )
        self.counter_label = Label(
            text="Detections: 0",
            color=get_color_from_hex("#FFFFFF"),  # white
            font_size='20sp',
        )

        status_layout.add_widget(self.status_label)
        status_layout.add_widget(self.counter_label)
        layout.add_widget(status_layout)

        # Stop button
        btn = Button(text='Stop & Save CSV', size_hint=(1, 0.15))
        btn.bind(on_press=self.stop_and_save)
        layout.add_widget(btn)

        # Schedule updates
        Clock.schedule_interval(self.update, 1.0 / 30.0)
        return layout

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.static_back is None:
            self.static_back = gray
            return

        diff_frame = cv2.absdiff(self.static_back, gray)
        thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
        thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

        contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion = 0
        for contour in contours:
            if cv2.contourArea(contour) < 10000:
                continue
            motion = 1
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

        self.motion_list.append(motion)
        self.motion_list = self.motion_list[-2:]

        if self.motion_list[-1] == 1 and self.motion_list[-2] == 0:
            self.time.append(datetime.now())
            self.motion_count += 1  # 👈 increment counter when new motion starts
        if self.motion_list[-1] == 0 and self.motion_list[-2] == 1:
            self.time.append(datetime.now())

        # --- Update labels ---
        if motion == 1:
            self.status_label.text = "Motion Detected!"
            self.status_label.color = get_color_from_hex("#FF0000")  # red
        else:
            self.status_label.text = "No motion detected"
            self.status_label.color = get_color_from_hex("#00FF00")  # green

        # Update motion counter
        self.counter_label.text = f"Detections: {self.motion_count}"

        # --- Display video feed ---
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
        texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
        self.img.texture = texture

    def stop_and_save(self, *args):
        self.capture.release()
        for i in range(0, len(self.time), 2):
            if i + 1 < len(self.time):
                self.df.loc[len(self.df)] = {"Start": self.time[i], "End": self.time[i + 1]}
        os.makedirs("output", exist_ok=True)
        filename = f"output/Time_of_movements_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        self.df.to_csv(filename, index=False)
        print(f"Motion log saved to: {filename}")
        App.get_running_app().stop()


if __name__ == '__main__':
    MotionApp().run()
