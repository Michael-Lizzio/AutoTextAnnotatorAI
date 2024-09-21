import os
from tkinter import Tk, Canvas, PhotoImage, NW
from PIL import Image, ImageTk


class ImageViewer:
    def __init__(self, root, output_dir):
        self.root = root
        self.output_dir = output_dir
        self.images = self.load_images(output_dir)
        self.index = 0

        self.canvas = Canvas(root, width=root.winfo_screenwidth(), height=root.winfo_screenheight())
        self.canvas.pack()

        self.display_image()
        self.root.bind("<Left>", self.prev_image)
        self.root.bind("<Right>", self.next_image)
        self.root.bind("<Escape>", self.quit_app)  # Bind the Esc key to quit_app method

    def load_images(self, output_dir):
        image_files = [f for f in os.listdir(output_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
        images = []
        for image_file in image_files:
            path = os.path.join(output_dir, image_file)
            image = Image.open(path)
            images.append(ImageTk.PhotoImage(image))
        return images

    def display_image(self):
        if self.images:
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=NW, image=self.images[self.index])
            self.root.title(f"Image Viewer: {self.index + 1}/{len(self.images)}")

    def next_image(self, event):
        if self.images:
            self.index = (self.index + 1) % len(self.images)
            self.display_image()

    def prev_image(self, event):
        if self.images:
            self.index = (self.index - 1) % len(self.images)
            self.display_image()

    def quit_app(self, event):
        self.root.destroy()  # Close the window


if __name__ == "__main__":
    output_dir = 'output_photos'
    root = Tk()
    root.attributes("-fullscreen", True)
    app = ImageViewer(root, output_dir)
    root.mainloop()
