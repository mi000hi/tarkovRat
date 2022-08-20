from tkinter import *

root = Tk()
root.title("window title")
root.geometry("1920x1080")
root.attributes('-type', 'dialog')

#root.overrideredirect(True)

#root.wm_attributes('-transparentcolor', 'red')

my_frame = Frame(root, width=200, height=200, bg='red')
my_frame.pack()

label = Label(root, text="this is a label", font=("helvetica", 20))
label.place(x=20, y=20)

label2 = Label(root, text="this is another label", font=("helvetica", 30))
label2.place(x=50, y=20)

root.mainloop()
