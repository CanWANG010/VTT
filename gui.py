from dearpygui.core import *
from dearpygui.simple import *

def button_callback(sender, data):
    set_value("label", "Hello, World!")

with window("Main Window"):
    add_text("点击按钮")
    add_button("点击我", callback=button_callback)
    add_label_text("label", default_value="")

start_dearpygui()