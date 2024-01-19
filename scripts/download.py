
import rich
from roboflow import Roboflow

from rich.prompt import Prompt

from user_secrets import API_KEY

rf = Roboflow(api_key=API_KEY)
# project = rf.workspace("jacobs-workspace").project("sku-110k")
# dataset = project.version(4).download("yolov8")

project = rf.workspace("fyp-ormnr").project("on-shelf-stock-availability-ox04t")
dataset = project.version(5).download("yolov8")

console = rich.console.Console()
console.print("Now you need to go to ... and edit the paths to be accurate! Else there are gonna be errors when starting the training process.")