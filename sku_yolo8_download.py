
from roboflow import Roboflow

from rich.prompt import Prompt

api_key = Prompt.ask("Enter roboflow API Key", password=True)
rf = Roboflow(api_key=api_key)
project = rf.workspace("jacobs-workspace").project("sku-110k")
dataset = project.version(4).download("yolov8")
