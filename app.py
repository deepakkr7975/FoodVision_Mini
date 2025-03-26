import gradio as gr
import os 
import torch 
from model import create_effnetb2_model
from timeit import default_timer as timer
from typing import Tuple,Dict

class_names=["pizza","steak","sushi"]
effnetb2,effnetb2_transforms=create_effnetb2_model(
    num_classes=len(class_names)
)
effnetb2.load_state_dict(
    torch.load(
        f="pretrained_effnetb2_feature_extractor.pth",
        map_location=torch.device("cpu")

    )

)

#predict function
def predict(img)->Tuple[Dict,float]:
  start_time=timer()
  img=effnetb2_transforms(img).unsqueeze(0) #unsqueeze=add batch dimension on 0th 
  effnetb2.eval()
  with torch.inference_mode():
    pred_probs=torch.softmax(effnetb2(img),dim=1)
  pred_labels_and_probs={class_names[i]:float(pred_probs[0][i]) for i in range (len(class_names))}
  
  end_time=timer()
  pred_time=round(end_time-start_time,4)
  return pred_labels_and_probs,pred_time

## Gradio app
title='FoodVision Mini 🍕🥩🍣'
description="An EfficientnetB2 feature extractor computer vision model to classify images as pizza steak and sushi"
article="Created at Pytorch model deployment"


example_list=[["examples/" + example ] for example in os.listdir("examples")]
#create a gradio demo
demo=gr.Interface(fn=predict,#maps input to output
                  inputs=gr.Image(type="pil"),
                  outputs=[gr.Label(num_top_classes=3,label="Predictions"),
                           gr.Number(label="Prediction time(s)")],
                  examples=example_list,
                  title=title,
                  description=description,
                  article=article)
demo.launch(debug=False,
            share=True)

