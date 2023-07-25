import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import neptune.new as neptune

APIKEY = "YOUR API KEY"

st.markdown('<h1 style="color:white;">Demonstrate MLOPs using Neptune</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">Classifying images of cats and dogs</h2>', unsafe_allow_html=True)

upload= st.file_uploader('Insert image for classification', type=['png','jpg'])


c1, c2= st.columns(2)
if upload is not None:
   #loading the image
   image= Image.open(upload)
   c1.header('Input Image')
   c1.image(image)
   c2.header("Prediction")
   c2.write("Starting prediction")
   image = image.convert('RGB')
   image = image.resize((224,224))
   transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
   inputtensor = transform(image)
   inputbatch = inputtensor.unsqueeze(0)
   c2.write("Loaded image")
   #loading the model from neptune
   c2.write("Downloading model")
   model = neptune.init_model(model="DVC-MOD",project="r3yna/dogsvscats",api_token=APIKEY)
   model_versions_df = model.fetch_model_versions_table().to_pandas()
   latestModelId = str(model_versions_df["sys/id"][0])
   model.stop()
   model = neptune.init_model_version(with_id = latestModelId, project="r3yna/dogsvscats",api_token=APIKEY)
   model["model"].download("./classification.pt")
   model.stop()
   #loading the model and doing predictions
   c2.write("Loading model")
   model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
   model.fc = torch.nn.Linear(512,2)
   model.to("cpu")
   model.load_state_dict(torch.load("./classification.pt"))
   model.eval()
   c2.write("Running image through model")
   with torch.no_grad():
    prediction = model(inputbatch)
   _, predictionT = torch.max(prediction, dim=1)
   predictionT = predictionT[0].cpu().numpy()
   if predictionT==0:
    c2.write("Final prediction: Cat")
   else:
     c2.write("Final prediction: Dog")

