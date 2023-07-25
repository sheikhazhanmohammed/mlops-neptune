import neptune.new as neptune

APIKEY = "YOUR API KEY"

model_version = neptune.init_model_version(
    model="DVC-MOD",
    project="r3yna/dogsvscats",
    api_token=APIKEY, # your credentials
)

model_version["model"].upload("./classificationModel.pt")
model_version["validation/dataset"].track_files("./data/")
model_version.change_stage("staging")

model_version.stop()