import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

train_data = pd.read_csv("./train.csv").fillna(0)
train_data["Sex"] = train_data["Sex"].map({"male": 1, "female": 0})
test_data = pd.read_csv("./test.csv")
test_data["Sex"] = test_data["Sex"].map({"male": 1, "female": 0})

EPOCHS = 100
BATCH_SIZE = 1
BEST_LOSS = 1e-8


class Train_Dataset(Dataset):
    def __init__(
        self, passengerIds, survivedS, pclassS, sexS, ageS, sibsS, parchS
    ) -> None:
        super().__init__()
        self.passengerIds = passengerIds
        self.survivedS = survivedS
        self.sexS = sexS
        self.pclassS = pclassS
        self.ageS = ageS
        self.sibsS = sibsS
        self.parchS = parchS

    def __len__(self):
        return len(self.passengerIds)

    def __getitem__(self, index):
        passenger_id = torch.tensor([self.passengerIds[index]], dtype=torch.float32)
        survived = torch.tensor([self.survivedS[index]], dtype=torch.float32)
        sex = torch.tensor([self.sexS[index]], dtype=torch.float32)
        passenger_class = torch.tensor([self.pclassS[index]], dtype=torch.float32)
        age = torch.tensor([self.ageS[index]], dtype=torch.float32)
        siblings = torch.tensor([self.sibsS[index]], dtype=torch.float32)
        parch = torch.tensor([self.parchS[index]], dtype=torch.float32)

        x = torch.cat([sex, passenger_class, age * 0.1, siblings, parch], dim=0)

        return {
            "passenger_id": passenger_id,
            "survived": survived,
            "x": x,
        }


# PassengerId,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
class Test_Dataset(Dataset):
    def __init__(self, passengerIds, pclassS, sexS, ageS, sibsS, parchS) -> None:
        super().__init__()
        self.passengerIds = passengerIds
        self.sexs = sexS
        self.pclassS = pclassS
        self.ageS = ageS
        self.sibsS = sibsS
        self.parchS = parchS

    def __len__(self):
        return len(self.passengerIds)

    def __getitem__(self, index):
        passenger_id = torch.tensor([self.passengerIds[index]], dtype=torch.float32)
        sex = torch.tensor([self.sexs[index]], dtype=torch.float32)
        passenger_class = torch.tensor([self.pclassS[index]], dtype=torch.float32)
        age = torch.tensor([self.ageS[index]], dtype=torch.float32)
        siblings = torch.tensor([self.sibsS[index]], dtype=torch.float32)
        parch = torch.tensor([self.parchS[index]], dtype=torch.float32)

        x = torch.cat([sex, passenger_class, age * 0.1, siblings, parch], dim=0)
        return {
            "passenger_id": passenger_id,
            "x": x,
        }


# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
train_dataset = Train_Dataset(
    train_data["PassengerId"],
    train_data["Survived"],
    train_data["Pclass"],
    train_data["Sex"],
    train_data["Age"],
    train_data["SibSp"],
    train_data["Parch"],
)
test_dataset = Test_Dataset(
    test_data["PassengerId"],
    test_data["Pclass"],
    test_data["Sex"],
    test_data["Age"],
    test_data["SibSp"],
    test_data["Parch"],
)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)


class Titanic_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        keep_going = self.relu(self.l1(x))
        keep_going = self.softmax(self.relu(self.l2(keep_going)))
        keep_going = self.l3(keep_going)
        return keep_going


# train


model = Titanic_Model(5 * BATCH_SIZE, 5 * BATCH_SIZE + 3, BATCH_SIZE)
model.load_state_dict(state_dict=torch.load(f="./model.pytorch")["state"])
optim = torch.optim.Adam(model.parameters(), lr=0.008)
loss_fun = nn.MSELoss()


model.train(mode=True)
for epoch in range(EPOCHS + 1):

    for i, data in enumerate(train_loader):
        output = model(data["x"].reshape(shape=(-1, 5 * BATCH_SIZE)))
        loss = loss_fun(output, data["survived"])
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (epoch % 10) == 0:
            print(
                f"epoch:{epoch} || loss:{loss} || output:{output} || survival:{data['survived']}"
            )

        if loss < BEST_LOSS and loss < torch.load(f="./model.pytorch")["loss"]:
            model_state = {"loss": loss, "state": model.state_dict()}
            torch.save(obj=model_state, f="./model.pytorch")

        # if loss < BEST_LOSS :
        #     model_state = {"loss": loss, "state": model.state_dict()}
        #     torch.save(obj=model_state, f="./model.pytorch")
        #     print('saving best model 🤔')


model.eval()
with torch.no_grad():
    y_preds = []
    ids = []
    for i, data in enumerate(test_loader):

        output = model(data["x"].reshape(shape=(-1, 5 * BATCH_SIZE)))
        if output.item() > 0.55:
            y_preds.append(int(1))
        else:
            y_preds.append(int(0))
        ids.append(int(data["passenger_id"].item()))

    df = pd.DataFrame({"PassengerId": ids, "Survived": y_preds}).sort_values(
        by="PassengerId", ascending=True
    )
    print(df.head())
    df.to_csv("./pred.csv", index=False)
