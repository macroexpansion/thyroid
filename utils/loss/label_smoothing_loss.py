import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLossWithLabelSmoothing(nn.Module):
    def __init__(self, classes=3, epsilon=0.1, dim=-1):
        super(CrossEntropyLossWithLabelSmoothing, self).__init__()
        assert 0.0 <= epsilon <= 1.0

        self.confidence = 1.0 - epsilon
        self.epsilon = epsilon
        self.classes = classes
        self.dim = dim

    def forward(self, outputs, target, reduction="mean", phase="train"):
        """
        outputs: (batch_size, n_classes)
        target: (batch_size)
        """
        # outputs = outputs.log_softmax(dim=self.dim)
        outputs = F.log_softmax(outputs, dim=self.dim)
        if phase == "train":
            with torch.no_grad():
                smoothed_label = torch.zeros_like(outputs)
                smoothed_label.fill_(self.epsilon / (self.classes - 1))
                smoothed_label.scatter_(1, target.unsqueeze(1), self.confidence)
                # print(smoothed_label)
        else:
            with torch.no_grad():
                smoothed_label = torch.zeros_like(outputs)
                smoothed_label.fill_(0)
                smoothed_label.scatter_(1, target.unsqueeze(1), 1.0)

        if reduction == "KL":
            return F.kl_div(outputs, smoothed_label, reduction="sum")

        if reduction == "mean":
            return torch.mean(torch.sum(-smoothed_label * outputs, dim=self.dim))

        if reduction == "sum":
            return torch.sum(torch.sum(-smoothed_label * outputs, dim=self.dim))


if __name__ == "__main__":
    label_epsilon = CrossEntropyLossWithLabelSmoothing(epsilon=0.05, classes=2)
    # out = torch.tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0]]).type(torch.float)
    # tar = torch.tensor([2, 1, 0]).type(torch.long)
    out = torch.tensor([[0, 1], [1, 0]]).type(torch.float)
    tar = torch.tensor([1, 0]).type(torch.long)
    print(label_epsilon(out, tar, reduction="mean", phase="train"))
    # print(nn.CrossEntropyLoss(reduction="mean")(out, tar))
